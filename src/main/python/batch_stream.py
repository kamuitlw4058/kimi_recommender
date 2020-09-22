import json
import redis

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split
from pyspark.sql.types import StructType,DoubleType,IntegerType,ArrayType,StringType,StructField
from pyspark.ml import Pipeline,PipelineModel
from pyspark.sql import Window
from pyspark.sql.functions import row_number,from_json,explode
from pyspark.sql.functions import desc,udf
from pyspark.sql import functions as f
from pyspark.sql import Row
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils

from kimi_common.kv.redis import RedisClient

sameModel = PipelineModel.load('model/spark-logistic-regression-model')

spark = SparkSession \
    .builder \
    .appName("StructuredNetworkWordCount") \
    .config("spark.default.parallelism", "800") \
    .getOrCreate()
    
ssc=StreamingContext(spark.sparkContext,5)
kafkaStreams = KafkaUtils.createDirectStream(ssc,['test'],kafkaParams={"metadata.broker.list": '10.15.0.106:9092'})
def get_user_features(input_list):
    user_dict ={}
    for k,v in input_list:
        kafka_value = json.loads(v)
        user_id = kafka_value.get('user_id',None)
        if user_id is not None:
            user_dict[str(user_id)] = kafka_value
    
    def get_user_id(value):
        return value['user_id']

    user_redis_key_list = [f'user:{get_user_id(i)}'  for i in list(user_dict.values())]
    redis_client = RedisClient('10.15.0.106',63790)
    ret = redis_client.batch_get(user_redis_key_list)
    for user_id,redis_result in ret:
        user_dict[str(user_id).split(':')[1]].update(json.loads(redis_result))
    print(user_dict)
    return iter(user_dict.values())
 
    



def getSparkSessionInstance(sparkConf):
    if ("sparkSessionSingletonInstance" not in globals()):
        globals()["sparkSessionSingletonInstance"] = SparkSession \
            .builder \
            .config(conf=sparkConf) \
            .getOrCreate()
    return globals()["sparkSessionSingletonInstance"]

def process(time, rdd):
    print("========= %s =========" % str(time))
    try:
        # Get the singleton instance of SparkSession
        spark = getSparkSessionInstance(rdd.context.getConf())

        # Convert RDD[String] to RDD[Row] to DataFrame
        rowRdd = rdd.map(lambda w: Row(**w))
        wordsDataFrame = spark.createDataFrame(rowRdd)
        probability = sameModel.transform(wordsDataFrame)
        pdf = probability.toPandas()
        pdf['probability'] = pdf['probability'].apply(lambda row: row[1])
        pdf =pdf.sort_values(['user_id','probability'],ascending=[1,0])
        pdf = pdf.drop_duplicates(['user_id','note_id'],keep='first')
        pdf = pdf.groupby('user_id').head(3)
        pdf = pdf.groupby('user_id').agg(top=('note_id',lambda row: list(row.unique())))
        print(pdf)


    except:
        pass


user_features_rdd =  kafkaStreams.mapPartitions(get_user_features)
user_features_rdd.foreachRDD(process)

ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate

