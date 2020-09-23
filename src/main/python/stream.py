import json
import redis
import pandas as pd

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

from kimi_common.kv.redis import RedisClient
from kimi_common.ml.sklearn import SklearnBinaryClassificationTrainer

cate_features = [
                ###
                # 商品侧
                #
                'note_type',
                'real_source',
                'label_level1_id',
                ###
                # 用户侧
                #
                'user_clk_label_topn',
                'user_active_date_7d',
                'gender',
                'age',
                'user_active_date_14d',
                ###
                # 上下文
                #
                ]
number_features = [
                ###
                # 商品侧
                #
                'play_number',
                'praise_number',
                'share_number',
                'comment_number_x',
                'favorite_number',
                'video_public_release_days',
]

spark = SparkSession \
    .builder \
    .appName("StructuredNetworkWordCount") \
    .config("spark.default.parallelism", "800") \
    .getOrCreate()

def quiet_logs( sc ):
  logger = sc._jvm.org.apache.log4j
  logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
  logger.LogManager.getLogger("kafka").setLevel( logger.Level.ERROR )

quiet_logs(spark.sparkContext)

df =  spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "10.15.0.106:9092") \
    .option('auto.offset.reset',True) \
    .option('enable.auto.commit',True) \
    .option('minPartitions',8)\
    .option("subscribe", "test") \
    .load() \
    .selectExpr("CAST(value AS STRING) as value")


def process(df,batch_id):
    print(f'batch_id :{batch_id}')
    def get_user_id_from_key(user_id_key):
        return str(user_id_key).split(':')[1]


    def get_user_by_redis(iterator): 
        user_redis_key_list =[]
        for i in iterator:
            value = json.loads(i.asDict().get('value'))
            user_id = value.get('user_id',None)
            if user_id is not None:
                user_redis_key_list.append(f'user:{user_id}')

        redis_client = RedisClient('10.15.0.106',63790)
        ret = redis_client.batch_get(user_redis_key_list)
        predict_data =[]
        for user_id_key,redis_result in ret:
            try:
                user_row =  json.loads(redis_result)
                if isinstance(user_row,list):
                    user_row = user_row[0]
                user_row['user_id'] = get_user_id_from_key(user_id_key)
                predict_data.append(user_row)
            except Exception  as e:
                print(e)
        df =  pd.DataFrame(predict_data)
        pdf = model.predict(df)
        pdf = pdf.sort_values(['user_id','predict1'],ascending=[1,0])
        pdf = pdf.drop_duplicates(['user_id','note_id'],keep='first')
        pdf = pdf.groupby('user_id').head(3)
        pdf = pdf.groupby('user_id').agg(top=('note_id',lambda row: list(row.unique())))
        user_reco_key_list =  []
        user_reco_value_list = []
        user_topn_list = pdf.reset_index().to_dict('records')
        for i in user_topn_list:
            user_id = i['user_id']
            user_reco_key_list.append(f'user:reco:{user_id}')
            user_reco_value_list.append( [int(i) for i in  i['top']])
        redis_client.batch_set(user_reco_key_list,user_reco_value_list)
    
    df.rdd.foreachPartition(get_user_by_redis)
  
    print('end batch')
    


query = df.writeStream.foreachBatch(process).trigger(processingTime='500 millisecond').start()

query.awaitTermination()