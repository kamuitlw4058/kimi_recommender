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
from kimi_common.kv.redis import RedisClient




spark = SparkSession \
    .builder \
    .appName("StructuredNetworkWordCount") \
    .config("spark.default.parallelism", "800") \
    .getOrCreate()
    
    #.config("spark.driver.extraJavaOptions", "-Xms256m -Xmx4069m")\

userSchema = StructType().add("note_type", "string") \
                        .add('real_source','string') \
                        .add('label_level1_id','string') \
                        .add('user_clk_label_topn','string') \
                        .add('user_active_date_7d','string') \
                        .add('gender','string') \
                        .add('age','string') \
                        .add('user_active_date_14d','string') \
                        .add('play_number','double') \
                        .add('praise_number','double') \
                        .add('share_number','double') \
                        .add('comment_number_x','double') \
                        .add('favorite_number','double') \
                        .add('video_public_release_days','double') \
                        .add('note_id','integer') \
                        .add('user_id','integer') 


# Create DataFrame representing the stream of input lines from connection to localhost:9999
# lines = spark \
#     .readStream \
#     .format("socket") \
#     .option("host", "localhost") \
#     .option("port", 9999) \
#     .load()x
# lines = spark\
#     .readStream \
#     .schema(userSchema) \
#     .json("data/json")
#     spark
df =  spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "10.15.0.106:9092") \
    .option("subscribe", "test") \
    .load() 
    #.option('spark.sql.streaming.forceDeleteTempCheckpointLocation',True)\
    #.option('kafkaConsumer.pollTimeoutMs',5000)\
        #
#df= df .selectExpr("CAST(value AS STRING)")



pool = redis.ConnectionPool(host='10.15.0.106', port=63790, db=0)

def get_by_redis(user_id):
    redis_client = redis.StrictRedis(connection_pool=pool)
    #print(json.loads(redis_client.get(f'user:{user_id}')))
    return json.loads(redis_client.get(f'user:{user_id}'))
    
# userSchema

# add("note_type", "string") \
# .add('real_source','string') \
# .add('label_level1_id','string') \
# .add('user_clk_label_topn','string') \
# .add('user_active_date_7d','string') \
# .add('gender','string') \
# .add('age','string') \
# .add('c','string') \
# .add('play_number','double') \
# .add('praise_number','double') \
# .add('share_number','double') \
# .add('comment_number_x','double') \
# .add('favorite_number','double') \
# .add('video_public_release_days','double') \
# .add('note_id','integer') \
# .add('user_id','integer') 

get_user_by_redis = udf(get_by_redis, returnType=StructType([
    StructField("note_type", DoubleType()),
    #StructField("favorite_number", DoubleType()),
    StructField("real_source", DoubleType()),
    StructField("label_level1_id", DoubleType()),
     StructField("praise_number", DoubleType()),
    StructField("note_id", IntegerType()),
    # StructField("user_clk_label_topn", DoubleType()),
    # StructField("user_active_date_7d", DoubleType()),
    # StructField("gender", StringType()),
    # StructField("age", StringType()),
    # StructField("user_active_date_14d", DoubleType())
]))

# get_user_by_redis = udf(get_by_redis, returnType=userSchema)

schema = StructType([StructField("user_id", StringType())])
df = df.select(from_json( df.value.cast('string'), schema).alias("json"))
df = df.withColumn('user_id', df["json"].getItem('user_id'))
df = df.withColumn("user_features", get_user_by_redis(df["user_id"]))
df = df.withColumn('note_type', df["user_features"].getItem('note_type'))
df = df.withColumn('real_source', df["user_features"].getItem('real_source'))
df = df.withColumn('label_level1_id', df["user_features"].getItem('label_level1_id'))
df = df.withColumn('praise_number', df["user_features"].getItem('praise_number'))
df = df.withColumn('note_id', df["user_features"].getItem('note_id'))

#df = df.withColumn('test2', df["json"].getItem('test2'))


sameModel = PipelineModel.load('model/spark-logistic-regression-model')
probability = sameModel.transform(df)
def agg_user(df,batch_id):
    pdf = df.toPandas()
    pdf['probability'] = pdf['probability'].apply(lambda row: row[1])
    pdf =pdf.sort_values(['user_id','probability'],ascending=[1,0])
    pdf = pdf.drop_duplicates(['user_id','note_id'],keep='first')
    pdf = pdf.groupby('user_id').head(3)
    pdf = pdf.groupby('user_id').agg(top=('note_id',lambda row: list(row.unique())))
    print(pdf)

result = probability

query = result.select('user_id','note_id','probability').writeStream.format('console').start()
# query.awaitTermination()
#query = result.select('user_id','note_id','probability').writeStream.foreachBatch(agg_user).start()
#query = result.select('user_id','probability').writeStream.foreachBatch(agg_user).start()
query.awaitTermination()