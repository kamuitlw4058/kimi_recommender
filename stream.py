from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split
from pyspark.sql.types import StructType,DoubleType,IntegerType,ArrayType
from pyspark.ml import Pipeline,PipelineModel
from pyspark.sql import Window
from pyspark.sql.functions import row_number
from pyspark.sql.functions import desc,udf
from pyspark.sql import functions as f

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
lines = spark\
    .readStream \
    .schema(userSchema) \
    .json("data/json")

sameModel = PipelineModel.load('model/spark-logistic-regression-model')
probability = sameModel.transform(lines)
def agg_user(df,batch_id):
    pdf = df.toPandas()
    pdf['probability'] = pdf['probability'].apply(lambda row: row[1])
    pdf =pdf.sort_values(['user_id','probability'],ascending=[1,0])
    pdf = pdf.drop_duplicates(['user_id','note_id'],keep='first')
    pdf = pdf.groupby('user_id').head(3)
    pdf = pdf.groupby('user_id').agg(top=('note_id',lambda row: list(row.unique())))
    print(pdf)



query = probability.select('user_id','note_id','probability').writeStream.foreachBatch(agg_user).start()



query.awaitTermination()