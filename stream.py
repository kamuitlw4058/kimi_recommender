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
#     .load()
lines = spark\
    .readStream \
    .schema(userSchema) \
    .json("data/json")

sameModel = PipelineModel.load('model/spark-logistic-regression-model')
probability = sameModel.transform(lines)

# Split the lines into words
# words = lines.select(
#    explode(
#        split(lines.value, " ")
#    ).alias("word")
# )
query =probability.writeStream \
    .format("console") \
    .start()
# Generate running word count
#wordCounts = lines.groupBy("note_type").count()
# result = probability.groupBy('user_id').agg(f.collect_list('probability').alias('probability_list'),
#                                  f.collect_list('note_id').alias('note_id_list')
#                             )


# result = ordered_probability.groupBy('user_id').head(20).writeStream \
#     .format("console") \
#     .start()

# def top_id(probability_list,note_id_list):
#     ziped_list =  zip(probability_list, note_id_list)
#     sorted_list =sorted(ziped_list, key=lambda x: float(x[0].toArray().tolist()[1]),reverse=True)
#     return [ note_id for probability,note_id in  sorted_list[:10]]

# top_id_udf = f.UserDefinedFunction(top_id, returnType=ArrayType(IntegerType()))


# #
# # top_id_udf = udf(top_id, ArrayType(DoubleType()),ArrayType(IntegerType()))

# query = result.select("user_id",  top_id_udf(result['probability_list'],result['note_id_list']))\
#     .writeStream \
#     .outputMode('complete') \
#     .format("console") \
#     .start()

query.awaitTermination()