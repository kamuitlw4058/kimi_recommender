from pyspark.ml import Pipeline,PipelineModel
sameModel = PipelineModel.load('model/spark-logistic-regression-model')

for i in sameModel.stages:
    print(i)
