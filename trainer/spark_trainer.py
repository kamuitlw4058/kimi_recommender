import time
import pandas as pd
from sklearn.utils import shuffle as shuffle

from pyspark.ml import Pipeline,PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, RegexTokenizer,StringIndexer,VectorIndexer, StandardScaler,VectorAssembler,OneHotEncoder
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.types import DoubleType, IntegerType



from  pyspark.sql import SparkSession

class SparkBinaryClassificationTrainer():
    def __init__(self):
        self.spark = SparkSession \
            .builder \
            .appName("Train model") \
            .getOrCreate()
    
    def negtive_sample(self,df,negtive_sample_ratio,true_value,need_shuffle=True):
        true_df = df[df.label==true_value]
        false_df = df[df.label!= true_value]
        false_orig_size = len(false_df)
        false_df = false_df.sample(len(true_df)* negtive_sample_ratio)
        sampled_df = pd.concat([true_df,false_df])
        if need_shuffle:
            sampled_df = shuffle(sampled_df)
        sample_rate = len(false_df) /false_orig_size
        print(f'true size:{len(true_df)} false size:{false_orig_size} negtive sample ratio:{negtive_sample_ratio} final sample rate:{sample_rate}')
        return sampled_df,sample_rate

    def data_split(self,df,test_split_number):
        train_df = df[:-test_split_number]
        test_df = df[-test_split_number:]
        return train_df,test_df


    def get_cat_features(self,cate_col):
        string_index_col = f'{cate_col}_index'
        onehot_col =f'{cate_col}_onehot_vec'
        si = StringIndexer(inputCol=cate_col,outputCol=string_index_col,handleInvalid='keep')
        oh = OneHotEncoder(inputCol=string_index_col,outputCol=onehot_col)
        return [si,oh],onehot_col

    


    def train(self,df,cate_features,number_features,keep_list,nagtive_sample=2,test_split_mode='last',test_split_number=10000,true_value=1,label='label'):
        df = df[cate_features + number_features+ keep_list +  [label]]
        sampled_df , sample_rate = self.negtive_sample(df,nagtive_sample,true_value)
        train_df,test_df  = self.data_split(sampled_df,test_split_number)

        training = self.spark.createDataFrame(train_df)
        print(f'training count:{len(train_df)}')
        print(f'test count:{len(test_df)}')

        cate_pipline = []
        cate_onehot_output_cols = []
        for i in cate_features:
            cat_pipline_item,output_col = self.get_cat_features(i)
            cate_pipline.extend(cat_pipline_item)
            cate_onehot_output_cols.append(output_col)

        number_assembler = VectorAssembler(
            inputCols=number_features,
            outputCol="number_features")
        scaler = StandardScaler(inputCol="number_features", outputCol="number_features_scaled",
                                withStd=True, withMean=False)
        assembler = VectorAssembler(
            inputCols=cate_onehot_output_cols + ['number_features_scaled'],
            outputCol="features")
        lr = LogisticRegression(maxIter=10, regParam=0.001)

        pipeline = Pipeline(stages=  cate_pipline + [number_assembler , scaler , assembler,lr])

        model = pipeline.fit(training)
        model.write().overwrite().save("model/spark-logistic-regression-model")

        lrmodel = model.stages[-1]
        # Prepare test documents, which are unlabeled (id, text) tuples.
        test = self.spark.createDataFrame(test_df)

        # Make predictions on test documents and print columns of interest.
        start =time.time()
        prediction = model.transform(test)
        end = time.time()
        print(f'elasped test:{end-start}')

        trainingSummary = lrmodel.summary

        trainingSummary.roc.show()
        print(f'train AUC: {  str(trainingSummary.areaUnderROC)}')

        # Instantiate metrics object
        metrics = BinaryClassificationMetrics(prediction
            .select( "probability", "label")
            .rdd.map(lambda r: (float(r[0][1]),float(r[1]))))


        # Area under ROC curve
        print(f'test AUC: {metrics.areaUnderROC}')