from etl import get_orig_data
from trainer.spark_trainer import SparkBinaryClassificationTrainer

df = get_orig_data(recum=False)
print(df.columns)
df['user_clk_label_topn'] = df['user_clk_label_topn'].astype(str)
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
keep_list =[]


trainer =  SparkBinaryClassificationTrainer()
trainer.train(df,cate_features,number_features,keep_list)