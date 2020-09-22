import json

from etl import get_orig_data

from kimi_common.ml.sklearn import SklearnBinaryClassificationTrainer


df = get_orig_data(recum=False)
print(f'orig len:{ len(df)}')
df = df.sort_values(by=['user_id','note_id','label'],ascending=[1,1,1])
df = df.drop_duplicates(subset=['user_id','note_id'],keep='last')
#df = df.groupby(['user_id','label']).head(30)
print(f'after drop size:{len(df)}')
print(df.columns)
df['user_clk_label_topn'] = df['user_clk_label_topn'].astype(str)
df[['play_number','praise_number','share_number','comment_number_x','favorite_number','video_public_release_days']] = df[['play_number','praise_number','share_number','comment_number_x','favorite_number','video_public_release_days']].fillna(0)
# d = df.to_dict('records')[0:10000]
# json_str =  json.dumps(d)
# with open('data/test.json','w') as f:
#     f.write(json_str)
# exit()
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
                # 'user_clk_label_topn',
                # 'user_active_date_7d',
                # 'gender',
                # 'age',
                # 'user_active_date_14d',
                ###
                # 上下文
                #
                ]
number_features = [
                ###
                # 商品侧
                #
                # 'play_number',
                'praise_number',
                # 'share_number',
                # 'comment_number_x',
                # 'favorite_number',
                # 'video_public_release_days',

]
keep_list =['user_id','note_id']



trainer =  SklearnBinaryClassificationTrainer('model')
#trainer.train(df,cate_features,number_features,keep_list)
trainer.predict(df,cate_features,number_features)




