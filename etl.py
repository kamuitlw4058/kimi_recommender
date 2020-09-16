import json
import os
from datetime import datetime,timedelta

import pandas as pd
import pandas_profiling
import pyspark
import numpy as np


from kimi_common.database.client.clients import Clients
from kimi_common.utils.pandas.data_convert import convert_datetime_to_date
from kimi_common.utils.pd_utils import apply_if_col_not_exists

from conf.type_mapper import (get_device_type_name,
                              get_ext1_name, get_ext2_name)
from conf.db import db_params

def simple_group_agg(df,col,label_col='label',agg_funcs=['count','mean'],by='mean'):
    if isinstance(col,str):
        col = [col]
    if isinstance(agg_funcs,dict):
        grouped_label = df.groupby(col)
        grouped_df = grouped_label.agg(**agg_funcs)
    else:
        grouped_label = df.groupby(col)[label_col]
        grouped_df = grouped_label.agg(agg_funcs)
    #grouped_df = grouped_label.agg(agg_funcs)
    columns = {}
  

    if by is not None:
        ret = grouped_df.sort_values(by=by,ascending=False)
    else:
        ret = grouped_df.sort_index()
    
    for func_name in  agg_funcs:
        if isinstance(col,list):
            col = '_'.join(col)
        columns[func_name] = f'{col}_{label_col}_{func_name}'
    ret = ret.rename(columns=columns)
    return ret 



def get_label_wide(client):
    label_level_df  = client.read_sql("select id as label_id, label_leave, parent_id,label_name  from t_fw_back_label",cache_pickle_path='data/note_label_level.pkl')
    label_dict_list = label_level_df.to_dict('records')
    label_level2tolevel1_dict = {}
    for i in label_dict_list:
        if i['label_leave'] == 2:
            label_level2tolevel1_dict[i['label_id']] = i['parent_id']

        
    label_level_df = apply_if_col_not_exists(label_level_df,'label_level1_id',lambda row: label_level2tolevel1_dict.get(row['label_id'],row['label_id']))
    print(label_level_df)
    return label_level_df

def get_note_wide(client):
    note_df  = client.read_sql("select * from face_value_circle_video",cache_pickle_path='data/note.pkl')
    note_df = note_df.rename(columns={'id':'note_id'})
    print(note_df)


    note_label_df  = client.read_sql("select note_id,label_id from t_fw_back_note_label",cache_pickle_path='data/note_label.pkl')
    print(note_label_df)

    label_df = get_label_wide(client)

    note_label_df =  note_label_df.merge(label_df,on='label_id',how ='left')
    note_label_df = note_label_df.drop_duplicates(['note_id','label_id'], 'last')

    note_df =  note_df.merge(note_label_df,on='note_id',how='left')
    note_df = note_df.rename(columns={'id':'note_id','user_id':'note_owner'})
    return note_df

def get_user_wide(client):
    user_df = client.read_sql("select * from t_fw_user",cache_pickle_path='data/user.pkl')
    user_df = user_df.rename(columns={'user_id':'temporary_id'})

    # user_note_interaction_df  = client.read_sql("select user_id,target_id as note_id,ope_type,create_time from t_fw_user_note_collection",cache_pickle_path='data/user_note_interaction.pkl')
    # print(user_note_interaction_df)
    # user_note_interaction_df = user_note_interaction_df.merge(note_label_df,on='note_id',how='left')
    # print(user_note_interaction_df)

    # grouped_user_note_interaction_df=simple_group_agg(user_note_interaction_df,['user_id','note_id'],'ope_type',agg_funcs=['count','sum'],by='count')
    # print(grouped_user_note_interaction_df)
    # print(simple_group_agg(user_note_interaction_df,'label_level1_id','ope_type',agg_funcs=['count','sum'],by='count'))

    return user_df

def get_log(client):
    log_df = client.read_sql("select * from face_value_circle_video_play_info",cache_pickle_path='data/log.pkl')
    print(log_df)
    return log_df


def get_orig_data(recum=False ):
    final_path = 'data/final.pkl'
    output_path = 'data/output.pkl'
    if not recum and  os.path.exists(output_path):
        return pd.read_pickle(output_path)
    if recum or   not os.path.exists(final_path) :
        print('start get by db....')
        client = Clients(clients_params = db_params).get_client()
        log_df = get_log(client)
        note_df = get_note_wide(client)
        user_df = get_user_wide(client)
        
        log_df = log_df.merge(note_df,on=['note_id'],how='left')
        log_df = log_df.merge(user_df,on=['temporary_id'],how='left')
        print(log_df)
        log_df['label'] = log_df['log_type'] -1 
        total_df = log_df
        total_df.to_pickle(final_path)
    else:
        total_df = pd.read_pickle(final_path)

    print('start get extend_data')
    total_df = apply_if_col_not_exists(total_df,'来源类型',lambda row: get_ext1_name(row['ext1_x']))
    total_df = apply_if_col_not_exists(total_df,'播放来源',lambda row: get_ext2_name(row['ext2_x']))
    total_df = apply_if_col_not_exists(total_df,'设备类型',lambda row: get_device_type_name(row['device_type']))
    convert_datetime = convert_datetime_to_date()
    total_df = apply_if_col_not_exists(total_df,'date',lambda row: convert_datetime(row['create_time_x']))
    def get_public_release_days(row):
        try:
            ret = datetime.now().date() -  convert_datetime(row['release_time'])
        except:
            return 100
        return int(ret.days)
    total_df = apply_if_col_not_exists(total_df,'video_public_release_days',get_public_release_days )


    print(simple_group_agg(total_df,'来源类型'))
    print(simple_group_agg(total_df,'播放来源'))
    print(simple_group_agg(total_df,'设备类型'))
    print(simple_group_agg(total_df,'date',by=None))
    print(simple_group_agg(total_df,'temporary_id',by='count'))
    print(simple_group_agg(total_df,['temporary_id','date'],by='count'))
    print(list(total_df.columns))
    total_df.to_pickle(final_path)
    total_df = total_df[['note_type','favorite_number',
                        'real_source','comment_number_x','share_number_x',
                        'praise_number','play_number','play_duration','video_duration','note_duration',
                        'birthday','gender','log_type',
                        'video_public_release_days','label_level1_id','note_id','temporary_id','date','label_id']]
    total_df.rename(columns={'temporary_id':'user_id'},inplace=True)
    total_df.dropna(subset=['note_type'],inplace=True)

    print(total_df)
    total_df['label'] = total_df['log_type'] -1
    note_history_df =simple_group_agg(total_df,'note_id',by='count').reset_index()
    print(note_history_df)
    total_df = total_df.merge(note_history_df,on='note_id',how='left')
    print(total_df.columns)
    print(total_df)
    print(total_df[['play_duration','note_duration']][total_df['play_duration'] > total_df['note_duration']])
    total_df = apply_if_col_not_exists(total_df,'video_play_percent',lambda row: 0.0 if  row.get('note_duration',86400) == 0 else row.get('play_duration',0)/row.get('note_duration',86400))
    note_history_df =simple_group_agg(total_df,'note_id',label_col='note_duration').reset_index()
    print(note_history_df)
    note_history_df =simple_group_agg(total_df,'note_id',label_col='video_play_percent',by='count').reset_index()
    print(note_history_df)
    total_df = total_df.merge(note_history_df,on='note_id',how='left')
    print(total_df.columns)
    print(total_df)


    user_histroy_df =simple_group_agg(total_df,['user_id','date'],by='count').reset_index()
    print(user_histroy_df)
    user_histroy_df_7d = user_histroy_df[user_histroy_df.date > datetime.now().date() - timedelta(days=7)]
    user_active_7d_df = simple_group_agg(user_histroy_df_7d,'user_id',label_col='user_id_date_label_count', by='count').reset_index()
    user_active_7d_df = user_active_7d_df[['user_id','user_id_user_id_date_label_count_count']]
    user_active_7d_df = user_active_7d_df.rename(columns={'user_id_user_id_date_label_count_count':'user_active_date_7d'})
    print(user_active_7d_df)

    user_histroy_df_14d = user_histroy_df[user_histroy_df.date > datetime.now().date() - timedelta(days=14)]
    user_active_14d_df = simple_group_agg(user_histroy_df_14d,'user_id',label_col='user_id_date_label_count', by='count').reset_index()
    user_active_14d_df = user_active_14d_df[['user_id','user_id_user_id_date_label_count_count']]
    user_active_14d_df = user_active_14d_df.rename(columns={'user_id_user_id_date_label_count_count':'user_active_date_14d'})
    print(user_active_14d_df)

    total_df = total_df.merge(user_active_7d_df,on='user_id',how='left')
    total_df = total_df.merge(user_active_14d_df,on='user_id',how='left')
    total_df[['user_active_date_7d','user_active_date_14d']] = total_df[['user_active_date_7d','user_active_date_14d']].fillna(0)
    print(total_df.columns)
    print(total_df)

    user_histroy_df =simple_group_agg(total_df,['user_id','label_level1_id'],by='count').reset_index()
    user_histroy_df.sort_values(['user_id','user_id_label_level1_id_label_count'],ascending=[1,0],inplace=True)
    print(user_histroy_df)
    user_histroy_df = user_histroy_df.groupby(['user_id']).head(1)

    print(user_histroy_df)
    user_label_count_df =  simple_group_agg(user_histroy_df,['user_id'],label_col='user_id_label_level1_id_label_count',by='count',agg_funcs={
        'count':    pd.NamedAgg(column="user_id_label_level1_id_label_count", aggfunc="count"),
       'topn_concat': pd.NamedAgg(column="user_id_label_level1_id_label_count", aggfunc=lambda row: "_".join([str(i) for i in list(row.unique())])) 
    }).reset_index()
    print(user_label_count_df)
    user_label_count_df = user_label_count_df[['user_id','user_id_user_id_label_level1_id_label_count_topn_concat']]
    user_label_count_df = user_label_count_df.rename(columns={'user_id_user_id_label_level1_id_label_count_topn_concat':'user_clk_label_topn'})

    total_df = total_df.merge(user_label_count_df,on='user_id',how='left')
    total_df['user_clk_label_topn'] = total_df['user_clk_label_topn'].fillna(0)

    total_df = apply_if_col_not_exists(total_df,'age',lambda row: -1 if  row.get('birthday',None) is None or str(row.get('birthday',None)).lower() == 'nan'  else  int((datetime.now().date() - row.get('birthday')).days / 356) ,overwrite=True)
    print(total_df.columns)
    print(total_df)
    print(total_df[total_df.label==1])

    total_df = total_df[['note_type', 'favorite_number', 
                        'real_source', 'comment_number_x',
                        'share_number_x', 'praise_number', 
                        'play_number', 'video_duration',
                        'note_duration', 'gender',
                        'video_public_release_days', 'label_level1_id', 
                        'label', 'note_id_label_mean',
                        'note_id_video_play_percent_mean', 'user_active_date_7d',
                        'user_active_date_14d', 'user_clk_label_topn','age','label_id','note_id','user_id'
                        ]]
    print(total_df)
    total_df = total_df.rename(columns={
        'share_number_x':'share_number',
        'note_id_label_mean':'note_ctr',
    })
    total_df.to_pickle(output_path)


    return total_df

if __name__ == '__main__':
    get_orig_data()
