from datetime import datetime

def get_ext1_name(value):
    d = {
    "1":"颜世界列表",
    "2":"关注列表",
    "3":"附近列表", 
    "4":'图文详情推荐列表',
    '5':'个人中心笔记列表',
    '6':'话题主页笔记列表',
    '7':'位置主页笔记列表',
    '8':'品牌主页笔记列表',
    '9':'影视主页笔记列表',
    '10':'收藏笔记列表',
    '11':'收藏专辑笔记列表',
    '12':'颜值圈首页视频列表',
    '13':'相关推荐列表',
    '14':'赞过的笔记列表',
    '15':'关注页面，无关注时推荐',
    '16':'推荐关注页面，用户笔记推荐列表',
    '17':'搜索结果页',
    '18':'消息中心笔记详情页',
    }
    ret = d.get(value,None)
    if ret is None:
        return f'{value} unknown'
    return ret



def get_ext2_name(value):
    d = {
    "1":"新浪分期",
    "":"小象优品",
    }
    ret = d.get(value,None)
    if ret is None:
        return f'{value} unknown'
    return ret


def get_device_type_name(value):
    d = {
    "1":"安卓",
    "2":"iOS",
    }
    ret = d.get(str(value).strip(),None)
    if ret is None:
        return f'{value} unknown'
    return ret

def convert_datetime_to_date(datetime_format='%Y-%m-%d %H:%M:%S'):
    def func(value):
        return datetime.strptime(str(value),datetime_format).date()
    return func
