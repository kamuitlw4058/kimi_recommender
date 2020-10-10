## common库地址
https://gitlab.xiaoxiangyoupin.com:9443/wangcenhan/python_common

## 启动路径
* src/main/python/trainer.py 用于模型训练
* src/main/python/stream.py 用于启动模型服务

## 原先项目的问题
* 通过反复take的方式进行数据处理。显然是会严重影响性能的。
* 召回部分都是单个item的服务。如果需要后续提升性能，召回部分也需要做成批量的。增加吞吐

## 当前实现碰到问题
* 通过spark streaming去调用 ml接口有一些局限性，不是rdd和df转换开销较大就是df自身无法直接通过mapPartition的方式优化性能。
  因此抛弃了sparkml直接使用python的sklearn做模型的预测和训练。目前看下来性能应该应该满足大部分需求。
* 后续如果起量，通过增加kafka的partition和服务的节点数量就应该能简单扩展。
