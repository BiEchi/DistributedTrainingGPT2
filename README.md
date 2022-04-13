# Distributed GPT2-Chinese

## Description

本repo为基于[GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese)的并行训练研究.研究使用PyTorch作为开发语言,使用PyTorch提供的数据并行接口进行代码修改.

研究目前仅涵盖data parallelization.另一并行种类为model parallel,要求对某一特定model进行细致的理解,目前正在推进理论部分,预计于2022年8月发表并行实验结果.

研究主要涉及Single Paramter Server(SPS)分布式, Distributed Parameter Server(DPS)分布式,Horovod框架分布式,以及基于DPS和Horovod的Apex混合精度训练.

## Environmental Setup

### 下载Repo

本项目所有资源的唯一官方地址为[github repo](https://github.com/BiEchi/DistributedTrainingGPT2).

```shell
# https
git clone https://github.com/BiEchi/DistributedTrainingGPT2
# ssh
git clone git@github.com:BiEchi/DistributedTrainingGPT2.git
```

### 进入待实验文件夹

实验有多个文件夹,每个文件夹完全独立于其他文件夹,也即每个文件夹是可以单独运行的程序.在复现代码时,注意自己所在的文件夹.注意,由于本实验为非常典型的消融实验(对照实验),所以如果你想要进行结果的有效对比,改变各个文件夹内的参数前请三思.最重要的是,应该对每个文件夹所做的事情有详细的认知,并且如果需要改变,则必须合理地更改batch size和learning rate.如果不修改,可以使用我们团队已经配置好的参数.

```shell
cd DistributedTrainingGPT2
cd GPT2-Chinese-Parallel # for example
```

### 创造数据集文件夹

本repo中,对于**每个**文件夹,都必须提供一个`data`子文件夹,用于存放数据集`train.json`.

```shell
mkdir data
touch train.json
```

注意,`train.json`是一个数组json(而非字典json).一个非常直觉的解释如下.

```json
[
  "Hello World", // item 1
  "I'm Fine" // item 2
]
{
  "Hello World": "1",
  "I'm Fine": "0"
}
```

该现象的原因是,GPT-2是一个生成模型,为了普适性而使用**纯unlabeled data**,进行,所以你可以使用**任何文本**进行训练——这也是作者选择使用GPT-2的原因,可以创造自己的pretrained model从而有非常精确的应用.

注意,数据json中,item的数量要符合直觉.也就是说,不要把两句有强烈关系的话打散,也不要把好几个段落都放在一个item中.如果想要偷懒不进行人为分类,可以直接将数据集按段作为平凡的解决方案.

### 安装python环境依赖

本实验的实验环境推荐使用虚拟环境进行,因为使用到大量的深度学习依赖,如果有旧版本可能会导致程序崩溃.作者的实验环境为HAL系统上自带的环境包`opence/1.5.1`,并且在此之上安装了自己配置的conda虚拟环境`opence-bert`,具体包内容已在`opence-bert.txt`文件中注明.注意,只需要安装本环境包的最小版本即可进行.换句话说,请优先考虑使用每个项目文件夹下的`requirements.txt`进行安装(推荐使用pip包管理工具,而非conda)

### 启动训练脚本

作者设计了一个训练脚本`run.swb`方便进行快速复现.

```shell
# 架设环境
module load opence/1.5.1
conda activate opence-bert
cd ~/gpt2/GPT2-Chinese-Parallel/

# 生成字典(所有出现的字的最大集合)
cd cache/ && time bash make_vocab.sh && cd ../

# 调用训练脚本,在这之后模型已经训练完成
time bash train.sh

# 使用产品(模型)进行inference(GPT-2的任务是generation)
time bash generate.sh

# 增加一些自己的注明,好在训练完成后回看完成的任务是啥
echo Comments: $2
```

其中,生成字典这一步可以使用已经架设完毕的几个文件(都在`cache`文件夹内),而generation这一步如果不考虑实际应用也可以跳过.这是因为,loss函数结果会在train的时候已经通过tensorboard钩子生成在文件夹`tensorboard_summary`内,跑一下`train.py`马上就知道作者的意思了.

如果不想要使用脚本,也可以直接使用命令行完成,但是等待的时间(特别是`train.py`)会很长,仅仅适合在开发环境中使用.在生产/实验环境中,请严格使用脚本提交到任务队列完成.

### 更多资源

对于实验中出现的issue,请到项目的[原repo](https://github.com/Morizeyao/GPT2-Chinese)中查找.作者仅仅花了五六个小时调试环境,说明了原repo的支持是多么宝贵.如果想要调整参数,原repo的readme也值得一读.

## Code

[GPT2-Chinese-Parallel](https://github.com/BiEchi/DistributedTrainingGPT2/tree/main/GPT2-Chinese-Parallel)

基于Single Parameter Server的Data Parallel,改动代码最少,提升最有限.GPU带宽一般的情况下,不推荐使用.

[GPT2-Chinese-Parallel-Distributed](https://github.com/BiEchi/DistributedTrainingGPT2/tree/main/GPT2-Chinese-Parallel-Distributed)

基于Distributed Parameter Server的Data Parallel,改动代码较多,提升幅度很大.几乎不受GPU带宽影响,但是batch size不能太大,否则会报错OOM.

[GPT2-Chinese-Parallel-Distributed-Horovod](https://github.com/BiEchi/DistributedTrainingGPT2/tree/main/GPT2-Chinese-Parallel-Distributed-Horovod)

基于Horovod的Data Parallel.之所以这么取名字(把Horovod放到Distributed后面),是因为Horovod本质是DPS套壳,加了一些稳定性、垃圾回收,还有多node的OpenMPI后端.

[GPT2-Chinese-Parallel-Distributed-Horovod-Apex](https://github.com/BiEchi/DistributedTrainingGPT2/tree/main/GPT2-Chinese-Parallel-Distributed-Horovod-Apex)

在Horovod框架下增加Apex支持.添加很少的代码,性能翻三倍.

[GPT2-Chinese-Parallel-Distributed-Apex](https://github.com/BiEchi/DistributedTrainingGPT2/tree/main/GPT2-Chinese-Parallel-Distributed-Apex)

使用DPS的同时增加Apex支持.添加很少的代码,性能增加有限.

## Conclusion

单节点下,无Apex情况下DPS性能最强,但稳定性略差于Horovod.Horovod性能微弱于DPS,但稳定性强.

多节点下,由于GPU通信瓶颈,速度提升不成线性,且Horovod的操作难度远低于DPS.

另外,Apex可以大幅增加DPS和Horovod的训练速度.

## Copyright

文章已经提交到会议HP3C,将于出版后发表论文链接,在此之前实验结果为保密产品.文章出版后欢迎复现,共同讨论带宽和效率的关系.如果急于获得笔者的结果,可以先自己试试.
