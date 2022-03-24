# Distributed GPT2-Chinese

## Description

本repo为基于[GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese)的并行训练研究.研究使用PyTorch作为开发语言,使用PyTorch提供的数据并行接口进行代码修改.

研究目前仅涵盖data parallelism,而model parallel是下一阶段研究的重点.

研究主要涉及Single Paramter Server(SPS)分布式, Distributed Parameter Server(DPS)分布式,Horovod框架分布式,以及基于DPS和Horovod的Apex混合精度训练.

## Conclusion

单节点下,无Apex情况下DPS性能最强,但稳定性略差于Horovod.Horovod性能微弱于DPS,但稳定性强.

另外,Apex可以大幅增加DPS和Horovod的训练速度.

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

## Pictures

图片部分将在文章发布后陆续公布,有兴趣可以先自己试试.
