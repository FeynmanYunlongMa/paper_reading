Augmenting Sequential Recommendation with Pseudo-Prior Items via Reversely Pre-training Transformer

# 1. Motivation

基于序列模型的推荐系统，关键在于建模序列内item之间的关系，如基于Transformer的模型SASRec与BERT4Rec。但是这些Transformer-based模型都面临着冷启动问题（cold-start issue），比如在很短的序列上会表现很差。

<img src='images/asrep_seq_length.jpg'>

作者通过实验证实，超短序列的效果确实远差于长序列，从而决定尝试**扩充序列**。

本文提出扩充序列的推荐模型。首先pre-train一个反向序列的Transformer，用来预测每个序列的前一个item；然后基于这个Transformer在比较短的序列头部添加一些item来扩充序列；再使用扩充后的序列，fine-tune Transformer进行学习。

思路上很棒，值得尝试与延伸扩展。

# 2. AsReP

<img src='images/asrep_model.jpg'>

### 2.1 Reversely Pre-training

训练一个反向Transformer（从右到左，默认的item序列是从左到右是从旧到新的时间序列），输入一个逆时间序的item序列，预测该序列前一个item。

### 2.2 Short sequences augmentation

用预训练好的反向Transformer对短序列预测$k$个序列之前的item。

这里有两个超参，短序列的长度$M$（本文中等于3）；以及需要补的item个数$k$（本文等于3）。

## 2.3 Left-to-right fine-tuning

将填充之后的序列与其它正常的序列一起，用正常的从左到右的序列方式fine-tune。

# 3. Experiment

<img src='images/asrep_result.jpg'>

优点：

（1）对于短序列学习反向Transformer，从而进行序列扩充，是一个很好的idea；对于预训练在推荐系统中的应用是一个很好的启发。

思考：

(1) 两个超参（短序列的长度、需要填充的item个数）在实际业务中该如何设置，效果对参数是否敏感？

(2) 线上模型一直在线学习的状态下，整个训练过程如何继续？需要训练一个旁路预训练网络，来定期更新反向Transformer？感觉单个前向流式训练的方式有点吃紧。


# 4. Preferences

[1] Liu, Zhiwei, et al. "Augmenting Sequential Recommendation with Pseudo-Prior Items via Reversely Pre-training Transformer." arXiv preprint arXiv:2105.00522 (2021).