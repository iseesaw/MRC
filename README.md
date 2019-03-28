### Machine Reading Comprehension(MRC)
机器阅读理解

### 概述
分类:  

- 抽取式
- 生成式

### 技术
- Pre-training
- self attention
- word embedding (word2vec ELMo Glove BERT)
- word segment

### 问题
>浅层的模式匹配(字符串、关键字)
>歧义

### 算法
- $P(a | q, d)$
- 基于特征的逻辑回归(SQuAD数据集baseline)


### 模型
- 传统模型(参考论文的模型介绍、背景部分)
    + 基于手工设计的语法
    + 基于检测谓词参数三元组的信息提取方法
    + 作为关系数据库进行查询
    + (缺乏大规模训练数据集)

- 监督机器学习模型
    + 对问题、篇章分别进行词法、句法分析，针对分析结果进行特征提取：
    + 基于特征采用诸如 LR、CRF 等模型进行答案边界预测；
    + 采用梯度下降类算法在训练集上进行优化，拟合数据分布。

- 神经网络模型解决机器阅读理解的开端
    + [Teaching Machines to Read and Comprehend](https://arxiv.org/pdf/1506.03340.pdf)
        + Attentive Reader
        + Impatient Reader

- End2End， Embed 层，Encode 层，Interaction 层和 Answer 层
    + Embed 层负责将原文和问题中的 tokens 映射为向量表示；
    + Encode 层主要使用 RNN 来对原文和问题进行编码，这样编码后每个 token 的向量表示就蕴含了上下文的语义信息；
    + Interaction 层是大多数研究工作聚焦的重点，该层主要负责捕捉问题和原文之间的交互关系，并输出编码了问题语义信息的原文表示，即 query-aware 的原文表示；
    + Answer 层则基于 query-aware 的原文表示来预测答案范围。
- 近年机器阅读理解发展
<center>
    <img src="ref/mrc_dev.jpg" height="50%" width="50%">
</center>

### state of the art
BERT+

### 评价指标
- F1
- recall
- precision
    + BLEU
    + METEOR
    + ROUGE
- Exact match(EM)
- similarity
    + MAP
    + MRR

### 数据集
- 文本填词
    + [iFLYTEK Research & HIT SCIR](https://www.aclweb.org/anthology/C16-1167)
- 完形填空
    + [CMU](https://aclweb.org/anthology/D17-1082)
- 文本段落
    + [SQuAD(Stanford Question Answer Dataset)](https://rajpurkar.github.io/SQuAD-explorer/)
    + 
    + [DuReader](https://aclweb.org/anthology/W18-2605)
    + [CoQA](https://arxiv.org/pdf/1808.07042.pdf)


### 比赛
- 2018机器阅读理解技术竞赛
    + Baseline: Match-LSTM 和 BiDAF(Dynatic Attention Flow)

### 应用
- 问答系统
- 对话系统
