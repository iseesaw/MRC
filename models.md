## Models

### Some Keywords Need to Know!!!
#### About Attention
![](https://mchromiak.github.io/articles/2017/Sep/01/Primer-NN/img/EncDecAttention.gif)
- 计算注意力主要分三个步骤：
    1. 计算query和每个key(Encoder中的项)之间的相似性$f_c{(q,k_i)}$以获得注意力分配权重。其中相似函数有点积、拼接、检测器等。
    2. 然后使用softmax函数来正则化这些权重。
    3. 最后将这些权重与相应的value(NLP任务key=value)一起加权并获得最终的值。
$$A(q, (k, v)) \rightarrow_{output}^{maps\ as} \rightarrow \Sigma_{i=1}^k {f_c(q, k_i)v_i, q\in Q, k\in K, v\in V} $$

- self Attention

- 三类模型
- Attention Reader
    - 通过动态attention机制从文本中提取相关信息，再依据该信息给出预测结果
- Attention-Sum Reader
    - 只计算一次attention weights，然后直接喂给输出层做最后的预测，也就是利用attention机制直接获取文本中各位置作为答案的概率，和pointer network类似思想，效果依赖于对query的表示
- Multi-hop Attention
    - 计算多次attention

改进模型
- BiDAF(Bi-Directional Attention Flow) 
<img src="https://allenai.github.io/bi-att-flow/BiDAF.png" height="30%" width="60%"/> 
主要是Context-to-query (C2Q) attention 和 Query-to-context (Q2C) attention。


### Main Models
- 传统模型(参考论文的模型介绍、背景部分)
    + 基于手工设计的语法
    + 基于检测谓词参数三元组的信息提取方法
    + 作为关系数据库进行查询
    + (缺乏大规模训练数据集)

- 监督机器学习模型
    + 对问题、篇章分别进行词法、句法分析，针对分析结果进行特征提取：
    + 基于特征采用诸如 LR、CRF 等模型进行答案边界预测；
    + 采用梯度下降类算法在训练集上进行优化，拟合数据分布。
    + 基于特征的逻辑回归(SQuAD数据集baseline)

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
    - BiDAF
    - R-Net
        + 在BiDAF上加入了self-matching
    - QANet
        + 训练快 + data augment(原语言 => 外语言 => 原语言, 依赖于翻译增加数据)
        + 对内存需求大
    - GPT & BERT + 
        + 预处理模型

#### state of the art
BERT+ AoA(Attention over Attention) + DAE(Data Augment Enhance, MT or Q-A)


### Trick
- Teacher model => student model\<Born-Again Neural Networks, Self-Competition>

