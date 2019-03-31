### 模型综述

#### Attention机制
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

#### Bert