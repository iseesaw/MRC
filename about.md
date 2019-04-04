### Machine Reading Comprehension(MRC)
机器阅读理解

### About
分类:  

- 抽取式
- 生成式

...

### Dataset
- 文本填词
    + [iFLYTEK Research & HIT SCIR](https://www.aclweb.org/anthology/C16-1167)
- 完形填空
    + [CMU](https://aclweb.org/anthology/D17-1082)
- 文本段落
    + [SQuAD(Stanford Question Answer Dataset)](https://rajpurkar.github.io/SQuAD-explorer/)
    + 
    + [DuReader](https://aclweb.org/anthology/W18-2605)
    + [CoQA](https://arxiv.org/pdf/1808.07042.pdf)


### Data Preparing
- word embedding
    + word2vec
    + glove
    + \<pos embedding>
- Pre-trainings
    + ELMO(迁移模型)
    + BERT
    + GPT
- question type embedding
    + who/when/num/how long/where/how/why
- data quality
    + 数据集标注一致性

### Question
>浅层的模式匹配(字符串、关键字)
>歧义

### Model
See [models.md](models.md)

### Metrics
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

### Competition
- 2018机器阅读理解技术竞赛(DuReader)
    + Baseline: Match-LSTM 和 BiDAF(Dynatic Attention Flow)
- cmrc2018
- SQUAD1.1/2.0

### Application
- Search engine
    + hightlight the answer
- Custom Services
- Finance/Education

### Future
- High level reasoning
    + Question A => Question B => Question C => Answer
- Using commonsense knowledge
- Unanswerable questions

### Our Model
See [model.md](model.md)
