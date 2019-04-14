### Lab

| model | parameter | result(mrc) |
| ----- | --------- | ------ |
| Bert baseline | train_batch_size=6</br>lr=3e-5</br>num_train_epochs=3</br>max_seq_length=384</br>doc_stride=128 | F1=84.54</br>EM=64.865 |
| ... | train_batch_size=16 | ... |
| +pointer-network | ... | ... |
| +query classifier | ... | ... |


### Analysis
- 数据集问答对不规范  
    + adj + n VS. n(形容词)  