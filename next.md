### fine tuning
baseline BERT  
On Colab, 3epochs...

- batch size: 16, 32
- lr(Adam): 5e-5, 3e-5, 2e-5
- number of epochs: 3, 4

### + Decoder
- Encoder: Bert Model, only multi-transformer encoder layers  
    + L=12, Transformer blocks(transformer encoder)
    + H=768, Hidden size
    + A=12, self-attention heads

### Ref Stanford final project: Bert ++
~~combine Bert with QANet  ~~

~~- context to query~~
~~- stacked layers~~
~~- pointer network???~~

Because BERT is a pre-trained contextual embedding, it has already performed some of the tasks included in the layers of QANet, such as the stacked model encoder blocks. As such, a simpler model was implemented using only the context-query attention layer from BiDAF followed by a linear output and softmax layer—CQ-BERT

- What You Can Do
    + according to the final project paper of cs224n, split the output of bert into context and query
    + Or post to attention layer but **Q==C**
        * Using PointerNetDecoder in DuReader(rc_model.py)
        * Context_to_Query_Attention_Layer(model.py/QANet)
    + Or only self attention block///multi-header attention
        * QANet Dureader

### Candidate
- question classifier
- data augment
- IN THE FUTURE...
- ensemble[x]
- shorten the context[x]/max length of context is subject to the force of computer