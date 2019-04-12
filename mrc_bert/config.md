### parameter

- max_seq_length

| max_seq_length | train_batch_size |
| -------------- | ---------------- |
|      256       |         16       |
|      320       |         14       |
|      384       |         12       |
|      512       |         6        |

- doc_stride 128/64/32

- predict_batch_size 4/8

- num_train_epochs 3.0/5.0/10.0

### code
- run cmrc directly
    + something is wrong
    + there are whitespace in english
    + change chinese dataset into english format
        * " ".join(w for w in text)
        * start_position*2 - 1

- n_best_size(to find the best answer...)

!python run_mrc.py \
  --vocab_file=chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt \
  --do_train=False \
  --do_predict=True \
  --predict_file=eval.json \
  --train_batch_size=6 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=output/