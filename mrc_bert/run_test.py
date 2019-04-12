"""
python bert/extract_features.py \
  --input_file=sentence.txt \
  --output_file=output.jsonl \
  --vocab_file=chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=8
"""
import json

dirpath = "output"

with open(dirpath+ '/predictions.json', 'r', encoding='utf-8') as f:
    final = json.load(f)

with open(dirpath + '/nbest_predictions.json', 'r', encoding='utf-8') as f:
    nbest = json.load(f)

with open('eval.json', 'r', encoding='utf-8') as f:
    eval_ = json.load(f)

for entry in eval_["data"]:
    for paragraph in entry["paragraphs"]:
        for qas in paragraph["qas"]:
            q = qas["question"]
            q_id = qas["id"]
            print("Q: {}".format(q))
            print("A: {}".format(final[q_id]))
            print("Nbest:")
            for i, a in enumerate(nbest[q_id][:5]):
                print("{}th {}".format(i, a["text"]))