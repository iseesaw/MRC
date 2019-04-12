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
            print("*************************************")
            print("Q: {}".format(q))
            print("A: {}".format(final[q_id]))
            if 1:
              print("Nbest:")
              for i, a in enumerate(nbest[q_id][:5]):
                  print("{}th {}".format(i + 1, a["text"]))