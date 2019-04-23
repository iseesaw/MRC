import json

dirpath = "output"

with open(dirpath+ '/dev_predictions.json', 'r', encoding='utf-8') as f:
    final = json.load(f)

with open(dirpath + '/dev_nbest_predictions.json', 'r', encoding='utf-8') as f:
    nbest = json.load(f)

with open('cmrc/cmrc2018_dev.json', 'r', encoding='utf-8') as f:
    eval_ = json.load(f)

for entry in eval_["data"]:
    for paragraph in entry["paragraphs"]:
        for qas in paragraph["qas"]:
            q = qas["question"]
            q_id = qas["id"]
            print("*************************************")
            print("Q: {}".format(q))
            print("A: {}".format(final[q_id]))
            if False:
              print("Nbest:")
              for i, a in enumerate(nbest[q_id][:5]):
                  print("{}th {}".format(i + 1, a["text"]))