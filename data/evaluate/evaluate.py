# -*- coding: UTF-8 -*-
import sys
from collections import Counter

def load_file(path):
	result = dict()
	file = open(path, 'r')
	for line in file.readlines():
		line = line.decode('utf-8').strip()
		qid, answer = line.split(' ||| ')
		result[qid] = answer
	return result

def f1_score(pred, ref):
	pred_tokens = list(pred)
	ref_tokens = list(ref)
	common = Counter(pred_tokens) & Counter(ref_tokens)
	num_same = sum(common.values())
	if num_same == 0:
		return 0
	precision = 1.0 * num_same / len(pred_tokens)
	recall = 1.0 * num_same / len(ref_tokens)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1

def evaluate(predictions, references):
	f1 = total = 0
	for ref_id in references:
		total += 1
		ref = references[ref_id]
		pred = predictions[ref_id]
		f1 += f1_score(pred, ref)
	f1 = 100.0 * f1 / total
	return f1

if __name__ == '__main__':
	reload(sys)
	sys.setdefaultencoding('utf8')
	prediction_path = sys.argv[1]
	reference_path = sys.argv[2]
	pred_dict =  load_file(prediction_path)
	ref_dict = load_file(reference_path)
	result = evaluate(pred_dict, ref_dict)
	print (result)
