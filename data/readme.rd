1. 文件说明
	/data
		train.doc_query 是训练集的文档和问题
		train.answer 是训练集的答案
		test.doc_query 是测试集的文档和问题
	/predict
		predict.sample 是训练集上的输出样例
		reference.answer 和train.answer一样
		evaluate.py 是评价代码

2. 任务说明
	我们的任务是给定一个问题和一篇文档，在文档中找到准确的答案。
	我们使用的评价指标是F1值，其计算方式为：
		precision = 1.0 * num_same / len(pred_tokens)
		recall = 1.0 * num_same / len(ref_tokens)
		F1 = (2 * precision * recall) / (precision + recall)
		其中 num_same 为预测答案和正确答案共有字符数

3. evaluate.py 使用样例
	python evaluate.py prediction_path reference_path

	eg: python evaluate.py predict.sample reference.answer

4. 课程结束后要写一份课程报告