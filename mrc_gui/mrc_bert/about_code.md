## Estimator
- `model_fn_builder`中调用`create_model`创建网络模型(fine tuning)
    + 读取`checkpoint`，并根据`tf.estimator.ModeKeys`，进行具体的训练和预测操作；训练包括定义`loss`函数和优化函数；预测则直接得到预测结果
- `estimator = tf.contrib.TPUEstimator(model_fn)`，创建时输入网络模型
- 训练，`estimator.train(train_input_fn)`
- 预测，`estimator.predict(predict_input_fn)`

## FLAGS.do_train
- `train_examples = read_squad_examples`读取样本数据，返回为`SquadExample`对象
- `train_writer = FeatureWriter()`，特征写入器
    + `tf.train.Example()`
- `convert_examples_to_features(train_exampes，train_writer.process_feature)`
    + 将`SquadExample`对象解析为`InputFeatures`对象
    + 使用回调函数`train_writer.process_feature`，将转换的`InputFeatures`特征写入文件
- `train_input_fn = input_fn_builder(train_writer.filename)`，用于训练时特征读取器
    + 将特征写入文件，包括`unique_ids`，`input_ids`，`input_mask`，`segment_ids`
    + 以及`tf.data.TFRecordDataset()`的创建和读取
- `estimator.train(train_input_fn)`，使用特征读取器训练

## FLAGS.do_predict
- `eval_examples = read_squad_examples()`读取测试数据，返回为`SquadExample`对象
- `eval_writer = FeatureWriter()`，特征写入器
- `convert_examples_to_features(eval_examples，tokenizer，append_feature)`
    + `eval_examples`
    + `tokenizer`
    + `append_feature`，回调函数，保存到`eval_features`中(便于得到预测结果)；用`eval_writer.process_feature`写入文件
- `predict_input_fn = input_fn_builder(eval_writer.filename)`，用于预测时的特征读取器
- `estimator.predict(predict_input_fn)`，使用特征读取器预测
- `write_predictions(eval_examples, eval_features, all_result)`，得到预测结果解析并保存文件

## About
- `create_model(bert_config, is_training, input_ids, input_mask, segment_ids)`
    - `is_training`，`do_train`时为`True`，否则为`False`
    - `input_ids`，`input_mask`，`segment_ids`，输入模型的特征向量