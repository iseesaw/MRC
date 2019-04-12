### run_squad.py
`bert`用于机器阅读理解的`fine tuning`脚本，主要作用：  
- 读取`squad`类型数据作为训练样本
- 使用`bert`预训练模型，取最后一层输出，类似`ELMO`词向量
- 将预训练模型输出接一层全连接层
- 输出为元组，表示答案片段的起始位置

### 分模块学习

#### SquadExample
>A single training/test example for simple sequence classification.
>For examples without an answer, the start and end position are -1.


- `__str__` 调用`__repr__`，`print(object)`时的输出

- `__repr__` 拼接字符串

#### InputFeatures
>A single set of features of data.

- read_squad_examples 
>Read a SQuAD json file into a list of SquadExample.  

使用`tf.gfile`操作文件。依次读取json文件，data -> entry -> paragraphs -> context/qas(-> id question answers text answer_start)(`Squad2`中包含`is_impossible`字段)；最后将每个样本(一个问题为一个样本，一篇文章可能在多个样本中)保存为`SquadExample`类型对象。  
详细处理包括：  
1. `char_to_word_offset` 用于根据答案和答案开始位置确定答案的起始位置(即模型的输出)
2. `tokenization.whitespace_tokenize`对原始答案进行取空白符`\s`处理，判断能否从`document`获取答案，不能则跳过(避免`weird Unicode stuff`)

>paragraph_text = 'This is a test, good luck!\r'
>doc_tokens = ['This', 'is', 'a', 'test,', 'good', 'luck!']
>char_to_word_offset = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 
>   3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]

- convert_examples_to_features
>Loads a data file into a list of `InputBatch`s.

将每个样本从`SquadExample`类型转为`InputFeatures`类型。  
1. 对每个样本的`question_text`用`tokenizer.tokenize`处理。  
2. 对每个问题进行`max_query_length`判断，超过最大长度则截断
3. 对文本中每个词进行`tokenizer.tokenize`处理
4. `doc_span`，将超过最大长度的文件进行窗口移动截断成多个片段
5. 连接文章和问题 `[CLS]`+ context + `[SEP]` + query + `[SEP]`
6. `input_ids` 使用`tokenizer.convert_tokens_to_ids(tokens)`将词用词表中的id表示
7. `input_mask` 词用1表示，填充用0表示
8. `segment_ids` 文章中词用0表示，问题中词用1表示
9. `output_fn(feature)`进行`run callback`，回调函数主要作用是进行特征写入

>`input_ids, input_mask, segment_ids`都用0进行填充

- def _improve_answer_span
>Returns tokenized answer spans that better match the annotated answer.

主要是将 (1895-1943) 处理为 ( 1895 - 1943 )

- _check_is_max_context
>Check if this is the 'max context' doc span for the token.

当使用`sliding window`方法后，
```
Doc: the man went to the store and bought a gallon of milk
Span A: the man went to the
Span B: to the store and bought
Span C: and bought a gallon of
```
要获得一个词的最大上下文，比如`bought`在B中有4个左上下文和0个右上下文，而在C中有1个左上下文和3个右上下文，最终选择片段C。

#### create_model
>Create a classification model

Bert fine tuning: 
```python
model = modeling.BertModel(
    config=bert_config,
    is_training=is_training,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    use_one_hot_embeddings=use_one_hot_embeddings)
# 得到词向量输出(run_classifier.py中model.get_pooled_output()是一维句子向量)
final_hidden = model.get_sequence_output()

# 输出维度为(batch_size, seq_length, word_vector_shape)
final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
batch_size = final_hidden_shape[0]
seq_length = final_hidden_shape[1]
hidden_size = final_hidden_shape[2]

# 获得weights和bias变量
output_weights = tf.get_variable(
    "cls/squad/output_weights", [2, hidden_size],
    initializer=tf.truncated_normal_initializer(stddev=0.02))
output_bias = tf.get_variable(
    "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

final_hidden_matrix = tf.reshape(final_hidden,
                                    [batch_size * seq_length, hidden_size])

# 全连接层：matmul + bias
logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
logits = tf.nn.bias_add(logits, output_bias)

# 维度转换与句子分解
logits = tf.reshape(logits, [batch_size, seq_length, 2])
logits = tf.transpose(logits, [2, 0, 1])

unstacked_logits = tf.unstack(logits, axis=0)

# 模型输出
(start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])
```

#### model_fn_builder
>Returns `model_fn` closure for TPUEstimator.

- model_fn
> The `model_fn` for TPUEstimator.

1. 构建模型函数的时候需要完成model_fh(features,labels,mode,params)这个函数
2. 这个函数需要提供代码来处理三种mode(TRAIN,EVAL,PREDICT)值，返回tf.estimator.EstimatorSpec的一个实例
3. tf.estimator.ModeKeys.TRAIN模式：主要是需要返回loss，train_op(优化器)
4. tf.estimator.ModeKeys.PREDICT模式：主要是需要返回predictions结果
5. tf.estimator.ModeKeys.EVAL模式：主要是需要返回loss,eval_metrics=[评价函数]


#### input_fn_builder
>Creates an `input_fn` closure to be passed to TPUEstimator.

设计一个模型的输出函数，完成读取tf.record文件，反序列化样本获得原始的样本，如果是训练的话，则打乱数据集，获取batch量的样本集

- _decode_record
>Decodes a record to a TensorFlow example.

- input_fn
>The actual input function.

#### write_predictions
>Write final predictions to the json file and log-odds of null if needed.


#### get_final_text
>Project the tokenized prediction back to the original text.
```
pred_text = steve smith
orig_text = Steve Smith's
```

- _strip_spaces


#### _get_best_indexes
>Get the n-best logits from a list.


#### _compute_softmax
>Compute softmax probability over raw logits.


#### FeatureWriter
>Writes InputFeature to TF example file

临时特征文件存储。

- process_feature
>Write a InputFeature to the TFRecordWriter as a tf.train.Example.

- create_int_feature
定义特征。

```python
features = collections.OrderedDict()
features['key'] = create_int_feature(value)
#...

# 定义一个Example，包含若干个feature，每个feature是key-value结构
tf_example = tf.train.Example(features=tf.train.Features(feature=features))

# 将样本序列化（压缩）保存到tf.record文件中
self.__writer.write(tf_example.SerializeToString())
```

#### validate_flags_or_throw
>Validate the input FLAGS or throw an exception

关于命令行输出参数的异常判断。

#### main
这部分主要是需要获取自定义参数，和构建运行逻辑

1. 自定义参数：
```python
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(name, default='', description)

# 设定某个参数是必须给定的
flags.mark_flag_as_required(name)
```
2. 预加载`bert`
```python
# config
bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
# tokenizer
# bert 自带的tokenizer
tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
```
3. 配置`run_config`

- `model_dir`：配置输出文件夹
- `save_checkpoint_steps`:训练多少步保存`checkpoint`

4. 加载数据集，计算训练步
```python
# 这里不同的是，考虑了epoch
num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
```
5. 构建模型和`Estimator`

- 构建`model_fn_builder()`
- 构建`estimator`
```python
tf.train.Estimator(model_fn=model_fn,config=run_config,train/eval/predict_batch)
```
6. 训练/验证/预测

- 构建`input_fn_builder()`:指定如何解析样本（如从`record`文件中读取解析）
- `estimator.train(input_fn=...,max_steps=num_train_steps)`，这个操作会将`mode`设为`ModeKeys.TRAIN`
- `predict`，一个一个样例返回
- `result`中包含了在`model_fn`中`predict`模式返回的实例中`predictions`参数的内容
```
for result in estimator.predict(predict_input_fn,yield_single_examples=True):
    print(result)
```
[Ref](https://github.com/NoneWait/bert_demo/tree/master/demo_aic_v2)

### TensorFlow知识点

#### tf.flags.DEFINE_xxx
用于添加命令行参数
```python
# 定义参数
tf.flags.DEFINE_string("strParam", value, "This is a string param")
#tf.flags.DEFINE_bool/integer/float/

# 使用参数
tf.flags.FLAGS.strParam

# 命令行输入，更换参数
# python file.py --strParam strname

```
[Ref](https://blog.csdn.net/spring_willow/article/details/80111993)

#### tf.gfile
TensorFlow的文件操作，包括但不限于：  
- tf.gfile.MakeDirs(FLAGS.output_dir)
- with tf.gfile.Open(file, 'r') as reader

### 其他

- 抛出异常 raise ValueError("This causes a value error.")