# -*- coding: utf-8 -*-
"""
read_squad_examples
    读取squad格式的json数据并保存为SquadExample格式返回

InputFeatures

"""
import json
import tensorflow as tf
from bert import tokenization

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def read_squad_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample.
    {data:[{paragraphs:[{id, context, qas:[{q, id, answers:[]}]}]}]}
    """
    with tf.gfile.Open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"][:1]:
            paragraph_text = paragraph["context"].lower()
            '''
            paragraph_text = 'This is a test, good luck!\r'
            doc_tokens = ['This', 'is', 'a', 'test,', 'good', 'luck!']
            '''
            doc_tokens = [c if not is_whitespace(c) else "#" for c in paragraph_text]

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                answers = qa["answers"][0]
                orig_answer_text = answers["text"]
                start_position = answers["answer_start"]
                end_position = start_position + len(orig_answer_text)
                is_impossible = False

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
                #print("Question: {}\n Answer: {}".format(question_text, "".join(doc_tokens[start_position:end_position])))
    return examples


class SquadExample(object):
    """A single training/test example for simple sequence classification.

       For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (tokenization.printable_text(
            self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


#read_squad_examples("cmrc/cmrc2018_train.json", True)