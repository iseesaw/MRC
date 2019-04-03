# -*- coding: utf-8 -*-
"""
read_squad_examples
    读取squad格式的json数据并保存为SquadExample格式返回

InputFeatures

"""
import json
import tensorflow as tf
from bert import tokenization


def read_squad_examples(input_file, is_training):
  """Read a SQuAD json file into a list of SquadExample."""
  with tf.gfile.Open(input_file, "r") as reader:
    input_data = json.load(reader)["data"]

  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False

  examples = []
  for entry in input_data:
    for paragraph in entry["paragraphs"]:
      paragraph_text = " ".join(paragraph["context"])
      doc_tokens = []
      char_to_word_offset = []
      prev_is_whitespace = True
      for c in paragraph_text:
        if is_whitespace(c):
          prev_is_whitespace = True
        else:
          if prev_is_whitespace:
            doc_tokens.append(c)
          else:
            doc_tokens[-1] += c
          prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

      for qa in paragraph["qas"]:
        qas_id = qa["id"]
        question_text = " ".join(qa["question"])
        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False
        if is_training:

          # if FLAGS.version_2_with_negative:
          #   is_impossible = qa["is_impossible"]
          if (len(qa["answers"]) != 1) and (not is_impossible):
            raise ValueError(
              "For training, each question should have exactly 1 answer.")
          if not is_impossible:
            answer = qa["answers"][0]
            orig_answer_text = " ".join(answer["text"])
            answer_offset = answer["answer_start"] * 2
            answer_length = len(orig_answer_text)
            start_position = char_to_word_offset[answer_offset]
            try:

              end_position = char_to_word_offset[answer_offset + answer_length - 1]
            except:
              print("wrong... qas_id: {}, question_text: {}".format(qas_id, question_text))
              continue
            # Only add answers where the text can be exactly recovered from the
            # document. If this CAN'T happen it's likely due to weird Unicode
            # stuff so we will just skip the example.
            #
            # Note that this means for training mode, every example is NOT
            # guaranteed to be preserved.
            actual_text = " ".join(
              doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = " ".join(
              tokenization.whitespace_tokenize(orig_answer_text))
            # print(actual_text, cleaned_answer_text)
            if actual_text.find(cleaned_answer_text) == -1:
              tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                                 actual_text, cleaned_answer_text)
              continue
          else:
            start_position = -1
            end_position = -1
            orig_answer_text = ""

        example = SquadExample(
          qas_id=qas_id,
          question_text=question_text,
          doc_tokens=doc_tokens,
          orig_answer_text=orig_answer_text,
          start_position=start_position,
          end_position=end_position,
          is_impossible=is_impossible)
        examples.append(example)

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
      s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
      s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
      if self.start_position:
        s += ", start_position: %d" % (self.start_position)
      if self.start_position:
        s += ", end_position: %d" % (self.end_position)
      if self.start_position:
        s += ", is_impossible: %r" % (self.is_impossible)
      return s

