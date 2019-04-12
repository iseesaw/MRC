# -*- coding: utf-8 -*-
import os
import sys
from PyQt5.QtWidgets import QStyleFactory
from jinja2 import Environment, FileSystemLoader
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtWidgets, QtGui
import time
import pyaudio
import wave
from concurrent.futures import ThreadPoolExecutor
import threading
from gui import *
import json
from xf import XF_text, XF_Speak

class MyPyQT_Form(QtWidgets.QWidget, Ui_Form):

    def __init__(self):
        super(MyPyQT_Form, self).__init__()
        
        self.setupUi(self)
        self.view = QWebEngineView(self)
        self.view.setGeometry(QtCore.QRect(10, 60, 561, 611))

        self.pushButton_start.clicked.connect(self.start)
        self.pushButton_end.clicked.connect(self.end)
        self.pushButton_ok.clicked.connect(self.get_answer)
        self.pushButton_choosefile.clicked.connect(self.open_file)

        self.fileName = "article.txt"

        self.show_article()

        self.pushButton_ok.setEnabled(False)

    def start(self):
        #self.start_time = time.time()
        self.timer_id = self.startTimer(1000, timerType=QtCore.Qt.VeryCoarseTimer)
        self.pushButton_start.setEnabled(False)
        self.pushButton_end.setEnabled(True)
        threading.Thread(target=self.record, ).start()

    def end(self):
        self.signal = False
        if self.timer_id:
            self.killTimer(self.timer_id)
            self.timer_id = 0
        self.pushButton_start.setEnabled(True)
        self.pushButton_end.setEnabled(False)

        # 显示询问
        self.question = XF_text("output.wav", 16000)
        self.textEdit_question.setText(self.question)
        self.pushButton_ok.setEnabled(True)

    def get_answer(self):
        """
        输入文本，调用模型预测
        """
        self.movie = QtGui.QMovie("icon/loading.gif")
        self.movie.setCacheMode(QtGui.QMovie.CacheAll)
        self.movie.setSpeed(100)

        self.label_mrc.setMovie(self.movie)
        self.movie.start()

        self.question = self.textEdit_question.toPlainText()
        # t = threading.Thread(target=self.run_predict, )
        # t.start()
        # t.join()
        self.run_predict()

    def run_predict(self):
        with open(self.fileName, "r", encoding="utf-8") as f:
            content = f.read()

        data = {
          "version": "v1.0", 
          "data": [
            {
              "paragraphs": [
                {
                  "id": "MRC", 
                  "context": content,
                  "qas": [
                    {
                      "question": self.question,
                      "id": "QUERY", 
                      "answers": [
                        {
                          "text": "", 
                          "answer_start": 0
                        }]
                    }
                  ]
                }
              ], 
              "id": "DEV_0", 
              "title": "title"
            }]
        }
        with open("eval.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # t = threading.Thread(target=self.run_cmd,)
        # t.start()
        # t.join()
        self.thread = Thread()
        self.thread.start()
        self.thread.trigger.connect(self.output)

        
    def output(self):
        # os.system("python D:\\ML_DL\\nlp\\mrc_bert\\run_mrc.py \
        #   --vocab_file=D:\\ML_DL\\nlp\\mrc_bert\\chinese_L-12_H-768_A-12/vocab.txt \
        #   --bert_config_file=D:\\ML_DL\\nlp\\mrc_bert\\chinese_L-12_H-768_A-12/bert_config.json \
        #   --init_checkpoint=D:\\ML_DL\\nlp\\mrc_bert\\chinese_L-12_H-768_A-12/bert_model.ckpt \
        #   --do_train=False \
        #   --do_predict=True \
        #   --predict_file=D:\\ML_DL\\nlp\\mrc_gui\\eval.json \
        #   --train_batch_size=6 \
        #   --predict_batch_size=4 \
        #   --learning_rate=3e-5 \
        #   --num_train_epochs=2.0 \
        #   --max_seq_length=384 \
        #   --doc_stride=128 \
        #   --output_dir=D:\\ML_DL\\nlp\\mrc_gui")

        with open("predictions.json", "r", encoding="utf-8") as f:
            self.answer = json.load(f)["QUERY"]

        # GUI相关对象不能再非GUI的线程中创建和使用
        self.label_mrc.setPixmap(QtGui.QPixmap("icon/结束.png"))
        self.label_mrc.setScaledContents(True)

        self.textBrowser_question.setText(self.answer)

        #读取答案
        #XF_Speak(answer)
        threading.Thread(target=self.speak,).start()
        # with open("nbest_predictions.json", "r", encoding="utf-8") as f:
        #     nbest = json.load(f)


    def speak(self):
        XF_Speak(self.answer)

    def timerEvent(self, event):
        end_time = time.strftime("%H:%M:%S")
        self.label_question.setText(end_time)

    def record(self):
        self.signal = True
        CHUNK = 256
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 11025

        p = pyaudio.PyAudio()
     
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
     
        frames = []
        while self.signal:
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()
     
        wf = wave.open("output.wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))


    def show_article(self, start=None, end=None):
        # 绝对地址，否则无法正常显示
        self.view.setHtml(self.get_article(start, end), baseUrl=QtCore.QUrl.fromLocalFile(os.path.abspath('article.html')))

    def get_article(self, start=None, end=None):
        """获取好友动态
        渲染生成HTML代码
        """
        # 读取用户好友动态
        env = Environment(loader=FileSystemLoader("./"))
        template = env.get_template("article.html")

        article = "Article"
        with open(self.fileName, "r", encoding="utf-8") as f:
            content = f.read()

        # 渲染结果
        if start:
            pre = "<span style=\"background:red\">"
            post = "</span>"
            content = content[:start] + pre + content[start:end] + post + content[end:]
        
        content = template.render(article=article, content=content)
        return content

    def open_file(self):
        """点击按钮、读取文件、并渲染到文本框中"""
        self.fileName, self.filetype = QFileDialog.getOpenFileName(self,
                                                          "选取文件",
                                                          "./",
                                                          "Text Files (*.txt);;All Files (*)")  # 设置文件扩展名过滤,注意用双分号间隔

        if self.fileName != "":
            self.show_article()
        else:
            self.fileName = "article.txt"

class Thread(QThread):
    """docstring for Thread"""
    trigger = pyqtSignal()
    def __init__(self):
        super(Thread, self).__init__()
    
    def run(self):
        os.system("python D:\\ML_DL\\nlp\\mrc_bert\\run_mrc.py \
          --vocab_file=D:\\ML_DL\\nlp\\mrc_bert\\chinese_L-12_H-768_A-12/vocab.txt \
          --bert_config_file=D:\\ML_DL\\nlp\\mrc_bert\\chinese_L-12_H-768_A-12/bert_config.json \
          --init_checkpoint=D:\\ML_DL\\nlp\\mrc_bert\\chinese_L-12_H-768_A-12/bert_model.ckpt \
          --do_train=False \
          --do_predict=True \
          --predict_file=D:\\ML_DL\\nlp\\mrc_gui\\eval.json \
          --train_batch_size=6 \
          --predict_batch_size=4 \
          --learning_rate=3e-5 \
          --num_train_epochs=2.0 \
          --max_seq_length=384 \
          --doc_stride=128 \
          --output_dir=D:\\ML_DL\\nlp\\mrc_gui")
        
        self.trigger.emit()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    # Fusion Windows
    app.setStyle(QStyleFactory.create('Fusion'))

    my_pyqt_form = MyPyQT_Form()
    my_pyqt_form.show()
    sys.exit(app.exec_())
