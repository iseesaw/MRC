### MRC-GUI
实现MRC的用户界面

包括但不限于：  
+ 文章文件选择
+ 语音询问(科大讯飞接口，语音识别和语音合成)
+ 回答在原文中不同颜色标注(nbest)
    * 解决方案
        - 使用<span style="background: red">Red <span style="background: blue"> Here</span> Here </span>
        - 重叠问题使用区间重叠方法
    * combox按钮，可以选择查看不同的问题进行答案查看
+ 考虑网页形式或PyQT内嵌网页