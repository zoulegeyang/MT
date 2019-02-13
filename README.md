# MT
基于注意力机制的机器翻译模型

![计算图](/MT/picture/计算图.png）

## 训练数据
* 来源： http://www.manythings.org/anki/
* 训练数据由两万多个成对的语句组成，如：I love you. 我爱你。
## 训练模型
```python
#在目录MT下，在命令行中依次运行下面语句
from model import Model
model = Model()
# 加载数据，默认参数：path='./dataset/mt_en2ch.txt'， num_examples = 20000, batch_size=64
model.load_data()
#训练,默认EPOCHS = 10
model.train()
```

## 使用训练后的模型将中文翻译成英文
```python
#在目录MT下，在命令行中依次运行下面语句
from model import Model
model = Model()
model.load_data()
sentence = input("请输入要翻译的中文句子：")
model.translate(sentence)
```

## 保存模型
1. 模型在训练中，每隔2epochs，会将训练数据保存在/MT/training_checkpoints目录下，下次训练模型会先判断是否有已经训练的数据，如果有会接着上次继续训练，从而提高训练效率，如果没有，则会从零开始训练。
2. 这是我训练后的模型：https://pan.baidu.com/s/1TngW_BDjkFH9rBW5jrCikg
下载后在/MT下新建training_checkpoints目录，将下载的文件放在training_checkpoints目录下，即可直接用来训练或翻译
