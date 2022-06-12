# MyAIDemo

#### 一、 项目介绍

将Machine Learning部署到实际项目中，做一个Demo

#### 二、 技术选型：

##### 前端：

1. HTML+CSS+JavaScript
2. vue.js
3. element-ui
4. vue-ele-form
5. axios
6. webpack

##### 后端：

1. Python
2. Flask

2. PaddlePaddle

3. music21

4. HuggingFace

##### 部署：

0. 前后端分离

1. Github Pages
2. Tencent Lighthouse
3. HuggingFace model API

#### 三、后端开发

1. 音乐生成

   使用paddle框架运行model，flask框架处理API请求，部署在Tencent Lighthouse上

   项目结构：

   ```markdown
   |-- music
       |-- dur_dic.json
       |-- Reader.py
       |-- scratch_17.py
       |-- Seq2Seq.py
       |-- Duration_Model
       |   |-- best_model
       |   |-- final_model
       |-- log
       |-- Midi_Model
       |   |-- best_model
       |   |-- final_model
       |-- results
       |-- tools
       |   |-- CreateXML.py
       |-- work
       |   |-- data
       |       |-- 一分钱.xml
       |       |-- 上学.xml
       |       |-- 上学歌.xml
       |       |-- 两只老虎.xml
       |       |-- 偏爱.xml
       |       |-- 圣诞快乐.xml
       |       |-- 天上星星.xml
       |       |-- 天空.xml
       |       |-- 好娃娃.xml
       |       |-- 小兔子乖乖.xml
       |       |-- 小小姑娘.xml
       |       |-- 小小的船.xml
       |       |-- 小星星.xml
       |       |-- 小树变成一只小鸟.xml
       |       |-- 小毛驴.xml
       |       |-- 小老鼠.xml
       |       |-- 找朋友.xml
       |       |-- 浇树.xml
       |       |-- 生日歌.xml
       |       |-- 粉刷匠.xml
       |-- __pycache__
           |-- Reader.cpython-37.pyc
           |-- Reader.cpython-38.pyc
           |-- Seq2Seq.cpython-37.pyc
           |-- Seq2Seq.cpython-38.pyc
   ```

   main.py

   ```python
   import os
   import json
   from flask import Flask, render_template, send_from_directory, request, jsonify
   from music21 import *
   
   import base64
   import paddle
   import paddle.nn as nn
   import numpy as np
   from Reader import Reader
   import Seq2Seq
   from flask import Flask, request
   from Crypto.Cipher import AES
   from flask_cors import *
   from binascii import b2a_hex
   app = Flask(__name__)
   CORS(app, supports_credentials=True) #跨域设置
   result = {
           'result': "error"
       }
   batch_size = 10
   train_reader = Reader(batch_size, './work/data')
   import json
   import time
   # 初始化log写入器
   
   # 模型参数设置
   embedding_size = 256
   hidden_size = 256
   num_layers = 1
   
   # 训练参数设置
   epoch_num = 5000
   learning_rate = 1e-5
   log_iter = 200
   
   # 定义一些所需变量
   global_step = 0
   log_step = 0
   max_acc = 0
   
   midi_model = Seq2Seq.Midi_Model(
       char_len=0x9FFF,  # 基本汉字的Unicode码范围为4E00-9FA5,这里设置0x9FFF长，基本够用
       embedding_size=embedding_size,
       hidden_size=hidden_size,
       num_layers=num_layers,
       batch_size=batch_size)
   dur_model = Seq2Seq.Duration_Model(
       char_len=200,  # midi范围一般在100左右,这里设置200长，基本够用
       embedding_size=embedding_size,
       hidden_size=hidden_size,
       num_layers=num_layers,
       batch_size=batch_size)
   midi_model.set_state_dict(paddle.load('Midi_Model/final_model'))
   dur_model.set_state_dict(paddle.load('Duration_Model/final_model'))
   def jiemihanshu(mima, readdata, iv):
       secret = mima  # 由用户输入的16位或24位或32位长的初始密码字符串
       iv = iv.encode('UTF-8')  # 随机获取16位变量
       encrypt_data = bytes().fromhex(readdata)
       cipher = AES.new(secret.encode('UTF-8'), AES.MODE_CBC, iv)
       decrypt_data = cipher.decrypt(encrypt_data)
       return decrypt_data
   def valkey(time_en):
       if int(round(time.time() * 1000))-(int(jiemihanshu('20060815200608152006081520060815', bytes.decode(b2a_hex(base64.b64decode(time_en))),'2006081520060815').decode('UTF-8')[0:13]))<=30000:
           return True
       else:
           return False
   
   @app.route('/api', methods=['GET'])
   def test_get():
       # 解析请求参数
       param = request.args.to_dict()
       print(param)
       key = param['time']
       input = param['input']
       if valkey(key):
           if len(input) <= 100000:
               input_lyrics = input
               print(type(input))
               lyrics = []
               for i, lyric in enumerate(input_lyrics.replace('\n', '')):
                   if i % batch_size == 0:
                       lyrics.append([])
                   lyrics[i // batch_size].append(ord(lyric))
               while len(lyrics[-1]) % batch_size != 0:
                   lyrics[-1].append(ord('#'))
               lyrics = paddle.to_tensor(lyrics)
   
               params_dict = paddle.load('Midi_Model/best_model')
               midi_model.set_dict(params_dict)
   
               # 设置为评估模式
               midi_model.eval()
   
               # 模型推理
               out = midi_model(lyrics)
   
               # 结果转换
               results = []
               for _ in np.argmax(out.numpy(), -1).reshape(-1):
                   results.append(_)
   
               midis = []
               dur_dic = {}
               with open('dur_dic.json', 'r') as f:
                   dur_str = f.readline()
                   dur_dic = json.loads(dur_str)
               for i, midi in enumerate(results):
                   if i % batch_size == 0:
                       midis.append([])
                   midis[i // batch_size].append(midi) if midi <= 200 else midis[i // batch_size].append(0)
               while len(midis[-1]) % batch_size != 0:
                   midis[-1].append(0)
               midis = paddle.to_tensor(midis)
   
               params_dict = paddle.load('Duration_Model/best_model')
               dur_model.set_dict(params_dict)
   
               # 设置为评估模式
               dur_model.eval()
   
               # 模型推理
               # out = nn.Softmax(dur_model(midis))
               out = dur_model(midis)
   
               # 结果转换
               durations = []
               for _ in np.argmax(out.numpy(), -1).reshape(-1):
                   durations.append(_)
   
               dur_dic = {}
               with open('dur_dic.json', 'r') as f:
                   dur_str = f.readline()
                   dur_dic = json.loads(dur_str)
                   print(dur_dic)
   
               stream1 = stream.Stream()
               for i, lyric in enumerate(input_lyrics.replace('\n', '')):
                   if results[i] != 0:
                       n1 = note.Note(results[i])
                   else:
                       n1 = note.Rest()
                   n1.addLyric(lyric)
                   n1.duration = duration.Duration(dur_dic[str(durations[i])])
                   stream1.append(n1)
               import random
               name = ''
               for i in range(8):
                   name += str(random.randint(0, 9))
               stream1.write("xml", './results/' + name + ".xml")
               stream1.write('midi', './results/' + name + '.midi')
               output = 'http://82.157.179.249:8080' + '/download/' + name + '.midi'
               result['result'] = output
           else:
               result['result'] = 'too lang'
       else:
           result['result'] = 'the key is wrong'
       # 返回json
       result_json = json.dumps(result)
       return result_json
   
   
   
   
   
   @app.route("/download/<path:filename>") #下载文件请求
   def downloader(filename):
       dirpath = os.path.join(app.root_path+'/results') 
       return send_from_directory(dirpath, filename, as_attachment=True)  # as_attachment=True 一定要写，不然会变成打开，而不是下载
   
   app.run(debug=True, host='0.0.0.0', port=8080)
   
   
   
   ```

   Seq2Seq.py

   ```python
   import paddle
   import paddle.nn as nn
   
   # 继承paddle.nn.Layer类
   class Midi_Model(nn.Layer):
       # 重写初始化函数
       # 参数：字符表长度、嵌入层大小、隐藏层大小、解码器层数、处理数字的最大位数
       def __init__(self, char_len, embedding_size=128, hidden_size=128, num_layers=1, batch_size=20):
           super(Midi_Model, self).__init__()
           # 初始化变量
           self.MAXLEN = 1
           self.batch_size = batch_size
           self.hidden_size = hidden_size
           self.char_len = char_len
           self.num_layers=num_layers
           self.embedding_size=embedding_size
   
           # 嵌入层
           self.emb = nn.Embedding(
               char_len,
               self.embedding_size
           )
   
           # 编码器
           self.encoder = nn.LSTM(
               input_size=self.embedding_size,
               hidden_size=self.hidden_size,
               num_layers=self.num_layers
           )
   
           # 解码器
           self.decoder = nn.LSTM(
               input_size=self.hidden_size,
               hidden_size=self.hidden_size,
               num_layers=self.num_layers
           )
   
           # 全连接层
           self.fc = nn.Linear(
               self.hidden_size,
               char_len
           )
   
       # 重写模型前向计算函数
       # 参数：输入[None, MAXLEN]、标签[None, DIGITS]
       def forward(self, inputs, labels=None):
           # 嵌入层
           out = self.emb(inputs)
   
           # 编码器
           out, (_, _) = self.encoder(out)
   
           # 按时间步切分编码器输出
           out = paddle.split(out, self.MAXLEN, axis=1)
   
           # 取最后一个时间步的输出并复制batch_size次
           out = paddle.expand(out[-1], [out[-1].shape[0], self.batch_size, self.hidden_size])
   
           # 解码器
           out, (_, _) = self.decoder(out)
   
           # 全连接
           out = self.fc(out)
   
           # 如果标签存在，则计算其损失和准确率
           if labels is not None:
               # 转置解码器输出
               tmp = paddle.transpose(out, [0, 2, 1])
   
               # 计算交叉熵损失
               loss = nn.functional.cross_entropy(tmp, labels, axis=1)
   
               # 计算准确率
               acc = paddle.metric.accuracy(paddle.reshape(out, [-1, self.char_len]), paddle.reshape(labels, [-1, 1]))
   
               # 返回损失和准确率
               return loss, acc
   
           # 返回输出
           return out
   
   
   # 继承paddle.nn.Layer类
   class Duration_Model(nn.Layer):
       # 重写初始化函数
       # 参数：字符表长度、嵌入层大小、隐藏层大小、解码器层数、处理数字的最大位数
       def __init__(self, char_len, embedding_size=128, hidden_size=64, num_layers=1, batch_size=20):
           super(Duration_Model, self).__init__()
           # 初始化变量
           self.batch_size = batch_size
           self.MAXLEN = 1
           self.hidden_size = hidden_size
           self.char_len = char_len
           self.num_layers=num_layers
           self.embedding_size=embedding_size
   
           # 嵌入层
           self.emb = nn.Embedding(
               self.char_len,
               self.embedding_size
           )
   
           # 编码器
           self.encoder = nn.LSTM(
               input_size=embedding_size,
               hidden_size=self.hidden_size,
               num_layers=self.num_layers
           )
   
           # 解码器
           self.decoder = nn.LSTM(
               input_size=self.hidden_size,
               hidden_size=self.hidden_size,
               num_layers=self.num_layers
           )
   
           # 全连接层
           self.fc = nn.Linear(
               self.hidden_size,
               self.char_len
           )
   
       # 重写模型前向计算函数
       # 参数：输入[None, MAXLEN]、标签[None, DIGITS]
       def forward(self, inputs, labels=None):
           # 嵌入层
           out = self.emb(inputs)
   
           # 编码器
           out, (_, _) = self.encoder(out)
   
           # 按时间步切分编码器输出
           out = paddle.split(out, self.MAXLEN, axis=1)
   
           # 取最后一个时间步的输出并复制batch_size次
           out = paddle.expand(out[-1], [out[-1].shape[0], self.batch_size, self.hidden_size])
   
           # 解码器
           out, (_, _) = self.decoder(out)
   
           # 全连接
           out = self.fc(out)
   
           # 如果标签存在，则计算其损失和准确率
           if labels is not None:
               # 转置解码器输出
               tmp = paddle.transpose(out, [0, 2, 1])
   
               # 计算交叉熵损失
               loss = nn.functional.cross_entropy(tmp, labels, axis=1)
   
               # 计算准确率
               acc = paddle.metric.accuracy(paddle.reshape(out, [-1, self.char_len]), paddle.reshape(labels, [-1, 1]))
   
               # 返回损失和准确率
               return loss, acc
   
           # 返回输出
           return out
   ```

   Reader.py

   ```python
   from music21 import note,converter
   import numpy as np
   import os
   import json
   import fractions
   
   def Reader(DIGITS,path = './work/data'):
       dur_dic = {}
       def read_data():
           for file in os.listdir(path):
               lyrics = []
               midis = []
               durations = []
               xml = converter.parseFile(os.path.join(path,file))
               #print(dir(stream.Score()))
               for i, note in enumerate(xml.recurse().notesAndRests):
                   if i%DIGITS == 0:
                       lyrics.append([])
                       midis.append([])
                       durations.append([])
                   lyric = note._getLyric()
                   if lyric == None:
                       lyric = '#'
                   lyrics[i//DIGITS].append(ord(lyric))
                   try:
                       midis[i//DIGITS].append(note.pitch.midi)
                   except:
                       midis[i//DIGITS].append(0)
                   durations[i//DIGITS].append(note.duration.quarterLength)
                   if type(note.duration.quarterLength) == fractions.Fraction and float(note.duration.quarterLength) not in list(dur_dic.values()):
                           dur_dic[len(dur_dic)] = float(note.duration.quarterLength)
                   elif type(note.duration.quarterLength) != fractions.Fraction and note.duration.quarterLength not in list(dur_dic.values()):
                       dur_dic[len(dur_dic)] = note.duration.quarterLength
               yield [midis,durations,lyrics]
           with open('dur_dic.json','w') as f:
               f.write(json.dumps(dur_dic))
       return read_data
   ```

2. 诗歌生成、对联生成、古文生成、歌词生成、中文阅读理解、英文阅读理解

   部署在HuggingFace Model上，使用其Inference API

#### 四、 前端开发

使用vue.js生成静态页面，部署在GitHub Pages上

1. 音乐生成

   项目结构：

   ```markdown
   |-- music
       |-- .babelrc
       |-- .gitignore
       |-- babel.config.js
       |-- deploy.sh
       |-- jsconfig.json
       |-- package-lock.json
       |-- package.json
       |-- README.md
       |-- vue.config.js
       |-- webpack.config.js
       |-- dist
       |   |-- favicon.ico
       |   |-- index.html
       |   |-- css
       |   |   |-- app.01eeae60.css
       |   |-- fonts
       |   |   |-- element-icons.f1a45d74.ttf
       |   |   |-- element-icons.ff18efd1.woff
       |   |-- js
       |       |-- app-legacy.954daf1b.js
       |       |-- app-legacy.954daf1b.js.LICENSE.txt
       |       |-- app-legacy.954daf1b.js.map
       |       |-- app.5969f770.js
       |       |-- app.5969f770.js.LICENSE.txt
       |       |-- app.5969f770.js.map
       |-- public
       |   |-- favicon.ico
       |   |-- index.html
       |-- src
           |-- App.vue
           |-- main.js
           |-- utils.js
           |-- assets
           |   |-- logo.png
           |-- components
               |-- App.vue
               |-- HelloWorld.vue
   
   ```

   app.vue

   ```vue
   <template>
     <div id="vue">
     <div class="outer"><p style="margin-top: 0px;margin-bottom: 0px;"><p321>&nbsp;</p321></p></div>
     <div class="outer"><p style="margin-top: 0px;margin-bottom: 0px;"><p123>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;xuqinyang</p123></p></div>
     <div class="outer"><p style="margin-top: 0px;margin-bottom: 0px;"><p321>&nbsp;</p321></p></div>
     <div id ="form">
     <ele-form
       v-bind="formConfig"
       v-model="formData"
       :request-fn="handleRequest"
       @request-success="handleRequestSuccess"
     />
     </div>
       </div>
   </template>
   <style>
   p {
       display: block;
       margin-block-start: 0em;
       margin-block-end: 0em;
       margin-inline-start: 0px;
       margin-inline-end: 0px;
       letter-spacing:5px;
   }
   body {
       display: block;
       margin: 0px;
       margin-top: 0px;
       margin-right: 0px;
       margin-bottom: 0px;
       margin-left: 0px;
       background-color: #F2F6FC;
   }
   p123 {
   font-size:30px;
   margin-top: 0px;
   color: #FFFFFF;
   }
   p321 {
   font-size:0px;
   color: #FFFFFF;
   }
   .outer {
   	line-height:15px;
         width: 100%;
         height: 30%;
         background-color: #409EFF; 
       }
   .form{
   	background-color: #409EFF;
   }
   </style>
   <script>
   import axios from 'axios';
   import Utils from '@/utils.js'
   export default {
     data() {
       var validQC = (rule, value, callback) => {
         if (value) {
           if (/[\u0020-\u4DFF]/g.test(value)) {
   		//this.$set(this.formData, "input",value.replace(/[^\u4e00-\u9fa5]/g, ''));
   	      callback(new Error("只能输入汉字"));
   		}
           } else {
           // 验证通过
             callback();
           }
   	  if (/[\u9FA6-\uFFFF]/g.test(value)) {
   		//this.$set(this.formData, "input",value.replace(/[^\u4e00-\u9fa5]/g, ''));
             	callback(new Error("只能输入汉字"));
   		} else {
   		    callback();
           callback();
         }
       };
       return {
         formData: {},
         formConfig: {
           order: [
             "time",
             "input",
             "mybutton",
           ],
           rules: {
   		input: [{ required: true, type: 'string', message: '必须填写歌词' },{validator:validQC,message: "请输入中文歌词，无标点"}]
           },
   	  labelPosition: "top",
           formDesc: {
             time: {
               type: "text",
               label: "",
               isOptions: false,
               default: "text→music:一个可以自动根据歌词进行谱曲的AI"+'\r\n'+"在20首儿歌的数据集上训练了10K step得到的模型"
             },
             input: {
               type: "textarea",
               label: "请输入要谱曲的歌词（只能是中文，无标点）",
             },
   	    mybutton: {
               type: "button",
               label: "按钮",
               default: "去除所有非中文字符",
               isShowLabel: false,
               on: {
         // 按钮事件触发
         click: () => {
           this.$set(this.formData, "input",this.formData.input.replace(/[^\u4e00-\u9fa5]/g, ''));
               },
               attrs: {
                 round: false,
                 plain: true,
                 circle: false,
                 nativeType: "button",
                 type: "info"
               }
               }
           }
           }
         }
       };
     },
     methods: {
       handleRequest(data) {
   let loading = this.$loading({
           lock: true,//lock的修改符--默认是false
           text: "加载中，请稍候...",//显示在加载图标下方的加载文案
           background: "rgba(255,255,255,0.8)",//遮罩层颜色
         });
   	axios.defaults.withCredentials = true;
   	const path = 'http://xx.xxx.xxx.xxx:8080/api';
   	this.$set(data, "time",Utils.Encrypt(new Date().getTime()));
         axios.get(path,{params:data})
         .then(response=>{
   
     	setTimeout(() => {
   	window.location.href=response.data.result
         loading.close();
     	}, 3000)
   
         });
         return Promise.resolve();
       },
       handleRequestSuccess() {
         this.$message.success("提交成功");
       }
     }
   };
   </script>
   
   ```

   utils.js

   ```javascript
   import CryptoJS from 'crypto-js/crypto-js'
   
   const KEY = CryptoJS.enc.Utf8.parse('20060815200608152006081520060815')
   const IV = CryptoJS.enc.Utf8.parse('2006081520060815')
   export default{
    Encrypt (word, keyStr, ivStr) {
     let key = KEY
     let iv = IV
     if (keyStr) {
       key = CryptoJS.enc.Utf8.parse(keyStr)
       iv = CryptoJS.enc.Utf8.parse(ivStr)
     }
     let srcs = CryptoJS.enc.Utf8.parse(word)
     var encrypted = CryptoJS.AES.encrypt(srcs, key, {
       iv: iv,
       mode: CryptoJS.mode.CBC,
       padding: CryptoJS.pad.Pkcs7
     })
     // console.log("-=-=-=-", encrypted.ciphertext)
     return CryptoJS.enc.Base64.stringify(encrypted.ciphertext)
   }}
   ```

2. 诗歌生成、对联生成、古文生成、歌词生成、中文阅读理解、英文阅读理解

   待补充

#### 五、 前后端通信

##### 1. 音乐生成API

① 请求方式：GET（HTTP）

② 请求地址：http://xx.xxx.xxx.xxx:8080/api

③ 请求参数：

| 参数名 | 位置   | 类型   | 必填 | 说明                                   |
| :----- | :----- | :----- | :--: | :------------------------------------- |
| time   | params | string |  是  | 说明：加密后的时间戳                   |
| input  | params | string |  是  | 说明：要谱曲的歌词，只能是中文，无标点 |

④返回响应：

```json
{
	“result”:“http://xx.xxx.xxx.xxx:8080/download/xxxxxxxx.midi”
}
```

##### 2. 音乐下载API

① 请求方式：GET（HTTP）

② 请求地址：http://xx.xxx.xxx.xxx:8080/download/

③ 请求参数：无

④返回响应：blob对象

##### 3. 诗歌生成API

① 请求方式：POST（HTTP/HTTPS）

② 请求地址：https://api-inference.huggingface.co/models/uer/gpt2-chinese-poem

③ 请求参数：

| 参数名        | 位置 | 类型   | 必填 | 说明                   |
| :------------ | :--- | :----- | :--: | :--------------------- |
| Authorization | head | string |  是  | 说明：用户凭证         |
| inputs        | body | string |  是  | 说明：要生成诗歌的开头 |

④返回响应：

```json
{
	“generated_text”:“xxxxx,xxxxx......”
}
```

##### 4. 对联生成

① 请求方式：POST（HTTP/HTTPS）

② 请求地址：https://api-inference.huggingface.co/models/uer/gpt2-chinese-couplet

③ 请求参数：

| 参数名        | 位置 | 类型   | 必填 | 说明                   |
| :------------ | :--- | :----- | :--: | :--------------------- |
| Authorization | head | string |  是  | 说明：用户凭证         |
| inputs        | body | string |  是  | 说明：要生成对联的开头 |

④返回响应：

```json
{
	“generated_text”:“xxxxx,xxxxx......”
}
```

##### 5. 古文生成

① 请求方式：POST（HTTP/HTTPS）

② 请求地址：https://api-inference.huggingface.co/models/uer/gpt2-chinese-ancient

③ 请求参数：

| 参数名        | 位置 | 类型   | 必填 | 说明                   |
| :------------ | :--- | :----- | :--: | :--------------------- |
| Authorization | head | string |  是  | 说明：用户凭证         |
| inputs        | body | string |  是  | 说明：要生成古文的开头 |

④返回响应：

```json
{
	“generated_text”:“xxxxx,xxxxx......”
}
```

##### 6. 歌词生成

① 请求方式：POST（HTTP/HTTPS）

② 请求地址：https://api-inference.huggingface.co/models/uer/gpt2-chinese-lyric

③ 请求参数：

| 参数名        | 位置 | 类型   | 必填 | 说明                   |
| :------------ | :--- | :----- | :--: | :--------------------- |
| Authorization | head | string |  是  | 说明：用户凭证         |
| inputs        | body | string |  是  | 说明：要生成歌词的开头 |

④返回响应：

```json
{
	“generated_text”:“xxxxx,xxxxx......”
}
```

##### 7. 中文阅读理解

① 请求方式：POST（HTTP/HTTPS）

② 请求地址：https://api-inference.huggingface.co/models/uer/roberta-base-chinese-extractive-qa

③ 请求参数：

| 参数名        | 位置  | 类型   | 必填 | 说明                                   |
| :------------ | :---- | :----- | :--: | :------------------------------------- |
| Authorization | head  | string |  是  | 说明：用户凭证                         |
| inputs        | body  | json   |  是  | 说明：input                            |
| question      | input | string |  是  | 说明：要问的问题                       |
| context       | input | string |  是  | 说明：问题情景（问题的答案需在情景中） |

④返回响应：

```json
{
    'score': 0.xxxxxxxxxxxxxxxxx, 'start': x, 'end': x, 'answer': 'xxxxx......'
}

```

##### 8. 英文阅读理解

① 请求方式：POST（HTTP/HTTPS）

② 请求地址：https://api-inference.huggingface.co/models/deepset/bert-large-uncased-whole-word-masking-squad2

③ 请求参数：

| 参数名        | 位置  | 类型   | 必填 | 说明                                   |
| :------------ | :---- | :----- | :--: | :------------------------------------- |
| Authorization | head  | string |  是  | 说明：用户凭证                         |
| inputs        | body  | json   |  是  | 说明：input                            |
| question      | input | string |  是  | 说明：要问的问题                       |
| context       | input | string |  是  | 说明：问题情景（问题的答案需在情景中） |

④返回响应：

```json
{
    'score': 0.xxxxxxxxxxxxxxxxx, 'start': x, 'end': x, 'answer': 'xxxxx......'
}

```

##### 
