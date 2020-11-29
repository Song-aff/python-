import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras import Input
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from attention.attention import attention_3d_block

from math import sqrt
# keras的API文档
# https://www.tensorflow.org/versions/r1.10/api_docs/python/tf/keras/Model

# 读取数据
dataset = 11
data = pd.read_excel('D:/inputdata2.xlsx')
data = data.iloc[1:3500, dataset-1:dataset].values
print(len(data))
# 划分训练集和测试集
data_length = len(data)
train_length = int(data_length*0.67)
test_length = len(data) - train_length
# print(train_length, test_length)

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data) # 映射数据到[0,1]之间

# 转换训练数据
features_set = []
labels = []
features_length = 50 #将前100天的数据作为特征集，第101天的数据作为标签，创建特征和标签集\n",
for i in range(features_length, train_length):
    features_set.append(data_scaled[i-features_length:i, 0])
    labels.append(data_scaled[i, 0])
features_set, labels = np.array(features_set), np.array(labels)
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
print('kkkkk')
print(features_set)
print(features_set.shape)
print(features_set.shape[1])


i = Input(shape=(features_set.shape[1], 1))
print(i)
LSTM1 = LSTM(32, return_sequences=True)(i)
# LSTM1 = Dropout(0.2)(LSTM1)
LSTM2 = LSTM(32,  return_sequences=True)(LSTM1)
# LSTM2 = Dropout(0.2)(LSTM2)
LSTM3 = LSTM(32,  return_sequences=True)(LSTM2)
# print(LSTM3.shape) (?,50,32)
att = attention_3d_block(LSTM3)

# print(att.shape)
# att = Dropout(0.2)(att)
out = Dense(1, activation='linear')(att)
model = Model(inputs=[i], outputs=[out])
model.compile(loss='mse', optimizer='adam')
print(model.summary())
model.fit(x=features_set, y=labels,   # 指定训练数据
          batch_size=32,     # batch大小为100
          epochs=1,   # 迭代100轮
          )




# 测试
test_features = []
for i in range(train_length, data_length):
    test_features.append(data_scaled[i - features_length:i, 0])
test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
predictions = model.predict(test_features)
predictions = scaler.inverse_transform(predictions)
plt.figure(figsize=(6, 4))
plt.plot(data[train_length:data_length, 0], color='blue', label='Actual')
plt.plot(predictions, color='red', label='Predicted')
plt.ylabel('mine_water_inflow')
plt.xlabel('LSTM_Attention')
plt.legend()
# plt.savefig('LSTM_Attentio预测模型2_2'+'.png')
plt.show()
mse = mean_squared_error(data[train_length:data_length, 0], predictions)
rmse = sqrt(mse)
r2 = r2_score(data[train_length:data_length, 0], predictions)
# f = open('test.txt','a+')
# f.write('LSTM+AttRMSE:'+'\n')
# f.write("RMSE"+'\t')
# f.write(str(rmse)+'\n')
# f.write('r2'+'\t')
# f.write(str(r2)+'\n')
# f.close()
print('RMSE: %.3f' % rmse)
print('R^2: %.3f' % r2)

