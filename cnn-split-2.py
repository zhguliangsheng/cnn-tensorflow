# -*- coding: utf-8 -*-
"""
用CNN处理脑影像数据

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_x_tumour=pd.read_csv('tumour_3.csv',header=None)
#删除值全为零的行
df_x_tumour_1=df_x_tumour #借用一个中间变量
for i in range(0,df_x_tumour.iloc[:,0].size):
    list_row=list(set(list(df_x_tumour.iloc[i])))#每一行不重复的数字如果只有0，则该行全为0
    if(list_row==[0.0]):
        df_x_tumour_1=df_x_tumour_1.drop(i)
df_x_tumour=df_x_tumour_1
#plt.imshow(df_x_tumour.iloc[i].as_matrix().reshape((61, 73)), cmap='gray')
df_x_tumour['label']=1

df_x_NC=pd.read_csv('NC_3.csv',header=None)
#删除值全为零的行
df_x_NC_1=df_x_NC
for i in range(0,df_x_NC.iloc[:,0].size):
    list_row=list(set(list(df_x_NC.iloc[i]))) #每一行不重复的数字如果只有0，则该行全为0
    if(list_row==[0.0]):
        df_x_NC_1=df_x_NC_1.drop(i)    
df_x_NC=df_x_NC_1
#plt.imshow(df_x_NC.iloc[i].as_matrix().reshape((61, 73)), cmap='gray')
df_x_NC['label']=0
#以上整理后得到的数据不含全为0的图像

array_x_tumour=df_x_tumour.values.astype(np.float32)
array_x_NC=df_x_NC.values.astype(np.float32)

x=np.row_stack((array_x_tumour,array_x_NC))

#随机划分
X=np.delete(x,61*73,axis=1)
Y=x[:,61*73]
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

import tensorflow as tf
tf.set_random_seed(1)   #随机种子
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001  
#test数据集
test_x = X_test
test_y_begin = y_test
test_y_begin=test_y_begin.astype(np.int32)
test_y=np.zeros([len(test_y_begin),2]).astype(np.int32)
for i in range(0,len(test_y_begin)):
    if(test_y_begin[i]==1):
        test_y[(i,1)]=1
    else:
        test_y[(i,0)]=1 


tf_x = tf.placeholder(tf.float32, [None, 61*73],name='tf_x')  
image = tf.reshape(tf_x, [-1, 61, 73, 1])              # (batch, height, width, channel)
tf_y = tf.placeholder(tf.int32, [None, 2],name='tf_y')            # input y


# CNN
conv1 = tf.layers.conv2d(   # shape (61, 73, 1)
    inputs=image,
    filters=16,
    kernel_size=5,
    strides=1,
    padding='valid',
    activation=tf.nn.relu
)           # -> (57, 69, 16)
print(conv1)
pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=2,
    strides=2,
)           # -> (28, 34, 16)
print(pool1)
conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'valid', activation=tf.nn.relu)    # -> (24, 30, 32)
print(conv2)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (12, 15, 32)
print(pool2)
conv3 = tf.layers.conv2d(pool2, 16, 3, 1, 'valid', activation=tf.nn.relu)    # -> (10, 13, 16)
print(conv3)
pool3 = tf.layers.max_pooling2d(conv3, 2, 2)    # -> (5, 6, 16)
print(pool3)
print(pool3.shape[1]*pool3.shape[2]*pool3.shape[3])
flat = tf.reshape(pool3, [-1, 5*6*16],name='flat')          # -> (5*6*32, )
print(flat)
output = tf.layers.dense(flat, 2,name='output')              # output layer
print(output)


loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

import datetime
starttime = datetime.datetime.now()
now=datetime.datetime.now()
print(now.strftime('%c'))

a=np.column_stack((X_train,y_train))
train_x=np.delete(a,61*73,axis=1)
train_y_all = a[:,61*73]
train_y_all=train_y_all.astype(np.int32)
train_y=np.zeros([len(train_y_all),2]).astype(np.int32)    
for i in range(0,len(train_y_all)):
    if(train_y_all[i]==1):
        train_y[(i,1)]=1
    else:
        train_y[(i,0)]=1 

for step in range(10000):
    np.random.shuffle(a)
    b_x_all=np.delete(a,61*73,axis=1)
    b_y_all = a[:,61*73]
    b_y_all=b_y_all.astype(np.int32)
    b_x=b_x_all[0:BATCH_SIZE,:]
    b_y_simple=b_y_all[0:BATCH_SIZE]
    b_y=np.zeros([len(b_y_simple),2]).astype(np.int32)
    
    for i in range(0,len(b_y_simple)):
        if(b_y_simple[i]==1):
            b_y[(i,1)]=1
        else:
            b_y[(i,0)]=1 
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 50 == 0:
        accuracy_batch, flat_representation_batch = sess.run([accuracy, flat], {tf_x: b_x, tf_y: b_y})
        accuracy_train, flat_representation_train = sess.run([accuracy, flat], {tf_x: train_x, tf_y: train_y})
        accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
        print('Step:', step, '| train loss: %.4f' % loss_, '|batch accuracy:%.4f' % accuracy_batch, '|train accuracy:%.4f' % accuracy_train, '| test accuracy: %.4f' % accuracy_)
        #写入到acc.txt
        with open('acc_2.txt','a') as f:
            f.write('step:')
            f.write(str(step))
            f.write(' ')
            f.write('train loss:')
            f.write(str(loss_))
            f.write(' ')
            f.write('batch accuracy:')
            f.write(str(accuracy_batch))
            f.write(' ')
            f.write('train accuracy:')
            f.write(str(accuracy_train))
            f.write(' ')
            f.write('test accuracy:')
            f.write(str(accuracy_))
            f.write(' ')
            f.write('\n')
            
import datetime
now=datetime.datetime.now()
print(now.strftime('%c'))
endtime = datetime.datetime.now()
print("%s秒" %((endtime - starttime).seconds))
#模型保存
saver=tf.train.Saver()
saver.save(sess, "saved_model/split_model_2")


#输出前十个数据的预测结果
test_output = sess.run(output, {tf_x: test_x[:10]})
pred_y = np.argmax(test_output, 1)
print(pred_y, 'prediction number')
print(np.argmax(test_y[:10], 1), 'real number')
    
