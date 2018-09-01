import tensorflow as tf
import tensorlayer as tl

from tensorlayer.layers import *
import numpy as np
import time,os,io
from PIL import Image

sess = tf.InteractiveSession()
x_train,y_train,x_test,y_test = tl.files.load_cifar10_dataset(shape=(-1,32,32,3),plotable=False)

def model(x,y_,reuse):
    W_init = tf.truncated_normal_initializer(stddev=5e-2)
    W_init2 = tf.truncated_normal_initializer(stddev=0.04)
    b_init2 = tf.constant_initializer(value=0.1)

    with tf.variable_scope('model',reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net = InputLayer(x,name='input')

        #CNN
        net = Conv2d(net,64,(5,5),(1,1),act=tf.nn.relu,padding='SAME',W_init=W_init,name='cnn1')
        net = MaxPool2d(net,(3,3),(2,2),padding='SAME',name='pool1')
        #局部响应归一化
        net = LocalResponseNormLayer(net,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm1')

        net = Conv2d(net,(5,5),(1,1),act=tf.nn.relu,padding='SAME',W_init=W_init,name='cnn2')
        net = LocalResponseNormLayer(net,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm2')
        net = MaxPool2d(net,(3,3,),(2,2),padding='SAME',name='pool2')

        net = FlattenLayer(net,name='flatten')
        net = DenseLayer(net,384,tf.nn.relu,W_init=W_init2,b_init=b_init2,name='d1relu')
        net = DenseLayer(net,192,tf.nn.relu,W_init=W_init2,b_init=b_init2,name='d2relu')
        net = DenseLayer(net,192,tf.identity,W_init=tf.truncated_normal_initializer(stddev=1/192.0),name='output')

        y = net.outputs

        ce = tl.cost.cross_entropy(y,y_,name='cost')

        L2 = 0
        for p in tl.layers.get_layers_with_name('relu/W',True,True):
            L2 += tf.contrib.layers.l2_regularizer(0.004)(p)

        cost = ce+L2

        correct = tf.equal(tf.argmax(y,1),y_)

        acc = tf.reduce_mean(tf.cast(correct,tf.float32))

        return net,cost,acc


def model_batch_norm(x,y_,reuse,is_train):
    W_init = tf.truncated_normal_initializer(stddev=5e-2)
    W_init2 = tf.truncated_normal_initializer(stddev=0.04)
    b_init2 = tf.constant_initializer(value=0.1)

    with tf.variable_scope('model',reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net = InputLayer(x,name='input')

        #CNN
        net = Conv2d(net,64,(5,5),(1,1),act=tf.nn.relu,padding='SAME',W_init=W_init,b_init=None,name='cnn1')
        # 批归一话
        net = BatchNormLayer(net, is_train, act=tf.nn.relu, name='batch1')
        net = MaxPool2d(net,(3,3),(2,2),padding='SAME',name='pool1')


        net = Conv2d(net,64,(5,5),(1,1),act=tf.nn.relu,padding='SAME',W_init=W_init,b_init=None,name='cnn2')
        net = BatchNormLayer(net,is_train,act=tf.nn.relu,name='batch2')
        net = MaxPool2d(net,(3,3,),(2,2),padding='SAME',name='pool2')

        net = FlattenLayer(net,name='flatten')
        net = DenseLayer(net,384,tf.nn.relu,W_init=W_init2,b_init=b_init2,name='d1relu')
        net = DenseLayer(net,192,tf.nn.relu,W_init=W_init2,b_init=b_init2,name='d2relu')
        net = DenseLayer(net,192,tf.identity,W_init=tf.truncated_normal_initializer(stddev=1/192.0),name='output')

        y = net.outputs

        ce = tl.cost.cross_entropy(y,y_,name='cost')

        L2 = 0
        for p in tl.layers.get_variables_with_name('relu/W',True,True):
            L2 += tf.contrib.layers.l2_regularizer(0.004)(p)

        cost = ce+L2

        correct = tf.equal(tf.argmax(y,1),y_)

        acc = tf.reduce_mean(tf.cast(correct,tf.float32))

        return net,cost,acc

#数据增强
def distort_fn(x,is_train=False):

    #当不是训练时，则截取中央24x24的子图像
    x = tl.prepro.crop(x,24,24,is_random=is_train)

    #当训练时，随机左右翻转和亮度调整
    if is_train:
        x = tl.prepro.flip_axis(x,axis=1,is_random=True)    #左右翻转
        x = tl.prepro.brightness(x,gamma=0.1,gain=1,is_random=True) #亮度调整

    #把像素值规范化
    x = (x-np.mean(x))/max(np.std(x),1e-5) #np.std(a)：标准差
    return x

x = tf.placeholder(tf.float32,shape=[None,24,24,3],name='x')
y_= tf.placeholder(tf.int64,shape=[None,],name='y_')

network,cost,acc = model_batch_norm(x,y_,False,is_train=True)
_,cost_test,acc = model_batch_norm(x,y_,True,is_train=False)

#训练参数
n_epoch = 50000
learning_rate = 0.0001
print_frep = 1
batch_size = 128

train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False).minimize(cost,var_list=train_params)

tl.layers.initialize_global_variables(sess)
network.print_params(False)
network.print_layers()

print('learning_rate:%f' % learning_rate)
print('batch_size:%d' % batch_size)

for epoch in range(n_epoch):
    start_time = time.time()

    for x_train_a,y_train_a in tl.iterate.minibatches(x_train,y_train,batch_size,shuffle=True):
        x_train_a = tl.prepro.threading_data(x_train_a,fn=distort_fn,is_train=True)

        sess.run(train_op,feed_dict={x:x_train_a,y_:y_train_a})


    if epoch+1 == 1 or (epoch+1) % print_frep == 0:
        print('Epoch %d of %d took %fs',(epoch+1,n_epoch,time.time()-start_time))
        test_loss,test_acc,n_batch = 0,0,0
        for x_test_a, y_test_a in tl.iterate.minibatches(x_test, y_test, batch_size, shuffle=True):
            x_test_a = tl.prepro.threading_data(x_test_a, fn=distort_fn, is_train=False)
            err,ac = sess.run([cost_test,acc], feed_dict={x: x_test_a, y_: y_test_a})
            test_loss += err
            test_acc += ac
            n_batch+=1

        print('test lose:%f' % (test_loss/n_batch))
        print('test acc:%f' % (test_acc/n_batch))