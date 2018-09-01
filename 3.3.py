import tensorflow as tf
import tensorlayer as tl

#定义超参数
learning_rate = 0.0001
lambda_l2_w = 0.01
n_epochc = 200
batch_size = 128
print_interval = 200

#模型结构参数
hidden_size = 196
input_size = 784
image_width = 28
model = 'sigmoid'

x = tf.placeholder(tf.float32,shape=[None,784],name='x')

print('Build Network')

if model == 'relu':
    #输入层
    network = tl.layers.InputLayer(x,name='input')
    network = tl.layers.DenseLayer(network,hidden_size,tf.nn.relu,name='relu1')
    #隐层输出
    encoded_img = network.outputs
    #重构层输出
    recon_layer1 = tl.layers.DenseLayer(encoded_img,input_size,tf.nn.softplus,name='recon_layer1')

elif model =='sigmoid':
    # 输入层
    network = tl.layers.InputLayer(x, name='input')
    network = tl.layers.DenseLayer(network, hidden_size, tf.nn.sigmoid, name='sigmoid1')
    # 隐层输出
    encoded_img = network.outputs
    # 重构层输出
    recon_layer1 = tl.layers.DenseLayer(encoded_img, input_size, tf.nn.sigmoid(), name='recon_layer1')


y = recon_layer1.outputs
train_params = recon_layer1.all_params[-4:]
