import tensorflow as tf
import input as ip

def variable_summaries(var):
    # 変数Summaries
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)                      #Scalar出力(平均)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)             #Scalar出力(標準偏差)
        tf.summary.scalar('max', tf.reduce_max(var))    #Scalar出力(最大値)
        tf.summary.scalar('min', tf.reduce_min(var))    #Scalar出力(最小値)
        tf.summary.histogram('histogram', var)          #ヒストグラム出力
        

def _variable_with_weight_decay(name, shape, stddev, wd):#L2正則化のための関数
    var = tf.get_variable(name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def l_conv(element,size,chanel,kinds,l_name):
    with tf.variable_scope(l_name) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[size, size, chanel, kinds], stddev=0.02, wd=0.0)
        conv = tf.nn.conv2d(element, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[kinds], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(bias, name=scope.name)
        variable_summaries(conv)
        return conv

def l_full(element,con_from,con_to,l_name):
    with tf.variable_scope(l_name) as scope:
        weights = _variable_with_weight_decay('weights', shape=[con_from, con_to], stddev=0.02, wd=0.0)
        biases = tf.get_variable('biases', shape=[con_to], initializer=tf.constant_initializer(0.0))
        fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(element, weights), biases), name=scope.name)
        variable_summaries(fc)
        return fc

def model(images,NUM_CLASS):
    conv1_1=l_conv(images,3,3,64,"conv1_1")
    conv1_2=l_conv(conv1_1,3,64,64,"conv1_2")
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    conv2_1=l_conv(pool1,3,64,128,"conv2_1")
    conv2_2=l_conv(conv2_1,3,128,128,"conv2_2")
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    conv3_1=l_conv(pool2,3,128,256,"conv3_1")
    conv3_2=l_conv(conv3_1,3,256,256,"conv3_2")
    conv3_3=l_conv(conv3_2,3,256,256,"conv3_3")
    pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    conv4_1=l_conv(pool3,3,256,512,"conv4_1")
    conv4_2=l_conv(conv4_1,3,512,512,"conv4_2")
    conv4_3=l_conv(conv4_2,3,512,512,"conv4_3")
    pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    shape = pool4.get_shape()[1::2].as_list()#pool4のサイズとチャネルを取得
    dim = (shape[0]**2)*shape[1]             #pool4の要素数を取得
    reshape = tf.reshape(pool4,[50,dim])

    fc1=l_full(reshape,dim,2048,"fc1")
    fc2=l_full(fc1,2048,1024,"fc2")
    fc3=l_full(fc2,1024,256,"fc3")

    with tf.variable_scope("fc4") as scope:
        weights = tf.get_variable("weight", shape=[256, NUM_CLASS], initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable("biases", shape=[NUM_CLASS], initializer=tf.constant_initializer(0.0))
        fc4 = tf.nn.bias_add(tf.matmul(fc3, weights), biases, name=scope.name)
        variable_summaries(fc4)

    return fc4
