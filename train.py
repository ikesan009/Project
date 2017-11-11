import tensorflow as tf
import os
import shutil
import input
sess = tf.InteractiveSession()

log_dir = 'logs/Project'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)

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

x = tf.placeholder(tf.float32, shape=[None,56,56,3])
y_ = tf.placeholder(tf.float32, shape=[None,5])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

with tf.name_scope('W_conv1'):
    W_conv1 = weight_variable([5,5,3,32])
    variable_summaries(W_conv1)
with tf.name_scope('b_conv1'):
    b_conv1 = bias_variable([32])
    variable_summaries(b_conv1)

with tf.name_scope('h_conv1'):
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    variable_summaries(h_conv1)
with tf.name_scope('h_pool1'):
    h_pool1 = max_pool_2x2(h_conv1)
    variable_summaries(h_pool1)

with tf.name_scope('W_conv2'):
    W_conv2 = weight_variable([5,5,32,64])
    variable_summaries(W_conv2)
with tf.name_scope('b_conv2'):
    b_conv2 = bias_variable([64])
    variable_summaries(b_conv2)

with tf.name_scope('h_conv2'):
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    variable_summaries(h_conv2)
with tf.name_scope('h_pool2'):
    h_pool2 = max_pool_2x2(h_conv2)
    variable_summaries(h_pool2)

with tf.name_scope('W_fc1'):
    W_fc1 = weight_variable([14*14*64,1024])
    variable_summaries(W_fc1)
with tf.name_scope('b_fc1'):
    b_fc1 = bias_variable([1024])
    variable_summaries(b_fc1)

with tf.name_scope('h_fc1'):
    h_pool2_flat = tf.reshape(h_pool2, [-1, 14*14*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    variable_summaries(h_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('W_fc2'):
    W_fc2 = weight_variable([1024,5])
    variable_summaries(W_fc2)
with tf.name_scope('b_fc2'):
    b_fc2 = bias_variable([5])
    variable_summaries(b_fc2)

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    variable_summaries(cross_entropy)
with tf.name_scope('accuracy'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    variable_summaries(accuracy)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_dir, sess.graph)

saver = tf.train.Saver()

record=[["stations_train"]]
img,lab=input.input(list(map(lambda x:x[0]+".tfrecords", record)),56,50)
lab = tf.one_hot(lab,5)
lab = tf.cast(lab, tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(1000):
        batch = sess.run([img,lab])
        if i % 100 == 0:
            run_options  = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary = sess.run(merged,
                feed_dict={x: batch[0], y_:batch[1], keep_prob: 1.0},
                options=run_options,
                run_metadata=run_metadata)
            writer.add_summary(summary, i)
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    coord.request_stop()
    coord.join(threads)
    writer.close()
    saver.save(sess, "model.ckpt")
