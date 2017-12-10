import tensorflow as tf
import os
import shutil
import numpy as np
import math
from PIL import Image

"""
実行例

ディレクトリ構造が以下の場合
~/Station_samples/0梅郷駅西口/(省略)
                  1梅郷駅東口/(省略)
                  2東京駅/(省略)
                  3柏駅/(省略)
                  4池袋駅/(省略)
                  5運河駅西口/(省略)
                  6運河駅東口/(省略)
  stations_train.tfrecords
  stations_test.tfrecords
  main.py

$python3
>>>import main
>>>m = main.main()
>>>m.train(n_class=7,size_image=56,model='cnn1')

実行後
~/logs/Project/train/(tfeventsファイル)
  save_files/(ckptファイル群)
  Station_samples/(省略)
  stations_train.tfrecords
  stations_test.tfrecords
  main.py

>>>m.test(n_class=7,size_image=56,model='cnn1')

実行後
~/logs/Project/train/(tfeventsファイル)
               test/(tfeventsファイル)
  save_files/(ckptファイル群)
  Station_samples/(省略)
  stations_train.tfrecords
  stations_test.tfrecords
  main.py
"""

class main(object):

    #tfrecordsファイルから画像データと対応するラベルを取得する
    def input(self,rec_name,IMAGE_SIZE,channel,BATCH_SIZE):
        file_name_queue = tf.train.string_input_producer(rec_name)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_name_queue)
        # デシリアライズ
        features = tf.parse_single_example(
            serialized_example,
            features={
                "label": tf.FixedLenFeature([], tf.int64),
                "image": tf.FixedLenFeature([], tf.string),
            })
        img_path, labels = tf.train.shuffle_batch(
            [features['image'], tf.cast(features['label'], tf.int32)],
            batch_size=BATCH_SIZE,capacity=234730+channel*BATCH_SIZE,min_after_dequeue=234730
        )
        def image_from_path(path):
            png_bytes = tf.read_file(path)
            image = tf.image.decode_png(png_bytes, channels=channel)
            return image
        images = tf.map_fn(image_from_path,img_path,dtype='uint8')
        #images = tf.map_fn(tf.image.per_image_standardization,images,dtype='float32')

        return images,labels

    #dirで指定されたパスが存在しない場合ディレクトリ作成
    def make_dir(self,dir,format=False):
        if not os.path.exists(dir):
            os.makedirs(dir)
        if format and os.path.exists(dir):
            shutil.rmtree(dir)

    #tensorboardのサマリに追加する
    def variable_summaries(self,var):

        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    #重みベクトルを初期化して返す
    def _variable_with_weight_decay(self,name,shape,stddev,wd):

        var = tf.get_variable(name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    #畳込み層
    def l_conv(self,element,size,chanel,kinds,l_name):

        with tf.variable_scope(l_name) as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[size, size, chanel, kinds], stddev=0.01, wd=0.0)
            conv = tf.nn.conv2d(element, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[kinds], initializer=tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv = tf.nn.relu(bias, name=scope.name)
            self.variable_summaries(conv)
            return conv

    #全結合層
    def l_full(self,element,con_from,con_to,l_name):

        with tf.variable_scope(l_name) as scope:
            weights = self._variable_with_weight_decay('weights', shape=[con_from, con_to], stddev=0.01, wd=0.005)
            biases = tf.get_variable('biases', shape=[con_to], initializer=tf.constant_initializer(0.0))
            fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(element, weights), biases), name=scope.name)
            self.variable_summaries(fc)
            return fc

    #CNNモデルの定義
    def cnn1(self,images,channel,n_class,keep_prob):
        #畳込み、プーリング1
        conv1 = self.l_conv(images,5,channel,32,"conv1")
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        #畳込み、プーリング2
        conv2 = self.l_conv(pool1,5,32,64,"conv2")
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        shape = pool2.get_shape().as_list()
        dim = (shape[1]**2)*shape[3]
        reshape = tf.reshape(pool2,[-1,dim])
        #全結合層1
        fc1 = self.l_full(reshape,dim,1024,"fc1")
        #ドロップアウト層
        fc1_drop = tf.nn.dropout(fc1, keep_prob)
        #全結合層2
        with tf.variable_scope("fc2") as scope:
            weights = tf.get_variable("weight", shape=[1024, n_class], initializer=tf.truncated_normal_initializer(stddev=0.01))
            biases = tf.get_variable("biases", shape=[n_class], initializer=tf.constant_initializer(0.0))
            fc2 = tf.nn.bias_add(tf.matmul(fc1_drop, weights), biases, name=scope.name)
            self.variable_summaries(fc2)
        return fc2

    #CNNモデルの定義
    def cnn2(self,images,channel,n_class,keep_prob):
        #畳込み、プーリング1
        conv1 = self.l_conv(images,5,channel,32,"conv1")
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        #畳込み、プーリング2
        conv2 = self.l_conv(pool1,5,32,64,"conv2")
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        #畳込み、プーリング3
        conv3 = self.l_conv(pool2,5,64,128,"conv3")
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        shape = pool3.get_shape().as_list()
        dim = (shape[1]**2)*shape[3]
        reshape = tf.reshape(pool3,[-1,dim])
        #全結合層1
        fc1 = self.l_full(reshape,dim,1024,"fc1")
        #ドロップアウト層
        fc1_drop = tf.nn.dropout(fc1, keep_prob)
        #全結合層2
        with tf.variable_scope("fc2") as scope:
            weights = tf.get_variable("weight", shape=[1024, n_class], initializer=tf.truncated_normal_initializer(stddev=0.01))
            biases = tf.get_variable("biases", shape=[n_class], initializer=tf.constant_initializer(0.0))
            fc2 = tf.nn.bias_add(tf.matmul(fc1_drop, weights), biases, name=scope.name)
            self.variable_summaries(fc2)
        return fc2

    #CNNモデルの定義
    def cnn3(self,images,channel,n_class,keep_prob):
        #畳込み、プーリング1
        conv1 = self.l_conv(images,3,channel,16,"conv1")
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        #畳込み、プーリング2
        conv2 = self.l_conv(pool1,3,16,32,"conv2")
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        #畳込み、プーリング3
        conv3 = self.l_conv(pool2,3,32,64,"conv3")
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        shape = pool3.get_shape().as_list()
        dim = (shape[1]**2)*shape[3]
        reshape = tf.reshape(pool3,[-1,dim])
        #全結合層1
        fc1 = self.l_full(reshape,dim,1024,"fc1")
        #ドロップアウト層
        fc1_drop = tf.nn.dropout(fc1, keep_prob)
        #全結合層2
        with tf.variable_scope("fc2") as scope:
            weights = tf.get_variable("weight", shape=[1024, n_class], initializer=tf.truncated_normal_initializer(stddev=0.01))
            biases = tf.get_variable("biases", shape=[n_class], initializer=tf.constant_initializer(0.0))
            fc2 = tf.nn.bias_add(tf.matmul(fc1_drop, weights), biases, name=scope.name)
            self.variable_summaries(fc2)
        return fc2

    #VGGモデルの定義
    def vgg1(self,images,channel,n_class,keep_prob):
        #畳込み1_1、畳込み1_2、プーリング1
        conv1_1 = self.l_conv(images,3,channel,64,"conv1_1")
        conv1_2 = self.l_conv(conv1_1,3,64,64,"conv1_2")
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        #畳込み2_1、畳込み2_2、プーリング2
        conv2_1 = self.l_conv(pool1,3,64,128,"conv2_1")
        conv2_2 = self.l_conv(conv2_1,3,128,128,"conv2_2")
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        #畳込み3_1、畳込み3_2、畳込み3_3、プーリング3
        conv3_1 = self.l_conv(pool2,3,128,256,"conv3_1")
        conv3_2 = self.l_conv(conv3_1,3,256,256,"conv3_2")
        conv3_3 = self.l_conv(conv3_2,3,256,256,"conv3_3")
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
        #畳込み4_1、畳込み4_2、畳込み4_3、プーリング4
        conv4_1 = self.l_conv(pool3,3,256,512,"conv4_1")
        conv4_2 = self.l_conv(conv4_1,3,512,512,"conv4_2")
        conv4_3 = self.l_conv(conv4_2,3,512,512,"conv4_3")
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        shape = pool4.get_shape().as_list()
        dim = (shape[1]**2)*shape[3]
        reshape = tf.reshape(pool4,[-1,dim])
        #全結合層1
        fc1 = self.l_full(reshape,dim,2048,"fc1")
        #全結合層2
        fc2 = self.l_full(fc1,2048,1024,"fc2")
        #全結合層3
        fc3 = self.l_full(fc2,1024,256,"fc3")
        #ドロップアウト層
        fc3_drop = tf.nn.dropout(fc3, keep_prob)
        #全結合層4
        with tf.variable_scope("fc4") as scope:
            weights = tf.get_variable("weight", shape=[256, n_class], initializer=tf.truncated_normal_initializer(stddev=0.01))
            biases = tf.get_variable("biases", shape=[n_class], initializer=tf.constant_initializer(0.0))
            fc4 = tf.nn.bias_add(tf.matmul(fc3_drop, weights), biases, name=scope.name)
            self.variable_summaries(fc4)
        return fc4

    #モデルの呼び出し
    def model(self,images,channel,n_class,keep_prob,model):
        if model == 'cnn1':
            return self.cnn1(images,channel,n_class,keep_prob)
        elif model == 'cnn2':
            return self.cnn2(images,channel,n_class,keep_prob)
        elif model == 'cnn3':
            return self.cnn3(images,channel,n_class,keep_prob)
        elif model == 'vgg1':
            return self.vgg1(images,channel,n_class,keep_prob)

    #トレーニングを行う
    def train(self,n_class,size_image,channel,model,log_dir='logs/Project/train'):

        sess = tf.InteractiveSession()

        self.make_dir(log_dir,False)
        #画像とラベルとドロップアウト層のパラメータのプレイスホルダを生成
        x = tf.placeholder(tf.float32, shape=[None,size_image,size_image,channel])
        y_ = tf.placeholder(tf.float32, shape=[None,n_class])
        keep_prob = tf.placeholder(tf.float32)
        #画像をモデルにかける
        y_conv = self.model(x,channel,n_class,keep_prob,model)
        #損失関数よりlossを取得
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
            tf.add_to_collection('losses', cross_entropy)
            error=tf.add_n(tf.get_collection('losses'), name='total_loss')
            self.variable_summaries(error)
        #確率的勾配降下法により重みを最適化
        with tf.name_scope('accuracy'):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(error)
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.variable_summaries(accuracy)
        #サマリをマージする
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        saver = tf.train.Saver(max_to_keep=500)
        #画像データと対応するラベルを取得
        record=[["stations_train"]]
        img,lab = self.input(list(map(lambda x:x[0]+".tfrecords", record)),size_image,channel,50)
        lab = tf.one_hot(lab,n_class)
        lab = tf.cast(lab, tf.float32)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #スレッドを利用して並列処理
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for i in range(10000):
                #バッチサイズ分のデータを格納
                batch = sess.run([img,lab])
                if (i+1) % 100 == 0:
                    #100ステップ毎のパラメータをtensorboardに書き出す
                    run_options  = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary = sess.run(merged,
                        feed_dict={x: batch[0], y_:batch[1], keep_prob: 1.0},
                        options=run_options,
                        run_metadata=run_metadata)
                    writer.add_summary(summary, i)
                    #100ステップ毎の正答率をprintし、パラメータ情報を保存する
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch[0], y_: batch[1], keep_prob: 1.0})
                    print('step %d, training accuracy %g' % ((i+1), train_accuracy))
                    saver.save(sess, "save_files/model.ckpt", global_step=(i+1))
                #トレーニング
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            coord.request_stop()
            coord.join(threads)
            writer.close()

    #テストを行う
    def test(self,n_class,size_image,channel,model,log_dir='logs/Project/test'):

        sess = tf.InteractiveSession()
        #画像とラベルとドロップアウト層のパラメータのプレイスホルダを生成
        x = tf.placeholder(tf.float32, shape=[None,size_image,size_image,channel])
        y_ = tf.placeholder(tf.float32, shape=[None,n_class])
        keep_prob = tf.placeholder(tf.float32)
        #画像をモデルにかける
        y_conv = self.model(x,channel,n_class,keep_prob,model)
        #正答率の計算
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.variable_summaries(accuracy)
        #サマリをマージする
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        saver = tf.train.Saver()
        #画像データと対応するラベルを取得
        record=[["stations_test"]]
        img,lab=self.input(list(map(lambda x:x[0]+".tfrecords", record)),size_image,channel,50)
        lab = tf.one_hot(lab,n_class)
        lab = tf.cast(lab, tf.float32)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #スレッドを利用して並列処理
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for i in range(100):
                #ckptファイルからパラメータ情報を復元
                saver.restore(sess, "save_files/model.ckpt-"+str((i+1)*100))
                #バッチサイズ分のデータを格納
                batch = sess.run([img,lab])
                #100ステップ毎のパラメータをtensorboardに書き出す
                run_options  = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary = sess.run(merged,
                    feed_dict={x: batch[0], y_:batch[1], keep_prob: 1.0},
                    options=run_options,
                    run_metadata=run_metadata)
                writer.add_summary(summary, i)
                #100ステップ毎の正答率をprintする
                print('step %d, test accuracy %g' % ((i+1)*100, accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})))
            coord.request_stop()
            coord.join(threads)

    def pull_num(self,str):
        i=1
        for i in range(len(str)):
            try:
                int(str[i])
            except ValueError:
                return(int(str[0:i]))
            i+=1
        return int(str)

    def identification(self,n_class,size_image,img,channel,model,epoch):

        if not os.path.exists("save_files"):
            print('Please train')
        else:
            sess = tf.InteractiveSession()
            if channel == 3:
                image = [np.array(Image.open(img).convert("RGB").resize((size_image, size_image)))]
            elif channel == 1:
                image = [np.array(Image.open(img).convert("1").resize((size_image, size_image))).reshape(size_image,size_image,1)]
            x=tf.placeholder(tf.float32,shape=[None,size_image,size_image,channel])
            keep_prob=tf.placeholder(tf.float32)
            y_conv=self.model(x,channel,n_class,keep_prob,model)

            y_=tf.nn.softmax(y_conv)

            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                saver.restore(sess,"save_files/model.ckpt-"+str(epoch))
                result = np.round(sess.run(y_,feed_dict={x: image,keep_prob: 1.0}),3)
                stationnum = sess.run(tf.argmax(result,1))
                print('station {0} ,\n station number is {1}'.format(result,stationnum))

    def stepidentification(self,n_class,size_image,img,channel,model):

        if not os.path.exists("save_files"):
            print('Please train')
        else:
            sess = tf.InteractiveSession()
            if channel == 3:
                image = [np.array(Image.open(img).convert("RGB").resize((size_image, size_image)))]
            elif channel == 1:
                image = [np.array(Image.open(img).convert("1").resize((size_image, size_image))).reshape(size_image,size_image,1)]
            x=tf.placeholder(tf.float32,shape=[None,size_image,size_image,channel])
            keep_prob=tf.placeholder(tf.float32)
            y_conv=self.model(x,channel,n_class,keep_prob,model)

            y_=tf.nn.softmax(y_conv)

            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                for i in range(200):
                    saver.restore(sess,"save_files/model.ckpt-"+str((i+1)*100))
                    result = np.round(sess.run(y_,feed_dict={x: image,keep_prob: 1.0}),3)
                    stationnum = sess.run(tf.argmax(result,1))
                    print('step {0} {1} ,\n station number is {2}'.format(((i+1)*100),result,stationnum))

    def listidentification(self,n_class,size_image,dir,channel,model,epoch):

        if not os.path.exists("save_files"):
            print('Please train')
        else:
            if not os.path.exists(dir):
                print('No directry')
            else:
                imglist=os.listdir(dir)
                x=tf.placeholder(tf.float32,shape=[None,size_image,size_image,channel])
                keep_prob=tf.placeholder(tf.float32)
                y_conv=self.model(x,channel,n_class,keep_prob,model)

                y_=tf.nn.softmax(y_conv)

                saver = tf.train.Saver()
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    saver.restore(sess,"save_files/model.ckpt-"+str(epoch))

                    if channel == 3:
                        for img in imglist:
                            image = [np.array(Image.open(dir+"/"+img).convert("RGB").resize((size_image, size_image)))]
                            result = np.round(sess.run(y_,feed_dict={x: image,keep_prob: 1.0}),3)
                            stationnum = sess.run(tf.argmax(result,1))
                            print('station {0} ,\n station number is {1}'.format(result,stationnum))
                    elif channel == 1:
                        for img in imglist:
                            image = [np.array(Image.open(dir+"/"+img).convert("1").resize((size_image, size_image))).reshape(size_image,size_image,1)]
                            result = np.round(sess.run(y_,feed_dict={x: image,keep_prob: 1.0}),3)
                            stationnum = sess.run(tf.argmax(result,1))
                            print('station {0} ,\n station number is {1}'.format(result,stationnum))

    def liststepidentification(self,n_class,size_image,model,dir,channel,labels,epoch):

        if not os.path.exists("save_files"):
            print('Please train')
        else:
            sess = tf.InteractiveSession()

            dirlist = os.listdir(dir)
            denominbator = 0
            childlist = []
            for i in dirlist:
                samplelist=os.listdir(dir+'/'+i)
                childlist.append(list(map(lambda x:i+'/'+x ,samplelist)))
                denominbator=denominbator+len(samplelist)
            testlabel = [[[j]*len(childlist[i])for j in labels[i]] for i in range(len(dirlist))]
            print(testlabel)

            x = tf.placeholder(tf.float32,shape=[None,size_image,size_image,channel])
            keep_prob = tf.placeholder(tf.float32)
            y_conv = self.model(x,channel,n_class,keep_prob,model)

            y_ = tf.nn.softmax(y_conv)

            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                for i in range(epoch//100):
                    saver.restore(sess,"save_files/model.ckpt-"+str((i+1)*100))
                    print('step'+str((i+1)*100))

                    correctans=0

                    if channel == 3:
                        for sdir,slabel,name in zip(childlist,testlabel,dirlist):
                            colorlist=[np.array(Image.open(dir+"/"+img).convert("RGB").resize((size_image, size_image))) for img in sdir]

                            result = np.round(sess.run(y_,feed_dict={x: colorlist,keep_prob: 1.0}),3)
                            stationnum = sess.run(tf.argmax(result,1))

                            childanses=(tf.equal(stationnum,slabel))

                            childanses=list(sess.run((tf.cast(childanses,tf.float32))))

                            childans=(tf.add_n(childanses))
                            correctans=correctans+tf.reduce_sum(childans)

                            print(' {0} accurancy: {1}'.format(name,sess.run(
                            tf.reduce_mean(childans))))
                    elif channel == 1:
                        for sdir,slabel,name in zip(childlist,testlabel,dirlist):
                            colorlist=[np.array(Image.open(dir+"/"+img).convert("1").resize((size_image, size_image))).reshape(size_image,size_image,1) for img in sdir]

                            result = np.round(sess.run(y_,feed_dict={x: colorlist,keep_prob: 1.0}),3)
                            stationnum = sess.run(tf.argmax(result,1))

                            childanses=(tf.equal(stationnum,slabel))

                            childanses=list(sess.run((tf.cast(childanses,tf.float32))))

                            childans=(tf.add_n(childanses))
                            correctans=correctans+tf.reduce_sum(childans)

                            print(' {0} accurancy: {1}'.format(name,sess.run(
                            tf.reduce_mean(childans))))

                    print(' All accurancy: {0}'.format(sess.run(correctans/denominbator)))

    def test_listidentification(self,n_class,correct_label,size_image,dir,channel,model,epoch,swich=False,mk_dir=False):
        incorrect=0
        l_num=""
        fst_dir=dir
        print(dir)

        if not os.path.exists("save_files"):
            print('Please train')
        else:
            if not os.path.exists(dir):
                print('No directry')
            else:
                if swich:
                    l_num_list=[]

                    for label_num in correct_label:
                        maped_num = map(str, label_num)
                        label_number="".join(maped_num)
                        l_num_list.append(label_number)

                    if mk_dir:
                        #print(l_num_list)
                        for l_num_swich in l_num_list:
                            self.make_dir("./incorrect_pics/incorrect_"+l_num_swich,format=False)
                            print("./incorrect_pics/incorrect_"+l_num_swich)

                    filelist=os.listdir(dir)
                    img_nums_mul=0
                    #imglistに各フォルダの画像をまとめていく
                    imglist=[]
                    #各フォルダの画像数の和([フォルダAの画像数],[フォルダAの画像数+フォルダＢの画像数],...)
                    Fileimg_count=[]
                    for filename in filelist:
                        #listFile_img初期化
                        listFile_img=[]
                        #各ファイルの画像をリストに追加
                        listFile_img=os.listdir(dir+'/'+filename)
                        imglist.extend(listFile_img)
                        img_nums_mul=len(imglist)
                        Fileimg_count.append(img_nums_mul)
                    print(Fileimg_count)

                else:

                    for label_num in correct_label:
                        l_num+=str(label_num)

                    if mk_dir:
                        self.make_dir("./incorrect_pics/incorrect_"+l_num,format=False)

                    imglist=os.listdir(dir)

                img_nums=len(imglist)

                x = tf.placeholder(tf.float32,shape=[None,size_image,size_image,channel])
                keep_prob=tf.placeholder(tf.float32)
                y_conv=self.model(x,channel,n_class,keep_prob,model)

                y_=tf.nn.softmax(y_conv)

                saver = tf.train.Saver()
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    saver.restore(sess,"save_files/model.ckpt-"+str(epoch))
                    img_count=0
                    count_check=0
                    if swich:
                        count_check=Fileimg_count.pop(0)+1
                        correct_label_=correct_label.pop(0)
                        label_dir=filelist.pop(0)
                        l_num=l_num_list.pop(0)
                        dir=fst_dir+"/"+label_dir

                    else:
                        correct_label_=correct_label

                    if channel == 3:
                        for img in imglist:
                            #img:画像一枚に対応するパス
                            img_count=img_count+1
                            if img_count==count_check:
                                if img_count != img_nums:
                                    count_check=Fileimg_count.pop(0)+1
                                    correct_label_=correct_label.pop(0)
                                    label_dir=filelist.pop(0)
                                    l_num=l_num_list.pop(0)
                                    dir=fst_dir+"/"+label_dir

                            image = [np.array(Image.open(dir+"/"+img).convert("RGB").resize((size_image, size_image)))]

                            result = np.round(sess.run(y_,feed_dict={x: image,keep_prob: 1.0}),3)
                            stationnum = sess.run(tf.argmax(result,1))

                            check=self.check_number(correct_label_,stationnum)
                            if check==0:
                                incorrect=incorrect+1
                                if mk_dir:
                                    shutil.copyfile("./"+dir+"/"+img, "./incorrect_pics/incorrect_"+l_num+"/"+img)

                                print(img+'{0}'.format(stationnum))
                    elif channel == 1:
                        for img in imglist:
                            #img:画像一枚に対応するパス
                            img_count=img_count+1
                            if img_count==count_check:
                                if img_count != img_nums:
                                    count_check=Fileimg_count.pop(0)+1
                                    correct_label_=correct_label.pop(0)
                                    label_dir=filelist.pop(0)
                                    l_num=l_num_list.pop(0)
                                    dir=fst_dir+"/"+label_dir

                            image = [np.array(Image.open(dir+"/"+img).convert("1").resize((size_image, size_image))).reshape(size_image,size_image,1)]

                            result = np.round(sess.run(y_,feed_dict={x: image,keep_prob: 1.0}),3)
                            stationnum = sess.run(tf.argmax(result,1))

                            check=self.check_number(correct_label_,stationnum)
                            if check==0:
                                incorrect=incorrect+1
                                if mk_dir:
                                    shutil.copyfile("./"+dir+"/"+img, "./incorrect_pics/incorrect_"+l_num+"/"+img)

                                print(img+'{0}'.format(stationnum))

                    percent=((img_nums-incorrect)/img_nums)
                    correctness=self.my_round(percent,2)*100
                    print('correctness is {0}% /{1}pics'.format(correctness,img_nums))

        """

            【multi_test_listidentification関数への補足】

                *anser_label_list: [[0],[1,2],..]のように、画像を含むディレクトリ順で、
                                    それに対応するラベルを要素とするリストのリスト

                *dir: 処理を施したい複数のディレクトリを含む上位のディレクトリ
                        例: dir=A、東京と運河の画像ファイルに関して処理を施すとき
                            A/  01東京駅/(東京駅の画像ファイル群)
                                02運河駅/(運河駅の画像ファイル群)

                *mk_dir: 誤って認識した画像ファイルを別のディレクトリにコピーするか
                        True : コピー
                        False: コピーしない

        """
        #複数のディレクトリに関して処理を行う
    def multi_test_listidentification(self,anser_label_list,n_class,size_image,dir,channel,model,mk_dir=False):
        if mk_dir:
            self.make_dir("./incorrect_pics",format=False)
        self.test_listidentification(n_class,anser_label_list,size_image,dir,channel,model,True,mk_dir)

    def check_number(self,correct_label,label):
        check=0

        for i in correct_label:
            if i==label:
                check=1
                break

        return check

    def my_round(self,x, d=0):
        p = 10 ** d
        return float(math.floor((x * p) + math.copysign(0.5, x)))/p
