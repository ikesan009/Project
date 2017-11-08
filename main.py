import tensorflow as tf
import input
import model
import loss
import train
import os
import shutil

"""
実行例

ディレクトリ構造が以下の場合
~/logs/Project
  main.py

$python3 main.py

実行後
~/logs/Project/(events.out.tfeventsファイル)
  main.py

$tensorboard --logdir=logs/Project

ローカルホストのポート6006に出力される
"""

# TensorBoard情報出力ディレクトリ
log_dir = 'logs/Project'

#dirで指定されたパスが存在しない場合ディレクトリ作成
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if format and os.path.exists(log_dir):
    shutil.rmtree(log_dir)

record=[["stations_train"]]
img,lab=input.input(list(map(lambda x:x[0]+".tfrecords", record)),224,50)
mod=model.model(img,4)
losses=loss.loss(mod,lab)
global_step = tf.Variable(0, trainable=False)
tn=train.train(total_loss=losses,global_step=global_step)
print("start")
#gpuConfig = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.80))
saver = tf.train.Saver(tf.all_variables(), max_to_keep=21)#tensorboardに出力する変数の指定
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(1000):
        if i % 10 == 0:
            summary = sess.run(merged)
            writer.add_summary(summary, i)

        elif i % 100 == 99:
            print("\n\n\n"+str(i)+"\n\n\n")
            print(losses.eval())
            run_options  = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary = sess.run(merged,
                              options=run_options,
                              run_metadata=run_metadata)
            writer.add_run_metadata(run_metadata, 'step%04d' % i)
            writer.add_summary(summary, i)
        if (i + 1) == 1000:#250回に1回,もしくは指定したステップ数で実行
            print("saving")
            saver.save(sess, os.getcwd()+"/model.ckpt", global_step=i)#モデル変数の保存
        _,losses_now=sess.run([tn,losses])


    coord.request_stop()
    coord.join(threads)
    writer.close()
print("fin")
