#conding:utf-8
import os
import cv2
import glob
import shutil
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

"""
実行例

ディレクトリ構造が以下の場合
~/Stations/0西口/(mp4ファイル群)
           1東口/(mp4ファイル群)
           2東京駅/(mp4ファイル群)
           3柏駅/(mp4ファイル群)
           4池袋駅/(mp4ファイル群)
           PrepareDataset.py

$python3
>>>import PrepareDataset as PD
>>>pd = PD.PrepareDataset()
>>>pd.capture(video="Stations/2東京駅/MOV_0094.mp4",dir="Station_samples/2東京駅",fname="東京駅00",fps=1)
>>>pd.split_sample(dir="Station_samples")
>>>pd.make_samples(width=224,height=224)

現在file=format=Trueにするとバグる模様.今後修正予定

実行後
~/Station_samples/0梅郷駅西口/(pingファイル群)
                  1梅郷駅西口/(pingファイル群)
                  2東京駅/(pingファイル群)
                  3柏駅/(pingファイル群)
                  4池袋駅/(pingファイル群)
  samples/test/0梅郷駅西口/(pingファイル群)
               1梅郷駅西口/(pingファイル群)
               2東京駅/(pingファイル群)
               3柏駅/(pingファイル群)
               4池袋駅/(pingファイル群)
          train/0梅郷駅西口/(pingファイル群)
                1梅郷駅西口/(pingファイル群)
                2東京駅/(pingファイル群)
                3柏駅/(pingファイル群)
                4池袋駅/(pingファイル群)
  Stations/(省略)
  stations_test.tfrecords
  stations_train.tfrecords
  PrepareDataset.py
"""

class PrepareDataset(object):

    #dirで指定されたパスが存在しない場合ディレクトリ作成
    def make_dir(self,dir,format):
        if not os.path.exists(dir):
            os.makedirs(dir)
        if format and os.path.exists(dir):
            shutil.rmtree(dir)

    #フレーム画像をキャプチャする
    def capture(self,video,dir,fname,file_format=False,fps=30):
        self.make_dir(dir,file_format)
        cap = cv2.VideoCapture(video)
        v_fps = round(cap.get(cv2.CAP_PROP_FPS))
        interval = round(v_fps/fps)
        i = 0
        print('capturing from video...')
        while(cap.isOpened()):
            flag,frame = cap.read()
            if (flag==False):
                break
            cv2.imwrite(os.path.join(dir,fname+str(i).zfill(5)+'.png'), frame)
            i += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, interval*i)

        print('Done.')
        cap.release()

    def convert_to_grayscale(self,dir_img,dir_save,file_format=False):
        self.make_dir(dir,file_format)
        list_img = sorted(glob.glob(dir_img+'/*'))
        print('converting to grayscale...')
        for f_img in list_img:
            img = cv2.imread(f_img)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(os.path.join(dir_save,os.path.splitext(os.path.basename(f_img))[0]+'_gray.png'), img_gray)
        print('Done.')

    def detect_edge(self,dir_img,dir_save,minVal,maxVal,file_format=False,sobel=3):
        self.make_dir(dir,file_format)
        list_img = sorted(glob.glob(dir_img+'/*'))
        print('detecting edge...')
        for f_img in list_img:
            img = cv2.imread(f_img)
            img_edge = cv2.Canny(img,minVal,maxVal,sobel)
            cv2.imwrite(os.path.join(dir_save,os.path.splitext(os.path.basename(f_img))[0]+'_edge.png'), img_edge)
        print('Done.')

    #一括操作
    def prepare_from_video(self,video,dir,fname,minVal,maxVal,file_format=False,fps=30,sobel=3):
        self.capture(video,dir,fname,file_format,fps)
        self.convert_to_grayscale(dir,dir+'_gray',file_format)
        self.detect_edge(dir,dir+'_edge',minVal,maxVal,file_format,sobel)

    #トレーニング/テストデータの分割
    def split_sample(self,dir,train_size=0.8,file_format=False,output_file="./samples"):
        samples=[]
        labels=[]
        files = os.listdir(dir)             #dir直下のディレクトリリストを取得

        self.make_dir(output_file,file_format)   #出力ディレクトリの生成
        self.make_dir(output_file+"/train",file_format)
        self.make_dir(output_file+"/test",file_format)

        for i in files:
            tmp=os.listdir(dir+"/"+i)
            samples+=list(map(lambda x:dir+"/"+i+"/"+x,tmp))#画像パスリストの取得
            labels+=[i]*len(tmp)                            #ラベルリストの生成

            #トレーニング,テストデータ格納ディレクトリの生成
            self.make_dir(output_file+"/train/"+i,file_format)
            self.make_dir(output_file+"/test/"+i,file_format)

        #トレーニング,テストデータを分割
        samples_train, samples_eval, labels_train, labels_eval = train_test_split(samples, labels, train_size=train_size)

        for i in range(len(samples_train)):#トレーニングデータのコピー
            shutil.copy(samples_train[i], output_file+"/train/"+labels_train[i])

        for i in range(len(samples_eval)):#テストデータのコピー
            shutil.copy(samples_eval[i], output_file+"/test/"+labels_eval[i])

    def make_TFR(self,rec_file_name,img_data,width=0,height=0):#画像→TFRecord
        with tf.python_io.TFRecordWriter(rec_file_name) as writer:
            for img_name,label in img_data:
                if width!=0:    #サイズ変更があれば変換を行う
                    img = Image.open(img_name).convert("RGB").resize((width, height))
                else:
                    img =Image.open(img_name)
                w, h = img.size

                img_obj=np.array(img).tostring() #バイナリに変換

                record = tf.train.Example(features=tf.train.Features(feature={ #画像のパラメータ設定
                    "label": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[label])),
                    "image": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[img_obj])),
                    "height": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[h])),
                    "width": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[w])),
                    "depth": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[3])),
                }))
                writer.write(record.SerializeToString())            #書き込み

    def image_lister(self,dir):
        img_data=[]
        files = os.listdir(dir)#dir直下のディレクトリ名リストを取得
        for i in files:
            img_files=os.listdir(dir+"/"+i)#ディレクトリ内部の画像名リストを取得
            label=int(i[0])#ディレクトリ名をラベルに変換
            img_data+=list(map(lambda x:[dir+"/"+i+"/"+x,label],img_files))#リストに追加
        return img_data

    def make_samples(self,width=0,height=0):
        img_data_train = self.image_lister("samples/train")
        record_file_train = './stations_train.tfrecords'
        self.make_TFR(record_file_train,img_data_train,width,height)
        img_data_test = self.image_lister("samples/test")
        record_file_test = './stations_test.tfrecords'
        self.make_TFR(record_file_test,img_data_test,width,height)
