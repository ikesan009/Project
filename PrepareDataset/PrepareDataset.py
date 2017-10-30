import os
import cv2
import glob

class PrepareDataset(object):

    def capture(self,video,dir,fname,fps=30):
        #dirで指定されたパスが存在しない場合ディレクトリ作成
        if not os.path.exists(dir):
            os.makedirs(dir)

        #フレーム画像をキャプチャする
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

    def convert_to_grayscale(self,dir_img,dir_save):
        #dirで指定されたパスが存在しない場合ディレクトリ作成
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)

        list_img = sorted(glob.glob(dir_img+'/*'))
        print('converting to grayscale...')
        for f_img in list_img:
            img = cv2.imread(f_img)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(os.path.join(dir_save,os.path.splitext(os.path.basename(f_img))[0]+'_gray.png'), img_gray)
        print('Done.')

    def detect_edge(self,dir_img,dir_save,minVal,maxVal,sobel=3):
        #dirで指定されたパスが存在しない場合ディレクトリ作成
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)

        list_img = sorted(glob.glob(dir_img+'/*'))
        print('detecting edge...')
        for f_img in list_img:
            img = cv2.imread(f_img)
            img_edge = cv2.Canny(img,minVal,maxVal,sobel)
            cv2.imwrite(os.path.join(dir_save,os.path.splitext(os.path.basename(f_img))[0]+'_edge.png'), img_edge)
        print('Done.')

    #一括操作
    def prepare_from_video(self,video,dir,fname,minVal,maxVal,fps=30,sobel=3):
        self.capture(video,dir,fname,fps)
        self.convert_to_grayscale(dir,dir+'_gray')
        self.detect_edge(dir,dir+'_edge',minVal,maxVal,sobel)
