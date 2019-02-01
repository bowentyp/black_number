import numpy as np
import cv2
from numpy.linalg import norm
import os
import datetime

# import matplotlib.pyplot as plt
SZ=20
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)

class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 100, gamma = 0.3):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)
    #训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    #字符识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()

starttime=datetime.datetime.now()
# print('time:\n{0}'.format( starttime))
model = SVM()
try:
    model.load("svm1.dat")
except:
    print("error :no trained file!!")
    exit()

chars_train = []
chars_label = []

path='image'
file=os.listdir(path)
kernal=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
kernal2=cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
for cnt in range(len(file)):
    # if file[cnt][:3]=='pjy':
    #     pass
    # else :
    #     continue
    # print(file[cnt])
    img=cv2.imread(os.path.join(path,file[cnt]))
    img=cv2.resize(img,(100,100))

    imgCanny=cv2.Canny(img.copy(),250,400)
    imgCanny=cv2.morphologyEx(imgCanny,cv2.MORPH_CLOSE,kernal,iterations=2)

    cv2.imshow('imgCanny',img)

    #数字分割1
    y_his =np.sum(imgCanny,axis=0)/255
    y_his=np.concatenate([np.zeros(1),y_his,np.zeros(1)])

    yloc = 0
    yloc_r = 0
    Weighty = 0
    for ycnt in range(len(y_his) - 1):
        if y_his[ycnt] == 0 and y_his[ycnt + 1] != 0:
            yloc_r = ycnt
        if y_his[ycnt] != 0 and y_his[ycnt + 1] == 0 and (ycnt - yloc_r) > Weighty:
            Weighty = ycnt - yloc_r
            yloc = yloc_r
        if Weighty > 10:
            break
    Weighty = Weighty + 1
    num1 = imgCanny[:, yloc:yloc + Weighty]

    x_his=np.sum(num1,axis=1)/255
    x_his = np.concatenate([np.zeros(1), x_his])
    xloc = 0
    xloc_r = 0
    Weightx = 0
    for xcnt in range(len(x_his) - 1):
        if (x_his[xcnt] == 0 or xcnt==0 )and x_his[xcnt + 1] != 0 :
            xloc_r = xcnt
        if x_his[xcnt] != 0 and x_his[xcnt + 1] == 0 and (xcnt - xloc_r) > Weightx:
            Weightx = xcnt - xloc_r
            xloc = xloc_r
        if Weightx > 10:
            break
    x0=xloc
    x1=len(x_his)-1
    Weightx = 0
    for xcnt1 in range(len(x_his)-1 ):
        xcnt=len(x_his)-xcnt1- 1
        if (x_his[xcnt] == 0 or xcnt==len(x_his)- 1)and x_his[xcnt - 1] != 0 :
            xloc_r = xcnt
        if x_his[xcnt] != 0 and x_his[xcnt - 1] == 0 and ( xloc_r-xcnt ) > Weightx:
            Weightx =  xloc_r-xcnt
            x1 = xloc_r
        if Weightx > 10:
            break
    cannynum1=imgCanny[x0:x1,yloc:yloc + Weighty].copy()
    num11=img[x0:x1,yloc:yloc + Weighty].copy()
    # cv2.imshow('num11', num11)

    number_pix=num11.shape[0]*num11.shape[1]+1#修正除数为0的情况

    B=num11[:,:,0].copy()
    G=num11[:,:,1].copy()
    R=num11[:,:,2].copy()

    Bhist = cv2.calcHist([B],[0],None,[256],[0,256]).flatten()
    Ghist = cv2.calcHist([G],[0],None,[256],[0,256]).flatten()
    Rhist = cv2.calcHist([R],[0],None,[256],[0,256]).flatten()

    B_RATE=np.sum(Bhist[180:])   / (number_pix - np.sum(Bhist[:100]))
    G_RATE=np.sum(Ghist[180:] ) / (number_pix  - np.sum(Ghist[:100]))
    R_RATE=np.sum(Rhist[180:] ) / (number_pix  - np.sum(Rhist[:100]))

    flag=False
    if B_RATE>0.6  :
        if G_RATE>0.6 or R_RATE> 0.6:
            flag =True
    elif G_RATE>0.6 and R_RATE> 0.6:
        flag = True

    if flag==True :
        num11=cv2.cvtColor(num11,cv2.COLOR_BGR2GRAY)
        _, num11 = cv2.threshold(num11, 127, 255, cv2.THRESH_BINARY)
    else :
        color_location = np.argmax([B_RATE, G_RATE, R_RATE])
        if color_location==0:
            _,num11 =cv2.threshold(B, 180, 255, cv2.THRESH_BINARY)
            num11=cv2.morphologyEx(num11,cv2.MORPH_OPEN,kernal,iterations=3)
            num11=cv2.dilate(num11,kernal2)
        elif color_location==1:
            _,num11 =cv2.threshold(G, 180, 255, cv2.THRESH_BINARY)
            num11=cv2.morphologyEx(num11,cv2.MORPH_OPEN,kernal,iterations=3)
            num11=cv2.dilate(num11,kernal2)
        else :
            _,num11 =cv2.threshold(R, 180, 255, cv2.THRESH_BINARY)
            num11=cv2.morphologyEx(num11,cv2.MORPH_OPEN,kernal,iterations=3)
            num11=cv2.dilate(num11,kernal2)

    num11 = cv2.erode(num11, kernal)
    num11=num11+cannynum1
    num11=cv2.morphologyEx(num11,cv2.MORPH_CLOSE,kernal2,iterations=1)
    if num11.shape[0]-num11.shape[1]>0:
        weight0=int((num11.shape[0]-num11.shape[1])/2)+2
        num11 = cv2.copyMakeBorder(num11, 1, 1, weight0, weight0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    num11=cv2.resize(num11,(18,18))
    num11 = cv2.copyMakeBorder(num11, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # cv2.imshow('num ', num11)

    test = deskew(num11)
    pre1 = preprocess_hog([test])
    text1 = model.predict(pre1)
    num_predict = text1[0].astype(int)


    #数字分割2
    num2 = np.zeros(1)
    y_his[yloc:yloc + Weighty] = np.zeros(Weighty)
    num = 1
    if np.sum(y_his) > 100:
        yloc1 = 0
        yloc_r = 0
        Weighty2 = 0
        for ycnt in range(len(y_his) - 1):
            if y_his[ycnt] == 0 and y_his[ycnt + 1] != 0:
                yloc_r = ycnt
            if y_his[ycnt] != 0 and y_his[ycnt + 1] == 0 and (ycnt - yloc_r) > Weighty2:
                Weighty2 = ycnt - yloc_r
                yloc1 = yloc_r
            if Weighty2 > 10:
                num = 2
        if num == 2 :
            num2 = imgCanny[:, yloc1:]

            x_his = np.sum(num2, axis=1)
            xloc = 0
            xloc_r = 0
            Weightx = 0
            for xcnt in range(len(x_his) - 1):
                if (x_his[xcnt] == 0 or xcnt == 0) and x_his[xcnt + 1] != 0:
                    xloc_r = xcnt
                if x_his[xcnt] != 0 and x_his[xcnt + 1] == 0 and (xcnt - xloc_r) > Weightx:
                    Weightx = xcnt - xloc_r
                    xloc = xloc_r
                if Weightx > 10:
                    break
            x3 = xloc
            x4 = len(x_his) - 1
            Weightx = 0
            for xcnt1 in range(len(x_his) - 1):
                xcnt = len(x_his) - xcnt1 - 1
                if (x_his[xcnt] == 0 or xcnt == len(x_his) - 1) and x_his[xcnt - 1] != 0:
                    xloc_r = xcnt
                if x_his[xcnt] != 0 and x_his[xcnt - 1] == 0 and (xloc_r - xcnt) > Weightx:
                    Weightx = xloc_r - xcnt
                    x4 = xloc_r
                if Weightx > 10:
                    break

            cannynum2 = imgCanny[x3:x4, yloc1:yloc1+Weighty2].copy()
            num22 = img[x3:x4, yloc1:yloc1+Weighty2].copy()
            # cv2.imshow('num2', num22)


            number_pix = num22.shape[0] * num22.shape[1] + 1  # 修正除数为0的情况
            B = num22[:, :, 0].copy()
            G = num22[:, :, 1].copy()
            R = num22[:, :, 2].copy()

            Bhist = cv2.calcHist([B], [0], None, [256], [0, 256]).flatten()
            Ghist = cv2.calcHist([G], [0], None, [256], [0, 256]).flatten()
            Rhist = cv2.calcHist([R], [0], None, [256], [0, 256]).flatten()

            B_RATE = np.sum(Bhist[180:]) / (number_pix - np.sum(Bhist[:100]))
            G_RATE = np.sum(Ghist[180:]) / (number_pix - np.sum(Ghist[:100]))
            R_RATE = np.sum(Rhist[180:]) / (number_pix - np.sum(Rhist[:100]))

            flag = False
            if B_RATE > 0.6:
                if G_RATE > 0.6 or R_RATE > 0.6:
                    flag = True
            elif G_RATE > 0.6 and R_RATE > 0.6:
                flag = True

            if flag == True:
                # print('gray')
                num22 = cv2.cvtColor(num22, cv2.COLOR_BGR2GRAY)
                _, num22 = cv2.threshold(num22, 127, 255, cv2.THRESH_BINARY)
            else:
                color_location = np.argmax([B_RATE, G_RATE, R_RATE])
                if color_location == 0:
                    _, num22 = cv2.threshold(B, 180, 255, cv2.THRESH_BINARY)
                    num22=cv2.morphologyEx(num22,cv2.MORPH_OPEN,kernal,iterations=3)
                    num22=cv2.dilate(num22,kernal2)
                elif color_location == 1:
                    _, num22 = cv2.threshold(G, 180, 255, cv2.THRESH_BINARY)
                    num22=cv2.morphologyEx(num22,cv2.MORPH_OPEN,kernal,iterations=3)
                    num22=cv2.dilate(num22,kernal2)
                else:
                    _, num22 = cv2.threshold(R, 180, 255, cv2.THRESH_BINARY)
                    num22=cv2.morphologyEx(num22,cv2.MORPH_OPEN,kernal,iterations=3)
                    num22=cv2.dilate(num22,kernal2)

            num22 = cv2.erode(num22, kernal)
            num22=num22+cannynum2
            num22=cv2.morphologyEx(num22,cv2.MORPH_CLOSE,kernal2,iterations=1)
            if num22.shape[0] - num22.shape[1] > 0:
                weight0 = int((num22.shape[0] - num22.shape[1]) / 2 + 1)
                num22 = cv2.copyMakeBorder(num22, 0, 0, weight0, weight0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            num22 = cv2.resize(num22, (18, 18))
            num22 = cv2.copyMakeBorder(num22, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            # cv2.imshow('num22', num22)


            test = deskew(num22)
            pre1 = preprocess_hog([test])
            text2 = model.predict(pre1)
            # print('text2:{0}'.format(int(text2[0])))

            num_predict = num_predict * 10 + text2[0].astype(int)
    # print('cnt:{0}'.format(cnt))
    print('num_predict:{0}'.format(num_predict))
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
    #     pass
endtime=datetime.datetime.now()
print('time:\n{0}.{1}'.format(( endtime-starttime).seconds,( endtime-starttime).microseconds ))




