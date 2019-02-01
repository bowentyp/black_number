import numpy as np
import cv2
from numpy.linalg import norm
import os
import matplotlib.pyplot as plt
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
    # if file[cnt][:-5]=='redline (10':
    #     pass
    # else :
    #     continue
    # print(file[cnt])
    img=cv2.imread(os.path.join(path,file[cnt]))
    img=cv2.resize(img,(100,100))

    imgCanny=cv2.Canny(img.copy(),250,400)
    imgCanny=cv2.morphologyEx(imgCanny,cv2.MORPH_CLOSE,kernal,iterations=2)

    cv2.imshow('imgCanny',imgCanny)

    #数字分割1
    y_his =np.sum(imgCanny,axis=0)/255
    y_his=np.concatenate([np.zeros(1),y_his,np.zeros(1)])
    # print(y_his)

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
    # print('x0:{0}'.format(x0))
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
    cv2.imshow('num11', num11)

    #######cv2.imwrite('compare\\'+file[cnt][:-4]+'_1.tif',num11)

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
    # print('num1:')
    # print('B_RATE:{0}'.format(B_RATE))
    # print('G_RATE:{0}'.format(G_RATE))
    # print('R_RATE:{0}'.format(R_RATE))




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
    cv2.imshow('num ', num11)

    ###### cv2.imwrite('compare\\'+file[cnt][:-4]+'_11.tif',num11)
    # chars_train.append(num11)
    label = file[cnt][-6]
    if  file[cnt][-8] == '(':
        label = file[cnt][-7]
    # print('label1:{0}'.format(label))
    # chars_label.append(int(label))

    test = deskew(num11)
    pre1 = preprocess_hog([test])
    text1 = model.predict(pre1)
    # print('text1:{0}'.format(int(text1[0])))
    num_predict = text1[0].astype(int)
    if label!=str(int(text1[0])):
        print('not match num1:{0}'.format(int(text1[0])))
    # cv2.imwrite('after_img\\'+file[cnt][:-4]+'_1.tif', num11)


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
            # print('x0:{0}'.format(x0))
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
            # print('xloc:{0}'.format(x1))
            # num2 = num2[x3:x4, :]
            cannynum2 = imgCanny[x3:x4, yloc1:yloc1+Weighty2].copy()
            num22 = img[x3:x4, yloc1:yloc1+Weighty2].copy()
            cv2.imshow('num2', num22)

            ##############cv2.imwrite('compare\\'+file[cnt][:-4]+'_2.tif',num22)

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
            # print('num2:')
            # print('B white rate:{0}'.format(B_RATE))
            # print('G white rate:{0}'.format(G_RATE))
            # print('R white rate:{0}'.format(R_RATE))
            # print(' ')
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
                    # print('B')
                    _, num22 = cv2.threshold(B, 180, 255, cv2.THRESH_BINARY)
                    num22=cv2.morphologyEx(num22,cv2.MORPH_OPEN,kernal,iterations=3)
                    num22=cv2.dilate(num22,kernal2)
                elif color_location == 1:
                    # print('G')
                    _, num22 = cv2.threshold(G, 180, 255, cv2.THRESH_BINARY)
                    num22=cv2.morphologyEx(num22,cv2.MORPH_OPEN,kernal,iterations=3)
                    num22=cv2.dilate(num22,kernal2)
                else:
                    # print('R')
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
            cv2.imshow('num22', num22)

            #############cv2.imwrite('compare\\'+file[cnt][:-4]+'_22.tif',num22)
            # chars_train.append(num22)
            label = file[cnt][-6]
            # print('label2:{0}'.format(label))
            # chars_label.append(int(label))

            test = deskew(num22)
            pre1 = preprocess_hog([test])
            text2 = model.predict(pre1)
            # print('text2:{0}'.format(int(text2[0])))
            if label != str(int(text2[0])):
                print('not match num2{0}'.format(int(text2[0])))
            num_predict = num_predict * 10 + text2[0].astype(int)
            # cv2.imwrite('after_img\\'+file[cnt][:-4]+'_2.tif', num22)

    # print('num_predict:{0}'.format(num_predict))
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        pass






    # B_num=np.arange(len(x_his))
    # G_num=np.arange(len(y_his))
    # FIG=plt.figure()
    # plt.subplot(121)
    # plt.bar(B_num,x_his,0.5,color='green')
    # plt.subplot(122)
    # plt.bar(G_num,y_his,0.5,color='green')
    # plt.savefig('canny\\hist'+file[cnt][:-4]+'.tif')
    # np.savetxt('tdf.txt',R,fmt='%d')

    #
    # cv2.imshow('B',B)
    # cv2.imshow('G',G)
    # cv2.imshow('R',R)
    #
    # if B_RATE>0.4 and G_RATE>0.4 and R_RATE>0.4 :
    #     print(file[cnt]+'has white region as main color')
    # #     pass
    # # else :
    # #     print('begin·')
    # #     for i in range(B.shape[0]):
    # #         for ii in range(B.shape[1]):
    # #             if B[i][ii]>200 and R[i][ii]>200 and R[i][ii]>200:
    # #                 B[i][ii] -=150
    # #                 G[i][ii] -=150
    # #                 R[i][ii] -=150
    # #             if B[i][ii]>100 and R[i][ii]>100 and R[i][ii]>100:
    # #                 B[i][ii] -=80
    # #                 G[i][ii] -=80
    # #                 R[i][ii] -=80
    #
    #
    # _, img1 = cv2.threshold(G, 180, 255, cv2.THRESH_BINARY)
    # _, img2 = cv2.threshold(R, 180, 255, cv2.THRESH_BINARY)
    # _, img3 = cv2.threshold(B, 180, 255, cv2.THRESH_BINARY)
    #
    # img1=cv2.morphologyEx(img1,cv2.MORPH_OPEN,kernal,iterations=2)
    # img2=cv2.morphologyEx(img2,cv2.MORPH_OPEN,kernal,iterations=2)
    # img3=cv2.morphologyEx(img3,cv2.MORPH_OPEN,kernal,iterations=2)
    #
    #
    # cv2.imshow('B1',img3)
    # cv2.imshow('G1',img1)
    # cv2.imshow('R2',img2)

# jh=os.listdir('after_img')
# for ij in range(len(jh)):
#     label=jh[ij][0]
#     img=cv2.imread('after_img\\'+jh[ij])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     chars_train.append(img)
#     chars_label.append(int(label))
# chars_train = list(map(deskew, chars_train))
# chars_train = preprocess_hog(chars_train)
# chars_label = np.array(chars_label)
# model.train(chars_train , chars_label)
# model.save("svm2.dat")


# print(file[cnt])
# B_num=np.arange(len(Bhist))
# G_num=np.arange(len(Ghist))
# R_num=np.arange(len(Rhist))
# FIG=plt.figure()
# plt.subplot(131)
# plt.bar(B_num,Bhist,0.5,color='green')
# plt.subplot(132)
# plt.bar(G_num,Ghist,0.5,color='green')
# plt.subplot(133)
# plt.bar(R_num,Rhist,0.5,color='green')
# plt.savefig('hist\\'+file[cnt][:-4]+'.tif')
# print(B_num)