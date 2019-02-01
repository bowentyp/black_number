import cv2
import numpy as np
from numpy.linalg import norm
import os
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
file=os.listdir('after_img')
if os.path.exists("svm1.dat"):
    model.load("svm1.dat")
else:
    chars_train = []
    chars_label = []
    for cnt in range(len(file)):
        if file[cnt][-4:]=='.tif':
            pass
        else :
            continue
        # print('file:{0}'.format(file[cnt]))
        label = file[cnt][-8]
        if file[cnt][-5]=='1' and  file[cnt][-10]=='(':
            label=file[cnt][-9]
        if  file[cnt][-5]==')':
            label = file[cnt][0]
        print(int(label))

        digit_img = cv2.imread('after_img\\'+file[cnt])
        digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
        chars_train.append(digit_img)
        chars_label.append(int(label))
    chars_train = list(map(deskew, chars_train))
    chars_train = preprocess_hog(chars_train)
    chars_label = np.array(chars_label)
    model.train(chars_train , chars_label)
    model.save("svm1.dat")

ycnt=0
filecnt=0
file=os.listdir('after_img')
for cnt in range(len(file)):
    if file[cnt][-4:]=='.tif':
        pass
    else :
        continue
    filecnt  = filecnt + 1

    # print('file:{0}'.format(file[cnt]))
    label = file[cnt][-8]
    if file[cnt][-5]=='1' and  file[cnt][-10]=='(':
        label=file[cnt][-9]
    if  file[cnt][-5]==')':
        label = file[cnt][0]
    # print(label)
    digit_img = cv2.imread('after_img\\'+file[cnt])
    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    _, digit_img = cv2.threshold(digit_img, 127, 255, cv2.THRESH_BINARY)
    digit_img = cv2.resize(digit_img, (20, 20))
    cv2.imshow('sss', digit_img)
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
    test = deskew(digit_img)
    pre1 = preprocess_hog([test])
    text2 = model.predict(pre1)
    if int(label)==text2:
        ycnt+=1
    else:
        print(text2[0])
        print('file:{0}'.format(file[cnt]))
print('rate:{0}'.format(ycnt/filecnt))