# -*- coding: UTF-8 -*-
import sys
import numpy as np
from tqdm import tqdm
from scipy import misc, ndimage
from PIL import Image
# from image_coder import encode, decode

# python pool_x.py fl1 F:\\seq2seq\\doubletrain_doubletest\\4folds_cross_validation\\testing_set_fold1_of_4cv_data.lst

fl=sys.argv[1]
print(fl)
frame_length = int(fl[2]) #半帧的像素高
num_dim = 50*frame_length*2  #图像宽50像素,200
num_frame = int(308/frame_length)-1 #图像包含几帧   图像最长308，为了计算简便，统一设为320 num_frame=320/frame_length-1

#lstpath 存储图像data路径的list文件的路径，eg：F:\\seq2seq\\doubletrain_doubletest\\4folds_cross_validation\\testing_set_fold1_of_4cv_data.lst
lstpath=sys.argv[2]


def expand(x, n=4):
    x = np.hstack([np.ones((x.shape[0], n)), x, np.ones((x.shape[0], n)),])
    x = np.vstack([np.ones((n, x.shape[1])), x, np.ones((n, x.shape[1])),])
    return x


def speckle(img):
    severity = np.clip(np.random.randn(*img.shape), 0, 1)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck = np.asarray(img_speck>0.5, dtype=img.dtype)
    return img_speck


def normal(img, w=50):
    img = misc.imresize(img, w/float(img.shape[1]), 'bicubic')>128
    if img.shape[1] != w:
        img = np.hstack((img, np.ones((img.shape[0], w-img.shape[1]), dtype='bool')))
    # m = np.nonzero(img == 0)
    # img = img[m[0].min():m[0].max()+1, m[1].min():m[1].max()+1]

    return img


def get_img(fn):
    im = misc.imread(fn)
    im = np.asarray(im > 128, dtype='float32')
    im = normal(im, 16)
    # return encode(im)
    I = np.zeros((1, 300, 16), dtype='float32')
    # print I[0,:im.shape[0],:].shape, im.shape

    I[0, :min(im.shape[0], 300), :] = im[:min(im.shape[0], 300), :]

    return I


def convert_img(fn):
    im = np.array(Image.open(fn).convert('L')).astype('float')
    # im = misc.imread(fn)
    im = np.asarray(im > 128, dtype='float32')
    # im = expand(im)
    # im = speckle(im)
    im = normal(im)
    X = np.zeros((1, num_frame, num_dim), dtype='float32')
    for i in range(num_frame):
        d = im[i*frame_length:i*frame_length+frame_length*2, :].flatten()
        X[0,num_frame-1-i,:d.shape[0]] = d*2-1
    return X



lst_plus = []
# for i in range(18):
with open(lstpath) as f:
    lst = f.readlines()
for i in lst:
    i = i.replace("\r\n", "")
    print(i)
    lst_plus.append(i)

lst = lst_plus

X = np.zeros((len(lst), num_frame, num_dim), dtype='int8')
print(X.shape)
k = 0
for j in tqdm(range(len(lst))):
   X[k, :, :] = convert_img(lst[j].strip())
   k += 1

np.save('%s_fl%d'%(lstpath[:-4],frame_length), X)