# -+- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os,sys,caffe
import scipy.io as sio 
import glob
from collections import OrderedDict
from copy import deepcopy
from sklearn import preprocessing
from tqdm import tqdm

IMAGE_ROOT = '/media/labseashell/软件/datas/flowers/'
TEST_TXT = open(IMAGE_ROOT + 'all_test_good.txt', 'r')
FILE_LINES = TEST_TXT.readlines()
MEAN_FILE = "/media/labseashell/软件/datas/all_mean_good_256.npy"

CAFFE_MODEL= "pruned_mobilenet_v2.caffemodel"
NET_FILE = "pruned_mobilenet_v2.prototxt"

if not os.path.isfile(CAFFE_MODEL): 
    print("caffemodel is not exist...")
# caffe.set_mode_gpu()
net = caffe.Net(NET_FILE, CAFFE_MODEL, caffe.TEST)

# print net.blobs['data'].data.shape

batch_size = 1
transformer = caffe.io.Transformer({'data': (batch_size,3,224,224)})
transformer.set_transpose('data', (2,0,1))
mean_matrix = np.load(MEAN_FILE)
mean_value = np.mean(mean_matrix, axis=(1,2))
# print(mean_value)
transformer.set_mean('data', mean_value)
transformer.set_raw_scale('data', 255) 
transformer.set_channel_swap('data', (2,1,0))
net.blobs['data'].reshape(batch_size,3,224,224)

top1_num = 0 
top5_num = 0
total_num = 0
labels = []
for i, FILE_LINE in enumerate(tqdm(FILE_LINES)):
    if i > 1000:
        break
    FILE_LINE = FILE_LINE.replace('\n', '')
    IMAGE_NAME = FILE_LINE.split(' ')[0]
    label = int(FILE_LINE.split(' ')[1])
    labels.append(label)
    try:
        im = caffe.io.load_image(IMAGE_ROOT + IMAGE_NAME)
        im_input_temp = transformer.preprocess('data', im)
        net.blobs['data'].data[i%batch_size] = im_input_temp
        if i % batch_size == batch_size - 1:
            net.forward()
            for index in range(batch_size):
                all_prob = np.squeeze(deepcopy(net.blobs['prob'].data[index]))
                max_prob_indices = np.argsort(-all_prob)[:5]
                # print(label, max_prob_indices[-1])
                if labels[index] == max_prob_indices[0]:
                    top1_num += 1
                if labels[index] in max_prob_indices:
                    top5_num += 1
                total_num += 1
            labels = []
    except ValueError:
        print('error occurred: %s' % IMAGE_NAME)
top1 = top1_num * 1.0 / total_num
top5 = top5_num * 1.0 / total_num
print("top1: %.4f\t top5: %.4f" % (top1, top5))


