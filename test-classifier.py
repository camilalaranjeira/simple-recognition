# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 02:14:43 2016

@author: laranjeira
"""

import pickle
import os
import cv2
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import caffe

caffe_root = '/home/laranjeira/caffe/' 

# VGG caffe model
model = "VGG_ILSVRC_19_layers_deploy.prototxt"
weights = "VGG_ILSVRC_19_layers.caffemodel"

VGG_LAYER = "fc7"

# Dataset info
test_images_dir = '/home/laranjeira/projects/Pokemom-Recognition/test-images/'

pikachu_label = 0
bulbasaur_label = 1
squirtle_label = 2
charmander_label = 3

caffe.set_mode_cpu()
net = caffe.Net(model, weights, caffe.TEST)
mean = numpy.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mean)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

# batch size, # 3-channel (BGR) images, # image size is 224x224
net.blobs['data'].reshape(1, 3, 224, 224)  
def extractFeatures(path):
    image = caffe.io.load_image(path)
    transformed_image = transformer.preprocess('data', image) 
    
    net.blobs['data'].data[...] = transformed_image
    net.forward()
    features = net.blobs[VGG_LAYER].data.copy()
 
    return features
    
def getStrName(prediction):
    if prediction == pikachu_label:
        return "Pikachu"
    elif prediction == bulbasaur_label:
        return "Bulbasaur"
    elif prediction == squirtle_label:
        return "Squirtle"
    else:
        return "Charmander"
    
def testClassifier():
    clf = pickle.load(open( "classifier.p", "rb" ))
    
    image_files = sorted(os.listdir(os.path.join(test_images_dir)))
    for image in image_files:
       feat = extractFeatures( str( os.path.abspath(os.path.join(test_images_dir, image)) ) )
       prediction = clf.predict(feat)
       img = mpimg.imread(str( os.path.abspath(os.path.join(test_images_dir, image)) ))
       fig = plt.figure()
       fig.suptitle(getStrName(prediction), fontsize=14, fontweight='bold')
       plt.imshow(img)
    
if __name__ == "__main__":
    testClassifier()
    