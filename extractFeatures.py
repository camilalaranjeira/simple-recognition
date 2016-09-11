# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 09:31:54 2016

@author: Camila Laranjeira
"""

import numpy
import caffe
import matplotlib.pyplot as plt
import os

caffe_root = '/home/laranjeira/caffe/' 

# VGG caffe model
model = "VGG_ILSVRC_19_layers_deploy.prototxt"
weights = "VGG_ILSVRC_19_layers.caffemodel"

VGG_LAYER = "fc7"

# Dataset info
dataset_root = '/home/laranjeira/projects/Pokemom-Recognition/dataset/'
pikachu_dir = 'pikachu/'
bulbasaur_dir = 'bulbasaur/'
squirtle_dir = 'squirtle/' 
charmander_dir = 'charmander/'

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
def extractFeatures(path, pokemon):
    image = caffe.io.load_image(path)
    transformed_image = transformer.preprocess('data', image) 
    
    net.blobs['data'].data[...] = transformed_image
    net.forward()
    features = net.blobs[VGG_LAYER].data.copy()
 
    filename = "features_" + pokemon[:-1] + ".txt"
    features_output = open(filename, 'a')
    for i in xrange(4096):    
        features_output.write(str(features[0][i]) + ' ')
    features_output.write('\n')
    features_output.close()
    
if __name__ == "__main__":          
    image_files = sorted(os.listdir(os.path.join(dataset_root, pikachu_dir)))
    for image in image_files:
       extractFeatures( str( os.path.abspath(os.path.join(dataset_root, pikachu_dir, image)) ), pikachu_dir )
       
    image_files = sorted(os.listdir(os.path.join(dataset_root, bulbasaur_dir)))
    for image in image_files:
       extractFeatures( str( os.path.abspath(os.path.join(dataset_root, bulbasaur_dir, image)) ), bulbasaur_dir )
       
    image_files = sorted(os.listdir(os.path.join(dataset_root, squirtle_dir)))
    for image in image_files:
       extractFeatures( str( os.path.abspath(os.path.join(dataset_root, squirtle_dir, image)) ), squirtle_dir )
       
    image_files = sorted(os.listdir(os.path.join(dataset_root, charmander_dir)))
    for image in image_files:
       extractFeatures( str( os.path.abspath(os.path.join(dataset_root, charmander_dir, image)) ), charmander_dir )