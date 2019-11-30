import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import argparse
import os
import os.path as osp
import pickle as pkl
import pandas as pd
import random

from darknet import Darknet
from util import *

# Parameters initialization
img_folder = "./imgs"
config_file = "./yolov3.cfg"
weight_file = "./yolov3.weights"

num_classes = 80
classes = load_classes("data/coco.names")

batch_size = 1
confidence = 0.5
nms_threesh = 0.4
reso = 416
CUDA = torch.cuda.is_available()

#Set up the neural network
print("Loading network..."")
model = Darknet(config_file)

print("Loading weights...")
model.load_weights(weight_file)

model.net_info["height"] = reso
inp_dim = int(model.net_info["height"])
if CUDA:
    model.cuda()

# Read images
imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(img_folder)]
imgs = [cv2.imread(x) for x in imlist]
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
if CUDA:
    im_dim_list = im_dim_list.cuda()

leftover = 0
if (len(im_dim_list) % batch_size):
    leftover = 1
write = 0

# Processing images
for i, batch in enumerate(im_batches):
#load the image
    if CUDA:
        batch = batch.cuda()
    with torch.no_grad():
        prediction = model(Variable(batch), CUDA)

    prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)

    if type(prediction) == int:

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    prediction[:,0] += i*batch_size    #transform the atribute from index in batch to index in imlist

    if not write:                      #If we have't initialised output
        output = prediction
        write = 1
    else:
        output = torch.cat((output,prediction))

    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()
try:
    output
except NameError:
    print ("No detections were made")
    exit()

im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

scaling_factor = torch.min(416/im_dim_list,1)[0].view(-1,1)
