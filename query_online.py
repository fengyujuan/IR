# -*- coding: utf-8 -*-
# Author: yongyuan.name
#from extract_features import extract_feat

import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import time
import pdb
import os


ap = argparse.ArgumentParser()
ap.add_argument("-query", required = False,
	help = "Path to query images which contains images to be indexed")
ap.add_argument("-path", required = False,
	help = "Path to database which contains images to be indexed")
ap.add_argument("-index", required = True,
	help = "Path to index")
ap.add_argument("-result", required = False,
	help = "Path for output retrieved images")
ap.add_argument("-weight", required = False,
	help = "Path for weight")
args = vars(ap.parse_args())


# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File(args["index"],'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()
        
print "--------------------------------------------------"
print "               searching starts"
print "--------------------------------------------------"
    
# read and show query image
queryDir = args["query"]
imageDir = args["path"]
weight_path = args["weight"]
'''
queryDir = args["query"]
queryImg = mpimg.imread(queryDir)
plt.title("Query Image")
plt.imshow(queryImg)
plt.show()
'''
# extract query image's feature, compute simlarity score and sort
if queryDir is None:
	clock = 0
	acc = 0
	for queryDir,queryVec in zip(imgNames,feats):
		#queryDir = os.path.join(imageDir,queryDir)
		time1 = time.time()
		
		scores = np.dot(queryVec, feats.T)
		rank_ID = np.argsort(scores)[::-1]
		rank_score = scores[rank_ID]
		
		time2 = time.time()
		clock = time2 - time1 + clock
		
		maxres = 11
		imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
		label0 = queryDir.split('/')[2]
		label = ''
		#compare the searching result with the original image
		for i in label0.split('_')[:-1]:
			label = label + i + '_'
		label = label[:-1]
		for i in imlist[1:]:
			l = ''
			for j in i.split('_')[:-1]:
				l = l + j + '_'
			l = l[:-1]
			if l == label:
				acc += 1
		
	#print rank_ID
	#print rank_score
	print 'acc:'+str(float(acc)/len(imgNames)/10)
	print 'total time:'+str(clock)
else:
	queryVec = extract_feat(queryDir)
	scores = np.dot(queryVec, feats.T)
	rank_ID = np.argsort(scores)[::-1]
	rank_score = scores[rank_ID]
	# number of top retrieved images to show
	maxres = 11
	imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
	print "top %d images in order are: " %maxres, imlist



# show top #maxres retrieved result one by one
'''
for i,im in enumerate(imlist):
    image = mpimg.imread(args["result"]+"/"+im)
    plt.title("search output %d" %(i+1))
    plt.imshow(image)
    plt.show()
'''