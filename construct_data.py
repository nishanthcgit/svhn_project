# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:31:42 2020

@author: Nishanth
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 13:28:27 2018
Credit - @author: PavitrakumarPC
"""

import numpy as np
import cv2
import os
import pandas as pd
import h5py
import tensorflow as tf
import hashlib as hashlib
from PIL import Image
import logging
######################################
train_folder = r"C:\Users\Nishanth\Documents\Nishanth\train"
test_folder = r"C:\Users\Nishanth\Documents\Nishanth\test"
extra_folder = r"C:\Users\Nishanth\Documents\Nishanth\extra"
resize_size = (64,64)

def collapse_col(row):
    global resize_size
    new_row = {}
    new_row['img_name'] = list(row['img_name'])[0]
    new_row['labels'] = row['label'].astype(np.str).str.cat(sep='_')
    new_row['top'] = max(int(row['top'].min()),0)
    new_row['left'] = max(int(row['left'].min()),0)
    new_row['bottom'] = int(row['bottom'].max())
    new_row['right'] = int(row['right'].max())
    new_row['width'] = int(new_row['right'] - new_row['left'])
    new_row['height'] = int(new_row['bottom'] - new_row['top'])
    new_row['num_digits'] = len(row['label'].values)
    return pd.Series(new_row,index=None)

def image_data_constuctor(img_folder, img_bbox_data):
    print('image data construction starting...')
    imgs = []
    for img_file in os.listdir(img_folder):
        if img_file.endswith('.png'):
            imgs.append([img_file,cv2.imread(os.path.join(img_folder,img_file))])
    img_data = pd.DataFrame([],columns=['img_name','img_height','img_width','img','cut_img'])
    print('finished loading images...starting image processing...')
    for img_info in imgs:
        row = img_bbox_data[img_bbox_data['img_name']==img_info[0]]
        full_img = img_info[1] #cv2.normalize(cv2.cvtColor(cv2.resize(img_info[1],resize_size), cv2.COLOR_BGR2GRAY).astype(np.float64), 0, 1, cv2.NORM_MINMAX)
        cut_img = full_img.copy()[int(row['top']):int(row['top']+row['height']),int(row['left']):int(row['left']+row['width']),...]
        row_dict = {'img_name':[img_info[0]],'img_height':[img_info[1].shape[0]],'img_width':[img_info[1].shape[1]],'img':[full_img],'cut_img':[cut_img]}
        img_data = pd.concat([img_data,pd.DataFrame.from_dict(row_dict,orient = 'columns')])
    print('finished image processing...')
    return img_data

def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]][()]])#[()]

def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [list(attr)[0][0]]
        attrs[key] = values
    return attrs

def img_boundingbox_data_constructor(mat_file):
    f = h5py.File(mat_file,'r') 
    all_rows = []
    print('image bounding box data construction starting...')
    bbox_df = pd.DataFrame([],columns=['height','img_name','label','left','top','width'])
    for j in range(f['/digitStruct/bbox'].shape[0]):
        if j%100 == 0:
            print('number %d of %d' %(j,f['/digitStruct/bbox'].shape[0]))
        img_name = get_name(j, f)
        row_dict = get_bbox(j, f)
        row_dict['img_name'] = img_name
        all_rows.append(row_dict)
        bbox_df = pd.concat([bbox_df,pd.DataFrame.from_dict(row_dict,orient = 'columns')],sort = True)
    bbox_df['bottom'] = bbox_df['top']+bbox_df['height']
    bbox_df['right'] = bbox_df['left']+bbox_df['width']
    print('finished image bounding box data construction...')
    return bbox_df
####################################
        

def create_tf_example(data_dict, filename,baseFolder):
    # Create tf example for serialization into a tfrecord - yolov3-tf2's input format.
    #data_dict is a dictionary with all reqd fields - xmins,xmaxs,ymins,ymaxs,labels,
    #Assumes - xmins, ymins, etc are not normalized - normalizes them here 
    #Has to be cleaned up further, for some reason there are bboxes outside the 
    #image area for some images - have to look into it, bboxes seem fine for the most part
    
    img_path = os.path.join(
        baseFolder, filename)
    img_raw = open(img_path, 'rb').read()
    key = hashlib.sha256(img_raw).hexdigest()    
    
    
    image = Image.open(img_path)
    width, height = image.size
    # image.show()
    
    filename = filename.encode('utf8')
    image_format = b'png'
    xmins = [loopvar/width for loopvar in data_dict['xmins']]
    xmaxs = [loopvar/width for loopvar in data_dict['xmaxs']]
    ymins = [loopvar/height for loopvar in data_dict['ymins']]
    ymaxs = [loopvar/height for loopvar in data_dict['ymaxs']]
    xmins =[(loopvar if loopvar <=1 else 1.0) for loopvar in xmins]
    ymins =[(loopvar if loopvar <=1 else 1.0) for loopvar in ymins]
    xmaxs =[(loopvar if loopvar <=1 else 1.0) for loopvar in xmaxs]
    ymaxs =[(loopvar if loopvar <=1 else 1.0) for loopvar in ymaxs]
    xmins =[(loopvar if loopvar >0 else 0.0) for loopvar in xmins]
    ymins =[(loopvar if loopvar >0 else 0.0) for loopvar in ymins]
    xmaxs =[(loopvar if loopvar >0 else 0.0) for loopvar in xmaxs]
    ymaxs =[(loopvar if loopvar >0 else 0.0) for loopvar in ymaxs]

    classes = data_dict['labels']
    classes_text = [(str(i).encode('utf8') if i!=10 else str('0').encode('utf8')) for i in classes]


    tf_example = tf.train.Example(features=tf.train.Features(feature={
        
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            filename])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            filename])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['png'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes))
    }))
    return tf_example

def construct_all_data(img_folder,mat_file_name,h5_name):
    img_bbox_data = img_boundingbox_data_constructor(os.path.join(img_folder,mat_file_name))
    img_bbox_data_grouped = img_bbox_data.groupby('img_name').apply(collapse_col) 
    # img_data = image_data_constuctor(img_folder, img_bbox_data_grouped)
    print('done constructing main dataframes...starting grouping')
    #df1 = img_bbox_data_grouped.merge(img_data,on='img_name',how='left')
    print('grouping done')
 
    return img_bbox_data,img_bbox_data_grouped

def generate_dataframes():
    testbbox,testimg = construct_all_data(test_folder,'digitStruct.mat','test_data_processed.h5')
    trainbbox,trainimg = construct_all_data(train_folder,'digitStruct.mat','train_data_processed.h5')
    # pickle.dump(testbbox,)
    return testbbox,trainbbox

def get_logger(name,log_file):
    logger = logging.getLogger(name)
	logger.setLevel(logging.DEBUG)
	if not logger.handlers:
	#if 
		log_file_handler = logging.FileHandler(log_file)
		log_file_handler.setLevel(logging.INFO)
		
		# Creating console handler which logs higher level messages to console
		console_handler = logging.StreamHandler()
		console_handler.setLevel(logging.INFO)
		
		# Creating formater and adding it to handlers
		formatter = logging.Formatter(
		    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		log_file_handler.setFormatter(formatter)
		console_handler.setFormatter(formatter)
		
		# Adding handlers to the logger
		logger.addHandler(log_file_handler)
		logger.addHandler(console_handler)
    return logger

if __name__=='__main__':
    logger = get_logger('detect_svhn','svhn_detect.log')
    # import pickle
    # with open(r'trainbbox.pkl', 'rb') as f:
    #     trainbbox = pickle.load(f)
    # with open(r'testbbox.pkl', 'rb') as f:
    #     testbbox = pickle.load(f)
    testbbox, trainbbox = generate_dataframes()
    
    index = -1
    imname = '1.png'
    imdict = {}
    xmins,xmaxs,ymins,ymaxs,width,height,label = [],[],[],[],[],[],[]
    ######get values from pandas dataframe - there are cleaner ways to do this probably
    for i in testbbox.iterrows():
        if i[0]>index:
            xmins.append(float(i[1]['left']))
            ymins.append(float(i[1]['top']))
            xmaxs.append(float(i[1]['right']))
            ymaxs.append(float(i[1]['bottom']))
            label.append(int(i[1]['label']))
        else:
            imdict.update({imname:{}})
            imdict[imname].update({'xmins':xmins,'ymins':ymins,'xmaxs':xmaxs,'ymaxs':ymaxs,'labels':label})
            xmins,xmaxs,ymins,ymaxs,width,height,label = [],[],[],[],[],[],[]
            xmins.append(float(i[1]['left']))
            ymins.append(float(i[1]['top']))
            xmaxs.append(float(i[1]['right']))
            ymaxs.append(float(i[1]['bottom']))
            label.append(int(i[1]['label']))
            logging.info('Image %s data processed' %(imname))
        imname = i[1]['img_name']
        index = i[0]
        
    #Create tf example and write to tf record        
    folder = r'C:\Users\Nishanth\Documents\Nishanth\test'
    writer = tf.io.TFRecordWriter('test.tfrecord')   
    for k,v in imdict.items():
        tf_example = create_tf_example(v, k,folder)    #image coordinates normalized here
        writer.write(tf_example.SerializeToString())
    writer.close()
    #construct_all_data(extra_folder,'digitStruct.mat','extra_data_processed.h5') #takes a long time