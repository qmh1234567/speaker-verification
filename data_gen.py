""" 
@ author: Qmh
@ file_name: data_gen.py
@ time: 2019:10:29:09:06
""" 
from keras.utils import Sequence,np_utils
import numpy as np
import math
import pickle
import glob
import constants as c
import os
from collections import Counter
from sklearn.model_selection import train_test_split
import python_speech_features as psf
import tqdm
import json
import random
import re
# 数据生成器类
class BaseSequence(Sequence):
    def __init__(self,batch_size,x,y,num_class):
        self.x_y = np.concatenate((np.array(x).reshape(len(x),1),np.array(y).reshape(len(y),1)),axis=1)
        self.batch_size = batch_size
        self.num_class = num_class
    def __len__(self):
        return  math.ceil(len(self.x_y)/self.batch_size)

    def standard_normaliztion(self,x_array,epsilon=1e-12):
        return np.array([(x-np.mean(x))/max(np.std(x),epsilon) for x in x_array])

    def preprocess_x(self,batch_x,batch_y):
        X,Y = [],[]
        for index,x in enumerate(batch_x):
            try:
                with open(x,"rb") as f:
                    load_dict = pickle.load(f)
                    x = load_dict["LogMel_Features"]
                    # x = self.standard_normaliztion(x)
                    # x = x[:,:,np.newaxis]
                    x1 = self.standard_normaliztion(x)
                    x2 = psf.delta(x1,1)
                    x3 = psf.delta(x1,2)
                    x1 = x1[:,:,np.newaxis]
                    x2 = x2[:,:,np.newaxis]
                    x3 = x3[:,:,np.newaxis]
                    x = np.concatenate((x1,x2,x3),axis=2)
                    if x.shape[0]!=299:
                        continue
                    else:
                        y = np_utils.to_categorical(batch_y[index],num_classes=self.num_class,dtype='int8')
                        X.append(x)
                        Y.append(y)
            except Exception as e:
                print(e)
        X = np.array(X)
        Y = np.array(Y)
        return X,Y
    
    def __getitem__(self,index):
        batch_x = self.x_y[index*self.batch_size:(index+1)*self.batch_size,0]
        batch_y = self.x_y[index*self.batch_size:(index+1)*self.batch_size,1]
        
        batch_x,batch_y = self.preprocess_x(batch_x,batch_y)
        return batch_x,batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.x_y)


# 标签的映射函数
def Map_label_to_dict(labels):
    labels_to_id = {}
    i = 0
    for label in np.unique(labels):
        labels_to_id[label] = i
        i+= 1
    return labels_to_id

# 数据流生成器
def data_flow(batch_size):
    dataset = os.path.join(c.TRAIN_DEV_SET,c.TRAIN_PICKLE_NAME)
    pickle_list = [pickle for pickle in glob.glob(dataset+"/*.pickle")]
    pickle_list.sort()
    # speaker number = 1251
    audio_labels = [os.path.basename(pickle).split("_")[0] for pickle in pickle_list]
    # label to id
    labels_to_id = Map_label_to_dict(audio_labels)
    # split dataset
    train_paths,val_paths,train_labels,val_labels = train_test_split(pickle_list,audio_labels,stratify=audio_labels,test_size=0.2,random_state=42)
    train_labels = [labels_to_id[label] for label in train_labels]
    val_labels = [labels_to_id[label] for label in val_labels]
    # build sequence
    num_class = len(set(audio_labels))
    train_sequence = BaseSequence(batch_size,train_paths,train_labels,num_class)
    batch_x,batch_y = train_sequence.__getitem__(6)
    # print(batch_x.shape)
    # print(batch_y.shape)
    val_sequence = BaseSequence(batch_size,val_paths,val_labels,num_class)
    return train_sequence,val_sequence,num_class

# 标准化
def standard_normaliztion(x_array,epsilon=1e-12):
    return np.array([(x-np.mean(x))/max(np.std(x),epsilon) for x in x_array])

# 读取官方给的测试文件
def read_test_txt(sv_txt):
    results = []
    with open(sv_txt,'r') as f:
        results = f.read().splitlines()
    labels = []
    test_A = []
    test_B = []
    for line in results:
        label,A,B = line.split(" ")
        labels.append(label)
        test_A.append(A)
        test_B.append(B)
    return labels,test_A,test_B


def new_read_test_txt(sv_txt):
    results = []
    with open(sv_txt,'r') as f:
        results = f.read().splitlines()
    labels = []
    test_A = []
    test_B = []
    enrollment_spks = []
    for line in results:
        label,A,B = line.split(" ")
        labels.append(label)
        test_A.append(A)
        speaker = A.split("/")[0]
        # if speaker not in enrollment_spks:
        enrollment_spks.append(speaker)
        test_B.append(B)

    return labels,test_A,test_B,Counter(enrollment_spks)


# # 加载所有的测试音频 3s-utterance for baseline
# def load_all_data():
#     dataset = os.path.join(c.TEST_SET,c.TEST_PICKLE_NAME)
#     pickle_list = [pickle for pickle in glob.iglob(dataset+"/*.pickle")]
#     test_dict = {}
#     for pk in tqdm.tqdm(pickle_list):
#         try:
#             with open(pk,'rb') as f:
#                 load_dict = pickle.load(f)
#                 x = load_dict["LogMel_Features"]
#                 # x = standard_normaliztion(x)
#                 # x = x[:,:,np.newaxis]
#                 x1 = standard_normaliztion(x)
#                 x2 = psf.delta(x1,1)
#                 x3 = psf.delta(x1,2)
#                 x1 = x1[:,:,np.newaxis]
#                 x2 = x2[:,:,np.newaxis]
#                 x3 = x3[:,:,np.newaxis]
#                 x = np.concatenate((x1,x2,x3),axis=2)
#                 pickle_name = str(os.path.basename(pk))
#                 if x.shape[0]!=299:
#                     print(x.shape)
#                 else:
#                     pickle_name = str(os.path.basename(pk))
#                     test_dict[pickle_name]=x
#         except Exception as e:
#             print(e)
#             exit()
#     return test_dict

# 3s-segment for proposed model
def load_all_data():
    dataset = os.path.join(c.TEST_SET,c.TEST_PICKLE_NAME)
    pickle_list = [pickle for pickle in glob.iglob(dataset+"/*.pickle")]
    test_dict = {}
    for pk in tqdm.tqdm(pickle_list):
        try:
            with open(pk,'rb') as f:
                load_dict = pickle.load(f)
                xx = load_dict["LogMel_Features"]
                x_vectors = []
                for x in xx:
                # x = standard_normaliztion(x)
                # x = x[:,:,np.newaxis]
                    x1 = standard_normaliztion(x)
                    x2 = psf.delta(x1,1)
                    x3 = psf.delta(x1,2)
                    x1 = x1[:,:,np.newaxis]
                    x2 = x2[:,:,np.newaxis]
                    x3 = x3[:,:,np.newaxis]
                    x = np.concatenate((x1,x2,x3),axis=2)
                    if x.shape[0]!=299:
                        print("x.shape=",x.shape)
                    x_vectors.append(x)
                x_vectors = np.array(x_vectors)
                # print("x_vectors.shape",x_vectors.shape)
                pickle_name = str(os.path.basename(pk))
                test_dict[pickle_name]=x_vectors
        except Exception as e:
            print(e)
            exit()
    return test_dict

if __name__ == "__main__":
    batch_size = 32
    data_flow(batch_size)
    # load_test_data()
