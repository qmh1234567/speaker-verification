""" 
@ author: Qmh
@ file_name: fine_tune.py
@ time: 2019:12:12:15:53
""" 
import pandas as pd
import constants as c
import os
import tensorflow as tf
from collections import Counter
import numpy as np
from progress.bar import Bar
import models
from keras.layers import Dense
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
import sys
import keras.backend as K
from keras.layers import Flatten
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.layers import Input, Lambda, concatenate
from sklearn.model_selection import train_test_split,RepeatedKFold
from keras import optimizers
from keras.utils import Sequence,np_utils
import tqdm
import python_speech_features as psf
import pickle
import random
import glob
import re 
import argparse
import math
import keras
from data_gen import new_read_test_txt,load_all_data
from run import evaluate_metrics

from keras.utils import plot_model


parser = argparse.ArgumentParser()


# data
parser.add_argument('--test_set',default='/home/dsp/Documents/wav_data/SpeakerRecognition_dataset/voxceleb/data/test/',type=str)
parser.add_argument('--pickle_folder',default='pickle1',type=str,help='folder of pickle files')
parser.add_argument('--SV_txt',default='./dataset/SV_dataset.txt',type=str)

parser.add_argument('--mode',required=True,choices=['train','test'])
parser.add_argument('--input_shape',default=(299,64,3))
parser.add_argument('--learning_rate',default=0.001,type=float)
parser.add_argument('--momentum',default=0.9,type=float)
parser.add_argument('--model_path',default='./checkpoint/fine_tune_best1.h5',type=str)
parser.add_argument('--pretrained_model_path',default='./checkpoint/proposed_eer=4.36.h5',type=str)

# train
parser.add_argument('--epochs',default=20,type=int)
parser.add_argument('--num_spk_per_batch',default=2,type=int)
parser.add_argument('--num_utt_per_batch',default=20,type=int)

parser.add_argument('--train_size',default=500,type=int)
parser.add_argument('--val_size',default=125,type=int)
parser.add_argument('--batch_size',default=5,type=int)


class Feeder():
    def __init__(self,args):
        self.args = args
        _,self.test_A,_,self.enrollment_spks = new_read_test_txt(args.SV_txt)
        self.num_spk_per_batch = args.num_spk_per_batch # 每个batch取5个说话人
        self.num_utt_per_batch = args.num_utt_per_batch  # 每个batch 10句话
    
    # 转化音频的路径
    def convert_path_to_pickle(self,audio_name):
        pickle_name = "_".join(audio_name.split("/")[-3:]).replace("wav",'pickle')
        pickle_dir = os.path.join(args.test_set,args.pickle_folder)
        pickle_path = os.path.join(pickle_dir,pickle_name)
        return pickle_path
    
    # 标准化
    def standard_normaliztion(self,x_array,epsilon=1e-12):
        return np.array([(x-np.mean(x))/max(np.std(x),epsilon) for x in x_array])
    
    # 加载pickle文件
    def load_pickle(self,pickle_name):
        with open(pickle_name,'rb') as f:
            load_dict = pickle.load(f)
            x = load_dict["LogMel_Features"]
            # x = standard_normaliztion(x)
            # x = x[:,:,np.newaxis]
            x1 = self.standard_normaliztion(x)
            x2 = psf.delta(x1,1)
            x3 = psf.delta(x1,2)
            x1 = x1[:,:,np.newaxis]
            x2 = x2[:,:,np.newaxis]
            x3 = x3[:,:,np.newaxis]
            x = np.concatenate((x1,x2,x3),axis=2)
            if x.shape[0]!=299:
                print("x.shape=",x.shape)
                exit()
            x_vector = np.array(x)
        return x_vector
    
    def create_train_spk(self):
        spk_list = list(self.enrollment_spks.keys())
        # print(spk_list)
        # 一次取M个说话人
        spk_batch = random.sample(range(len(spk_list)),k=self.num_spk_per_batch)
        # M*N
        batch_y = [spk for spk in range(self.num_spk_per_batch) for i in range(self.num_utt_per_batch)]
        batch_x = []
        # M个说话人
        for spk_id in spk_batch:  
            # 寻找该说话人的所有句子
            audio_list  = [audio for audio in self.test_A if re.search(spk_list[spk_id],audio)]
            audio_list = list(set(audio_list))
            num_utterances = len(audio_list)

            if num_utterances < self.num_utt_per_batch:
                print(f"less than {self.num_utt_per_batch} utts")
                pass
            # 选择N句话
            utt_idx_list = random.sample(range(num_utterances),k=self.num_utt_per_batch)
            for utt_idx in utt_idx_list:
                utt_audio = audio_list[utt_idx]
                pickle_path = self.convert_path_to_pickle(utt_audio)
                x = self.load_pickle(pickle_path)
                if x.shape[0] != 299:
                    print("x.shape=",x.shape)
                    continue
                batch_x.append(x)
        batch_x = np.asarray(batch_x)
        batch_y = np.asarray(batch_y)
        return batch_x,batch_y
    
    def select_utt_idx_for_spk(self,spk_id,is_enroll=False):
        if not is_enroll:
            utt_idx = np.random.randint(self.num_utt_per_batch)
            utt_idx = spk_id*self.num_utt_per_batch + utt_idx
        else:
            utt_idx = list(range(spk_id*self.num_utt_per_batch,(spk_id+1)*self.num_utt_per_batch))
        return utt_idx

    def create_train_batch(self):
        x_batch,y_batch = self.create_train_spk()
        # 选择1个anchor
        anchor_id = np.random.randint(self.num_spk_per_batch)
        # 选择1个enrollment
        enroll_id = np.random.randint(self.num_spk_per_batch)
        # 选择1+(M-1)*N个句子,M=5,N=10
        # print("anchor=",anchor_id)
        # print("enroll_id=",enroll_id)

        y_train = [0] # 初始化标签为False
        utt_idx = self.select_utt_idx_for_spk(anchor_id,is_enroll=False)
        evaluation_vec = x_batch[utt_idx]
        evaluation_vec = evaluation_vec[np.newaxis,:,:,:]
        # print("evaluation_vec=",evaluation_vec.shape)
 
        utt_idx_enroll = self.select_utt_idx_for_spk(enroll_id,is_enroll=True)
        # print("utt_idx_enroll=",utt_idx_enroll)
        # 如果测试的句子在注册句子中，则删去
        if utt_idx in utt_idx_enroll:
            utt_idx_enroll.remove(utt_idx)
            # print("utt_idx_enroll=",utt_idx_enroll)
            y_train = [1] # 标签为True 表示是同一个人所说
        else:
            utt_idx_enroll.pop()

        enroll_vec = x_batch[utt_idx_enroll]
        # print("enroll_vec=",enroll_vec.shape)
        # exit()
        # 垂直拼接
        x_train = np.vstack([evaluation_vec,enroll_vec])  # (N,299,64,3) or (N-1,299,64,3)
        # x_train = np.reshape(x_train,(x_train.shape[0]*299,64,3))
        y_train = np.asarray(y_train)
        return x_train,y_train



# 创建数据和标签
def load_data(args,is_train):
    feeder = Feeder(args)
    x_train,y_train = [],[]
    if is_train:
        total_size = args.train_size
    else:
        total_size = args.val_size

    for i in range(total_size):
        x,y = feeder.create_train_batch()
        x_train.append(x)
        y_train.append(y)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    np.save('x_train.npy',x_train)
    np.save('y_train.npy',y_train)

    return x_train,y_train


# 数据生成器类
class BaseSequence(Sequence):
    def __init__(self,batch_size,x,y,model):
        self.x = x
        self.y = y
        self.model = model
        self.batch_size = batch_size

    def __len__(self):
        return  math.ceil(len(self.x)/self.batch_size)

    def process(self,batch_x):
        new_batch = []
        for index,x in enumerate(batch_x):
            try:
                # 生成embedding
                x_in = x[0,:,:,:]
            except Exception as e:
                print(x_in.shape)
                exit()
            x_in = np.expand_dims(x_in, axis=0)
            test_x = np.squeeze(self.model.predict(x_in))#(1,1024)
            test_x = test_x.reshape(1,test_x.shape[-1])  

            x_in = x[1:,:,:,:]
            enroll_x = self.model.predict(x_in) #(9,1024)
            enroll_x = np.mean(enroll_x,axis=0).reshape(1,enroll_x.shape[-1])  # (1,1024)
            # # 求cosine similarity
            # score = np.dot(enroll_x,test_x)/(np.linalg.norm(enroll_x)*np.linalg.norm(test_x))

            x_train = np.concatenate((test_x,enroll_x),axis=0)    
            new_batch.append(x_train)

        return np.asarray(new_batch)  # (batch_size,2,1024)

    def __getitem__(self,index):
        batch_x = np.squeeze(self.x[index*self.batch_size:(index+1)*self.batch_size])
        batch_y = np.squeeze(self.y[index*self.batch_size:(index+1)*self.batch_size])
        
        batch_x = self.process(batch_x)

        return batch_x,batch_y

    def on_epoch_end(self):
        shuffle_idx = np.random.permutation(len(self.x))
        self.x = self.x[shuffle_idx]
        self.y = self.y[shuffle_idx]


# 数据流生成器
def data_flow(args,model):
    print("loading data...")
    x_train,y_train = load_data(args,is_train=True)
    # print(x_train.shape)
    # print(y_train)
    # print(y_train.shape)

    train_sequence = BaseSequence(args.batch_size,x_train,y_train,model)
    batch_x,batch_y = train_sequence.__getitem__(2)
    # print(batch_x.shape)
    # print(batch_y.shape)
    # exit()

    x_val,y_val = load_data(args,is_train=False)

    val_sequence = BaseSequence(args.batch_size,x_val,y_val,model)

    
    return train_sequence,val_sequence

def model_fn():
    input_shape = (2,1024)
    new_model = Finetune_Model(input_shape)
    print(new_model.summary())
    sgd = optimizers.SGD(lr=c.LEARN_RATE,momentum=0.9)
    new_model.compile(loss='binary_crossentropy',optimizer = sgd,metrics=['accuracy'])
    return new_model

  
def main(args):
    # build model
    # model = models.SE_ResNet(c.INPUT_SHPE)
    # model = models.Basic_Model(c.INPUT_SHPE,[2,3,2])
    model = models.Proposed_Model(c.INPUT_SHPE,[2,3,2])
    
    print(model.predict(np.zeros((1,299,64,3))))

    new_model = model_fn()
    
 
    if args.mode == 'train':
        # 加载预训练文件
        model.load_weights(args.pretrained_model_path,by_name=True)
        print(model.predict(np.zeros((1,299,64,3))))

        new_model.load_weights(args.model_path,by_name=True)

        # load data
        train_sequence,val_sequence = data_flow(args,model)

        new_model.fit_generator(train_sequence,
            steps_per_epoch=len(train_sequence),
            epochs = args.epochs,
            validation_data=val_sequence,
            validation_steps=len(val_sequence),
            callbacks=[
                ModelCheckpoint(args.model_path,monitor='val_loss',save_best_only=True,mode='min',save_weights_only=True),
                ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5,mode='min'),
                EarlyStopping(monitor='val_loss',patience=10),
        ])
        print("training done")

    elif args.mode == 'test':
        model.load_weights(args.pretrained_model_path,by_name='True')
        new_model.load_weights(args.model_path)
        # load data
        labels,test_A,test_B,enrollment_spks = new_read_test_txt(args.SV_txt)
        test_dict = load_all_data()

        y_true = list(map(int,labels))
        y_pre = []

        enroll_dict = {}

        # 遍历得到每个人的音频片段,然后得到每个说话人模型
        k = 0
        for item in tqdm.tqdm(enrollment_spks.items()):
            # enroll_dict[item[0]] = test_A[k:k+item[1]]
            audio_list = test_A[k:k+item[1]]
            enroll_dict[item[0]] = get_vectors_from_each_speaker(audio_list,test_dict,model)
            k = k + item[1]
        
        # # 再计算测试语句
        for index in tqdm.tqdm(range(0,len(y_true))):
            enroll_pre = enroll_dict[test_A[index].split('/')[0]]
            enroll_pre = enroll_pre.reshape(1,enroll_pre.shape[-1])

            test_pre = Map_path_to_feature(test_B[index],test_dict,model)
            test_pre = test_pre.reshape(1,test_pre.shape[-1])

            x_train = np.concatenate((test_pre,enroll_pre),axis=0)   
            # print(x_train.shape)

            x_train = x_train[np.newaxis,:,:]
            # print(x_train.shape)
            
            y_pred = new_model.predict(x_train)
            # print("y_pred=",y_pred)
            # score = np.dot(enroll_pre,test_pre)/(np.linalg.norm(enroll_pre)*np.linalg.norm(test_pre))
            # print("score=",score)
            # score = (score+1)/2 # 将余弦距离变成正值
            y_pre.extend(y_pred)
        y_pro,eer,prauc,acc,auc_score = evaluate_metrics(y_true,y_pre)
        print(f'eer={eer}\t prauc={prauc} \t acc={acc}\t auc_score={auc_score}\t')



# 3-segment for proposed model
def Map_path_to_feature(wav,test_dict,model):
    pickle_name = "_".join(wav.split("/")[-3:]).replace("wav",'pickle')
    label = wav.split("/")[-3]
    vecs = test_dict[pickle_name]  #(segments,299,64,3)
    pre = model.predict(vecs)  # (segments,1024)
    pre = np.mean(pre,axis=0)  # (1024)
    return pre


  # 得到每个说话人模型
def get_vectors_from_each_speaker(audio_list,test_dict,model):
    
    enroll_vecs = []
    audio_list = list(set(audio_list))  # 去除同一个说话人重复的音频
    for audio in audio_list:
        pickle_name = "_".join(audio.split("/")[-3:]).replace("wav",'pickle')
        vec = test_dict[pickle_name] # （segments,299,64,3)
        # 预测一段句子
        enroll_pre = model.predict(vec)  #(segments,1024)
        enroll_pre = np.mean(enroll_pre,axis=0)  # (1024)
        enroll_vecs.append(enroll_pre)
    enroll_vecs = np.array(enroll_vecs)  #(num_utterance,400)
    # 求均值
    enroll_pred = np.mean(enroll_vecs,axis=0)  #  求均值  (1024)
    return enroll_pred



def Finetune_Model(input_shape):
    x_in = Input(input_shape,name='input')
    # print("x_in.shape",x_in.shape)

    def slice(x,index):
        return x[:,index,:]
    
    x1 = Lambda(slice,arguments={'index':0})(x_in)
    x2 = Lambda(slice,arguments={'index':1})(x_in)

    # print("x1.shape=",x1.shape)

    # 添加计算cosine similarity 的层
    class CosineLayer():
        def __call__(self, x1, x2):
            def _cosine(x):
                print(len(x))
                print(x[1].shape)
                dot1 = K.batch_dot(x[0], x[1], axes=1)  # a*b
                dot2 = K.batch_dot(x[0], x[0], axes=1) # a*a
                dot3 = K.batch_dot(x[1], x[1], axes=1) # b*b
                max_ = K.maximum(K.sqrt(dot2 * dot3), K.epsilon()) # a*b/(a*a)*(b*b)
                return dot1 / max_
    
            output_shape = (1,)
            value = Lambda(_cosine,output_shape=output_shape)([x1, x2])
            return value

    cosine = CosineLayer()
    x = cosine(x1, x2)
    x = Dense(1,activation='sigmoid',name='logistic')(x)    
    # print(x.shape)
    return Model(x_in,x,name='logistic')



def main1(args):
    # model = models.Basic_Model(c.INPUT_SHPE,[2,3,2])

    new_model = Create_model(args)

    plot_model(new_model,'multi_input_model.png')

    if args.mode == 'train':
        # load data
        train_generator = myGenerator(args,is_train=True)
        val_generator = myGenerator(args,is_train=False)
        sgd = optimizers.SGD(lr=0.0001,momentum=0.9)
        new_model.compile(loss='binary_crossentropy',optimizer = sgd,metrics=['accuracy'])
        # train_sequence,val_sequence = data_flow(args,model)
        new_model.fit(train_generator,epochs=10)
        # new_model.fit_generator(train_generator,
        #     steps_per_epoch=args.train_size,
        #     epochs = 10,
        #     validation_data=val_generator,
        #     validation_steps=args.val_size,
        #     callbacks=[
        #         ModelCheckpoint(args.model_path,monitor='val_loss',save_best_only=True,mode='min',save_weights_only=True),
        #         ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5,mode='min'),
        #         EarlyStopping(monitor='val_loss',patience=10),
        # ])
        print("training done")


def myGenerator(args,is_train):
    # loading data
    feeder = Feeder(args)
    
    if is_train:
        total_size=args.train_size
    else:
        total_size= args.val_size

    i = -1

    loopcount = total_size 

    while 1:
        i = i + 1
        i = i%loopcount
        
        X_train,Y_train= feeder.create_train_batch()
        X_train = X_train.tolist()
        Y_train = Y_train.tolist()
        
        # for index in range(len(X_train)):
        #     dict1['input'+str(index)] = X_train[index]
          
        yield (X_train,Y_train)

    return myGenerator

def Create_model(args):
    input_shape = c.INPUT_SHPE

    in_list = []
    for i in range(args.num_utt_per_batch):
        in_list.append(Input(input_shape,name=f'input{i}'))

    
    backbone = models.Basic_Model(c.INPUT_SHPE,[2,3,2])
    # 加载预训练文件
    backbone.load_weights(args.pretrained_model_path,by_name=True)

    # 测试1句
    x1 = backbone(in_list[0])  # （?,1024)

    # print(out1.shape)
    # x1 = Lambda(lambda y: backbone(in_list[0]),name='x1')
    print(x1.shape)

    # 注册9句
    out1 = backbone(in_list[1])
    for i in range(2,args.num_utt_per_batch):
        out1 = concatenate([out1,backbone(in_list[i])],axis=0)
        # out_array = np.concatenate((out_array,backbone(in_list[i])),axis=0)
        # out_list.append(backbone(in_list[i])) 
    print(out1.shape)

    # 对注册语句求平均
     # 求enrollment
    x2 = Lambda(lambda y: K.reshape(K.mean(y,axis=0),(-1,1024)),name='average')(out1)

    print(x2.shape)

    # 添加计算cosine similarity 的层
    class CosineLayer():
        def __call__(self, x1, x2):
            def _cosine(x):
                print(len(x))
                print(x[1].shape)
                dot1 = K.batch_dot(x[0], x[1], axes=1)  # a*b
                dot2 = K.batch_dot(x[0], x[0], axes=1) # a*a
                dot3 = K.batch_dot(x[1], x[1], axes=1) # b*b
                max_ = K.maximum(K.sqrt(dot2 * dot3), K.epsilon()) # a*b/(a*a)*(b*b)
                return dot1 / max_
    
            output_shape = (1,)
            value = Lambda(_cosine,output_shape=output_shape)([x1, x2])
            return value

    cosine = CosineLayer()

    x = cosine(x1, x2)

    x = Dense(1,activation='sigmoid',name='output')(x)    

    model = Model(inputs=in_list, outputs=x)

    print(model.summary())

    return model






if __name__ == "__main__":
    # param
    args = parser.parse_args()
    main(args)
    # main1(args)