#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# __author__: Qmh
# __file_name__: run.py
# __time__: 2019:06:27:20:53

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
import keras.backend.tensorflow_backend as KTF
from keras.layers import Flatten
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,RepeatedKFold
from keras import optimizers
import glob
import pickle
from data_gen import data_flow,load_all_data,read_test_txt,new_read_test_txt
import tqdm

# OPTIONAL: control usage of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# # config.gpu_options.per_process_gpu_memory_fraction = 0.9
# sess = tf.Session(config=config)


def model_fn(model,num_class):
    # model = models.SE_ResNet(c.INPUT_SHPE,num_class)
    x = model.output
    x = Dense(num_class,activation='softmax',name='softmax')(x)
    model = Model(model.input,x)
    print(model.summary())
    sgd = optimizers.SGD(lr=c.LEARN_RATE,momentum=0.9)
    model.compile(loss='categorical_crossentropy',optimizer = sgd,metrics=['accuracy'])
    return model

def evaluate_metrics(y_true,y_pre):
    fpr,tpr,thresholds = metrics.roc_curve(y_true,y_pre,pos_label=1)
    auc = metrics.auc(fpr,tpr)
    plt.figure()
    plt.plot(fpr,tpr,color = 'green',label = 'ROC')
    plt.plot(np.arange(1,0,-0.01),np.arange(0,1,0.01))
    plt.legend()
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title(f'ROC curve, AUC score={auc}')
    plt.show()
    
    # far = fpr frr = 1-tpr   index = np.argmin(frr-far)
    # 计算DCF
    P_tar_2 = 0.01
    P_tar_3 = 0.001

    C_frr = C_far = 1
    FRR = 1-tpr
    FAR = fpr
    min_DCF_2 = min_DCF_3 = 100
    print("len(thresholds)=",len(thresholds))

    plt.figure()

    # np.save('./dataset/b_far_big.npy',FAR)
    # np.save('./dataset/b_frr_big.npy',FRR)

    plt.plot(FAR*100,FRR*100,color = 'green',label = 'model')
    plt.plot(np.arange(0,100,1),np.arange(0,100,1))

    plt.legend()
    plt.xlabel('far')
    plt.ylabel('frr')
    plt.xlim([0.1,60])
    plt.ylim([0.01,60])
    # plt.title(f'ROC curve, AUC score={auc}')
    plt.show()

    # 每次选取一个不同的threshold，我们就可以得到一组FPR和TPR，即ROC曲线上的一点。
    for index in range(len(thresholds)):
        
        # if FRR[index] >0 and FAR[index] > 0:
        DCF_2 = C_frr*FRR[index]*100*P_tar_2 + C_far*FAR[index]*100*(1-P_tar_2)

        DCF_3 = C_frr*FRR[index]*100*P_tar_3 + C_far*FAR[index]*100*(1-P_tar_3)

        if DCF_2 < min_DCF_2 :
            min_DCF_2 = DCF_2
        if DCF_3 < min_DCF_3 and DCF_3 > min_DCF_2+0.1:
            min_DCF_3 = DCF_3

    print("min_DCF_2=",min_DCF_2)
    print("min_DCF_3=",min_DCF_3)

    # 计算eer
    threshold_index = np.argmin(abs(FRR - FAR))  # 平衡点
    threshold = thresholds[threshold_index]
    eer = (FRR[threshold_index]+FAR[threshold_index])/2

    auc_score = metrics.roc_auc_score(y_true,y_pre,average='macro')

    y_pro =[ 1 if x > threshold else 0 for x in y_pre]
    acc = metrics.accuracy_score(y_true,y_pro)
    prauc = metrics.average_precision_score(y_true,y_pro,average='macro')
    return y_pro,eer,prauc,acc,auc_score


# # 3s utterance for basic model
# def Map_path_to_feature(wav,test_dict,model):
#     pickle_name = "_".join(wav.split("/")[-3:]).replace("wav",'pickle')
#     label = wav.split("/")[-3]
#     vec = test_dict[pickle_name]
#     vec = vec[np.newaxis,:,:,:]
#     pre = np.squeeze(model.predict(vec))
#     return pre

# # 3s utterance for basic model
# def new_evaluate_model(model,model_path,SV_txt):
#     # 加载模型
#     model.load_weights(model_path,by_name='True')

#     test_dict = load_all_data()
#     labels,test_A,test_B,enrollment_spks = new_read_test_txt(SV_txt)
#     y_true = list(map(int,labels))

#     y_pre = []

#     enroll_dict = {}

#     # 得到每个说话人模型
#     def get_vectors_from_each_speaker(audio_list,test_dict,model):
        
#         enroll_vecs = []
#         audio_list = list(set(audio_list))  # 去除同一个说话人重复的音频
#         for audio in audio_list:
#             pickle_name = "_".join(audio.split("/")[-3:]).replace("wav",'pickle')
#             vec = test_dict[pickle_name] # (299,64,3)
#             enroll_vecs.append(vec)
#         enroll_vecs = np.array(enroll_vecs)  #(num_utterance,299,64,3)
#         enroll_vecs = model.predict(enroll_vecs)   # （num_utterance，1024）
#         # 求均值
#         enroll_pred = np.mean(enroll_vecs,axis=0)  #  求均值  (1024)
#         return enroll_pred

#     # 遍历得到每个人的音频片段,然后得到每个说话人模型
#     k = 0
#     for item in tqdm.tqdm(enrollment_spks.items()):
#         # enroll_dict[item[0]] = test_A[k:k+item[1]]
#         audio_list = test_A[k:k+item[1]]
#         enroll_dict[item[0]] = get_vectors_from_each_speaker(audio_list,test_dict,model)
#         k = k + item[1]
    
#     # # 再计算测试语句
#     for index in tqdm.tqdm(range(0,len(y_true))):
#         enroll_pre = enroll_dict[test_A[index].split('/')[0]]
#         test_pre = Map_path_to_feature(test_B[index],test_dict,model)
#         score = np.dot(enroll_pre,test_pre)/(np.linalg.norm(enroll_pre)*np.linalg.norm(test_pre))
#         # print("score=",score)
#         score = (score+1)/2 # 将余弦距离变成正值
#         y_pre.append(score)

#     np.save("./dataset/p_y_pre.npy",y_pre)
#     y_pro,eer,prauc,acc,auc_score = evaluate_metrics(y_true,y_pre)
#     print(f'eer={eer}\t prauc={prauc} \t acc={acc}\t auc_score={auc_score}\t')


# 3-segment for proposed model
def Map_path_to_feature(wav,test_dict,model):
    pickle_name = "_".join(wav.split("/")[-3:]).replace("wav",'pickle')
    label = wav.split("/")[-3]
    vecs = test_dict[pickle_name]  #(segments,299,64,3)
    pre = model.predict(vecs)  # (segments,1024)
    pre = np.mean(pre,axis=0)  # (1024)
    return pre

# 3 segment for proposed model
def new_evaluate_model(model,model_path,SV_txt):
    # 加载模型
    model.load_weights(model_path,by_name='True')

    test_dict = load_all_data()
    labels,test_A,test_B,enrollment_spks = new_read_test_txt(SV_txt)
    y_true = list(map(int,labels))
    np.save('./dataset/p_y_true.npy',y_true)
    # exit()
    y_pre = []

    enroll_dict = {}

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
        test_pre = Map_path_to_feature(test_B[index],test_dict,model)
        score = np.dot(enroll_pre,test_pre)/(np.linalg.norm(enroll_pre)*np.linalg.norm(test_pre))
        # print("score=",score)
        score = (score+1)/2 # 将余弦距离变成正值
        y_pre.append(score)

    np.save("./dataset/p_y_pre.npy",y_pre)
    y_pro,eer,prauc,acc,auc_score = evaluate_metrics(y_true,y_pre)
    print(f'eer={eer}\t prauc={prauc} \t acc={acc}\t auc_score={auc_score}\t')


# # evaluation for each utterance
# def evaluate_model(model,model_path,SV_txt):
#     model.load_weights(model_path,by_name='True')
#     test_dict = load_all_data()
#     labels,test_A,test_B = read_test_txt(SV_txt)
#     y_true = list(map(int,labels))[:1000]
#     y_pre = []
#     for index in tqdm.tqdm(range(0,len(y_true))):
#         enroll_pre = Map_path_to_feature(test_A[index],test_dict,model)
#         test_pre = Map_path_to_feature(test_B[index],test_dict,model)
#         score = np.dot(enroll_pre,test_pre)/(np.linalg.norm(enroll_pre)*np.linalg.norm(test_pre))
#         # print("score=",score)
#         score = (score+1)/2 # 将余弦距离变成正值
#         y_pre.append(score)
#     y_pro,eer,prauc,acc,auc_score = evaluate_metrics(y_true,y_pre)
#     print(f'eer={eer}\t prauc={prauc} \t acc={acc}\t auc_score={auc_score}\t')

def main(typeName,model_path):
    # model = models.Basic_Model(c.INPUT_SHPE,[2,3,2])
    model = models.Proposed_Model(c.INPUT_SHPE,[2,3,2])
    if typeName.startswith('train'):
        # 加载预训练文件
        model.load_weights('./checkpoint/best_new.h5',by_name=True)  #  best_new eer = 0.0415
        # batch sequence
        train_sequence,val_sequence,num_class = data_flow(c.BATCH_SIZE)
        # train model 
        model = model_fn(model,num_class)
        model.fit_generator(train_sequence,
            steps_per_epoch=len(train_sequence),
            epochs = 50,
            validation_data=val_sequence,
            callbacks=[
                ModelCheckpoint(model_path,monitor='val_loss',save_best_only=True,mode='min',save_weights_only=True),
                ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5,mode='min'),
                EarlyStopping(monitor='val_loss',patience=10),
        ])
        print("training done")
    else:
        SV_txt = './dataset/SV_dataset.txt'
        new_evaluate_model(model,model_path,SV_txt)
        # evaluate_model(model,model_path,SV_txt)



if __name__ == "__main__":
    mode = sys.argv[1]
    model_path = os.path.join(c.MODEL_DIR,'best1.h5')
    # model_path = os.path.join(c.MODEL_DIR,'proposed_eer=4.36.h5')
    main(mode,model_path)
 