""" 
@ author: Qmh
@ file_name: run_GE2E.py
@ time: 2019:11:03:10:48
""" 
import argparse
import tensorflow as tf
import re
import os
from model_GE2E import GE2E
from feeder import Feeder
import sys
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from progress.bar import Bar


def set_args():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--train_dev_set",type=str,default='/home/dsp/Documents/wav_data/SpeakerRecognition_dataset/voxceleb/data/dev/')
    parser.add_argument("--test_set",type=str,default='/home/dsp/Documents/wav_data/SpeakerRecognition_dataset/voxceleb/data/test/')
    parser.add_argument("--test_txt",type=str,default='./dataset/SV_dataset.txt')
    parser.add_argument("--mode",default="train",choices=["train",'test',"infer"])
    # model dir
    parser.add_argument("--model_dir",type=str,default='./checkpoint/ge2e_new')
    # initialization
    parser.add_argument("--init_type",type=str,default="uniform")
    parser.add_argument("--init_weight_range",type=float,default=0.1)
    # optimization
    parser.add_argument("--loss_type",default="optional_softmax",choices=["basic_softmax","optional_softmax"])
    parser.add_argument("--optimizer", type=str, default="sgd", help="type of optimizer")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate")
    parser.add_argument("--l2_norm_clip", type=float, default=3.0, help="L2-norm of gradient is clipped at")
    # data 
    parser.add_argument("--segment_length",type=float,default=1.6)
    parser.add_argument("--overlap_ratio",type=float,default=0.5)
    parser.add_argument("--spectrogram_scale", type=int, default=64)
    parser.add_argument("--sample_rate",type=int,default=16000)

    # train
    parser.add_argument("--max_step",type=int,default=500000)
    parser.add_argument("--decay_step",type=int,default= 30000000)
    parser.add_argument("--checkpoint_freq",type=int,default=1000)

    parser.add_argument("--num_spk_per_batch",type=int,default=10)
    parser.add_argument("--num_utt_per_batch",type=int,default=15)
    # LSTM
    parser.add_argument("--lstm_proj_clip",type=float,default=0.5)
    parser.add_argument("--num_lstm_stacks",type=int,default=3)
    parser.add_argument("--num_lstm_cells",type=int,default=768)
    parser.add_argument("--dim_lstm_projection",type=int,default=256)
    # sacle cosine similarity
    parser.add_argument("--scale_clip",type=float,default=0.01)
    args = parser.parse_args()
    return args

def train(args):
    model = GE2E(args) # model
    feeder = Feeder(args,"train") # feeder
    graph = model.set_up_model("train")
    with graph.as_default():
        saver = tf.train.Saver()  # 保存checkpoint
    with tf.Session(graph=graph) as sess:
        train_writer = tf.summary.FileWriter(args.model_dir,sess.graph)
        # 找出所有模型中最新的模型
        ckpt = tf.train.get_checkpoint_state(args.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(f'Restoring Variables from {ckpt.model_checkpoint_path}')
            # 重新加载模型
            saver.restore(sess,ckpt.model_checkpoint_path)
            start_step = sess.run(model.global_step)
        else:
            print("start from 0")
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            start_step = 1

        for num_step in range(start_step,args.max_step+1):
            print(f"current step: {num_step} th step")
            x_batch,y_batch = feeder.create_train_batch()
            summary,training_loss,_ = sess.run([model.sim_mat_summary,model.total_loss,model.optimize],
            feed_dict={model.input_batch:x_batch,model.target_batch:y_batch})
            train_writer.add_summary(summary,num_step)
            print("batch loss:"+str(training_loss))
            if num_step % args.checkpoint_freq == 0:
                save_path = saver.save(sess,args.model_dir+"/model.ckpt",global_step=model.global_step)
                print("model saved in file: %s / %d th step" % (save_path, sess.run(model.global_step)))

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
    threshold_index = np.argmin(abs(1-tpr - fpr))  # 平衡点
    threshold = thresholds[threshold_index]
    print("threshold=",threshold)

    eer = ((1-tpr)[threshold_index]+fpr[threshold_index])/2

    auc_score = metrics.roc_auc_score(y_true,y_pre,average='macro')

    y_pro =[ 1 if x > threshold else 0 for x in y_pre]
    acc = metrics.accuracy_score(y_true,y_pro)
    prauc = metrics.average_precision_score(y_true,y_pro,average='macro')
    return y_pro,eer,prauc,acc,auc_score





def test(args):
    model = GE2E(args)
    feeder = Feeder(args,"test")
    graph = model.set_up_model("test")
    with graph.as_default():
        saver = tf.train.Saver()
    with tf.Session(graph=graph) as sess:
        # saver.restore(sess,args.model_file)
        saver.restore(sess,'./checkpoint/ge2e_new/model.ckpt-10000')
        y_true = list(map(int,feeder.labels))[:1000]
        y_pre = []
        bar = Bar('predicting...',max=len(y_true),fill='#',suffix='%(percent)d%%')
        k = 0
        while len(feeder.wave_pairs)>0:
            if k > 999:
                break
            # bar.next()
            wav1_data,wav2_data,label = feeder.create_test_batch()
            # 每句话分段的向量 (2,256)
            wav1_out = sess.run(model.norm_out,feed_dict={model.input_batch:wav1_data})
            wav2_out = sess.run(model.norm_out,feed_dict={model.input_batch:wav2_data})
            # 每句话完整的向量 (256)
            wav1_dvector = np.mean(wav1_out,axis=0)  
            wav2_dvector = np.mean(wav2_out,axis=0)
            score = np.dot(wav1_dvector,wav2_dvector)/(np.linalg.norm(wav1_dvector)*np.linalg.norm(wav2_dvector))
            score = (score+1)/2 
            # print("score=",score)
            y_pre.append(score)
            k += 1

        # bar.finish()
        y_pro,eer,prauc,acc,auc_score = evaluate_metrics(y_true,y_pre)
        print(f'eer={eer}\t prauc={prauc} \t acc={acc}\t auc_score={auc_score}\t') 


    


def new_test(args):
    model = GE2E(args)
    feeder = Feeder(args,"test")
    graph = model.set_up_model("test")
    with graph.as_default():
        saver = tf.train.Saver()
    with tf.Session(graph=graph) as sess:
        # saver.restore(sess,args.model_file)
        saver.restore(sess,'./checkpoint/ge2e_new/model.ckpt-10000')
        y_true = list(map(int,feeder.labels))[:100]
        y_pre = []
        bar = Bar('predicting...',max=len(y_true),fill='#',suffix='%(percent)d%%')
        k = 0
        while len(feeder.wave_pairs)>0:
            if k > 999:
                break
            # bar.next()
            wav1_data,wav2_data,label = feeder.create_test_batch()
            # 每句话分段的向量 (2,256)
            wav1_out = sess.run(model.norm_out,feed_dict={model.input_batch:wav1_data})
            wav2_out = sess.run(model.norm_out,feed_dict={model.input_batch:wav2_data})
            # 每句话完整的向量 (256)
            wav1_dvector = np.mean(wav1_out,axis=0)  
            wav2_dvector = np.mean(wav2_out,axis=0)
            score = np.dot(wav1_dvector,wav2_dvector)/(np.linalg.norm(wav1_dvector)*np.linalg.norm(wav2_dvector))
            score = (score+1)/2 
            # print("score=",score)
            y_pre.append(score)
            k += 1

        # bar.finish()
        y_pro,eer,prauc,acc,auc_score = evaluate_metrics(y_true,y_pre)
        print(f'eer={eer}\t prauc={prauc} \t acc={acc}\t auc_score={auc_score}\t') 




def main(args):
    if args.mode == 'train':
        train(args)
    else:
        test(args)

if __name__ == "__main__":
    args = set_args() # params
    main(args)