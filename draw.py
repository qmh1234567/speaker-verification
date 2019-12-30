import pandas as pd
import numpy as np
import constants as c
from sklearn import metrics
import matplotlib.pyplot as plt


def evaluate_metrics(b_y_true,b_y_pre):

    b_fpr,b_tpr,p_thresholds = metrics.roc_curve(b_y_true,b_y_pre,pos_label=1)
    b_frr = 1-b_tpr
    b_far = b_fpr


    # print(p_far.shape)
    print(b_far.shape)
    # length = min(len(p_far),len(b_far))
    # p_far = p_far[:length]
    # p_frr = p_frr[:length]

    # b_far = b_far[:length]
    # b_frr = b_frr[:length]
    fig ,axes = plt.subplots()
    axes.set_xscale('log')  
    axes.set_yscale('log')  

    # plt.plot(p_far*100,p_frr*100,color = 'red',label = 'proposed model')
    axes.plot(b_far*100,b_frr*100,color = 'green',label = 'baseline model')
    # axes.set_xticks([0.1,0.5,1,2,5,10,20,40,60])
    # axes.set_yticks([0.01,0.1,0.5,1,2,5,10,20,40,60])
    # set(axes,'xtick',[0.1,0.5,1,2,5,10,20,40,60])
    axes.set_xlim(0.01, 60)
    axes.set_ylim(0.1, 60)

    # # plt.plot(np.arange(0,100,1),np.arange(0,100,1))

    # plt.legend(fontsize=12)
    # plt.xlabel('False Alarm probability(in %)', fontsize=12)
    # plt.ylabel('Miss probability', fontsize=12)
    # plt.xticks([0.1,0.5,1,2,5,10,20,40,60])
    # plt.yticks([0.01,0.1,0.5,1,2,5,10,20,40,60])
    # plt.xlim([0.01,60])
    # plt.ylim([0.1,60])

    # plt.title(f'ROC curve, AUC score={auc}')
    plt.show()



# peo 说话人名字, datum 测试数据分数
def comput_far_frr(datum):
    up = np.max(datum)
    down = np.min(datum)

    print("up=",up)
    print("down",down)
    print(len(datum))

    pos_num = 18860

    far = []
    frr = []
    threshods = []

    dot_num = 1000  # DET图上的数据点数
    step = (up - down) / (dot_num + 1)
    print("step=",step)

    threshod = up
    size = len(datum)
    for i in range(dot_num):
        threshod -= step
        false_neg = 0
        false_pos = 0
        for d in range(size):
            if d < pos_num and datum[d] < threshod:
                false_pos += 1
            elif d > pos_num and datum[d] > threshod:
                false_neg += 1

        threshods.append(threshod)
        far.append(false_pos / size)
        frr.append(false_neg / size)

    far = np.array(far)
    frr = np.array(frr)
    threshods = np.array(threshods)
    return far,frr,threshods


def draw_DET(far,frr):

    # fig = plt.figure()

    plt.plot(far*100,frr*100,color = 'red',label = 'proposed model')
    # plt.plot(far*100,frr*100,color = 'green',label = 'baseline model')
    plt.plot(np.arange(0,60,1),np.arange(0,60,1))

    plt.legend(fontsize=12)
    plt.xlabel('False Alarm probability(in %)', fontsize=12)
    plt.ylabel('Miss probability', fontsize=12)
    # plt.xlim([0.01,15])
    # plt.ylim([0.01,15])
    plt.show()

def compute_min_DCF(thresholds,FAR,FRR):
    P_tar_2 = 0.01
    P_tar_3 = 0.001

    C_frr = C_far = 1

    min_DCF_2 = min_DCF_3 = 100
    # 每次选取一个不同的threshold，我们就可以得到一组FPR和TPR，即ROC曲线上的一点。
    for index in range(len(thresholds)):
        
        # if FRR[index] >0 and FAR[index] > 0:
        DCF_2 = C_frr*FRR[index]*100*P_tar_2 + C_far*FAR[index]*100*(1-P_tar_2)

        DCF_3 = C_frr*FRR[index]*100*P_tar_3 + C_far*FAR[index]*100*(1-P_tar_3)

        if DCF_2 < min_DCF_2 :
            min_DCF_2 = DCF_2
        if DCF_3 < min_DCF_3 :
            min_DCF_3 = DCF_3

    print("min_DCF_2=",min_DCF_2)
    print("min_DCF_3=",min_DCF_3)


    # 计算eer
    threshold_index = np.argmin(abs(FRR - FAR))  # 平衡点
    threshold = thresholds[threshold_index]
    print(FRR[threshold_index])
    print(FAR[threshold_index])
    eer = (FRR[threshold_index]+FAR[threshold_index])/2

    print("eer=",eer)


if __name__ == "__main__":


    p_far = np.load('./dataset/p_far.npy')
    p_frr = np.load('./dataset/p_frr.npy')

    b_far = np.load('./dataset/b_far_big.npy')
    b_frr = np.load('./dataset/b_frr_big.npy')

    y_true = np.load('./dataset/y_true.npy')
    b_y_pre = np.load('./dataset/y_pre.npy')


    # evaluate_metrics(y_true,b_y_pre)

    exit()
    # evaluate_metrics(p_far,p_frr,b_far,b_frr)

    y_pre = np.load('./dataset/y_pre.npy')
    far,frr,thresholds = comput_far_frr(y_pre)

    print(thresholds.shape)
    print(far.shape)
    draw_DET(far,frr)

    compute_min_DCF(thresholds,far,frr)


