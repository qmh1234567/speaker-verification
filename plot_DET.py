import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from scipy.stats import norm
import numpy as np

def plot_DET_curve():
    # 设置刻度范围
    pmiss_min = 0.001

    pmiss_max = 0.6  

    pfa_min = 0.001

    pfa_max = 0.6

    # 刻度设置
    pticks = [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005,
            0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
            0.1, 0.2, 0.4, 0.6, 0.8, 0.9,
            0.95, 0.98, 0.99, 0.995, 0.998, 0.999,
            0.9995, 0.9998, 0.9999, 0.99995, 0.99998, 0.99999]

    # 刻度*100
    xlabels = [' 0.001', ' 0.002', ' 0.005', ' 0.01 ', ' 0.02 ', ' 0.05 ',
            '  0.1 ', '  0.2 ', ' 0.5  ', '  1   ', '  2   ', '  5   ',
            '  10  ', '  20  ', '  40  ', '  60  ', '  80  ', '  90  ',
            '  95  ', '  98  ', '  99  ', ' 99.5 ', ' 99.8 ', ' 99.9 ',
            ' 99.95', ' 99.98', ' 99.99', '99.995', '99.998', '99.999']

    ylabels = xlabels

    # 确定刻度范围
    n = len(pticks)
    # 倒叙 
    for k, v in enumerate(pticks[::-1]):
        if pmiss_min <= v:
            tmin_miss = n - k - 1   # 移动最小值索引位置
        if pfa_min <= v:
            tmin_fa = n - k - 1   # 移动最小值索引位置
    # 正序
    for k, v in enumerate(pticks):
        if pmiss_max >= v:   
            tmax_miss = k+1         # 移动最大值索引位置
        if pfa_max >= v:            
            tmax_fa = k+1            # 移动最大值索引位置

    # FRR
    plt.figure()
    plt.xlim(norm.ppf(pfa_min), norm.ppf(pfa_max))

    plt.xticks(norm.ppf(pticks[tmin_fa:tmax_fa]), xlabels[tmin_fa:tmax_fa])
    plt.xlabel('False Alarm probability (in %)')

    # FAR
    plt.ylim(norm.ppf(pmiss_min), norm.ppf(pmiss_max))
    plt.yticks(norm.ppf(pticks[tmin_miss:tmax_miss]), ylabels[tmin_miss:tmax_miss])
    plt.ylabel('Miss probability (in %)')

    return plt

# 计算EER
def compute_EER(frr,far):
    threshold_index = np.argmin(abs(frr - far))  # 平衡点
    eer = (frr[threshold_index]+far[threshold_index])/2
    print("eer=",eer)
    return eer

# 计算minDCF P_miss = frr  P_fa = far
def compute_minDCF2(P_miss,P_fa):
    C_miss = C_fa = 1
    P_true = 0.01
    P_false = 1-P_true

    npts = len(P_miss)
    if npts != len(P_fa):
        print("error,size of Pmiss is not euqal to pfa")
    
    DCF = C_miss * P_miss * P_true + C_fa * P_fa*P_false

    min_DCF = min(DCF)

    print("min_DCF_2=",min_DCF)

    return min_DCF


# 计算minDCF P_miss = frr  P_fa = far
def compute_minDCF3(P_miss,P_fa,min_DCF_2):
    C_miss = C_fa = 1
    P_true = 0.001
    P_false = 1-P_true

    npts = len(P_miss)
    if npts != len(P_fa):
        print("error,size of Pmiss is not euqal to pfa")
    
    DCF = C_miss * P_miss * P_true + C_fa * P_fa*P_false

    # 该操作是我自己加的，因为论文中的DCF10-3指标均大于DCF10-2且高于0.1以上，所以通过这个来过滤一下,错误请指正
    min_DCF = 1
    for dcf in DCF:
        if dcf > min_DCF_2+0.1 and dcf < min_DCF:
            min_DCF = dcf

    print("min_DCF_3=",min_DCF)
    return min_DCF

def compute_frr_far(y_true,y_pre):
    # 计算FAR和FRR
    fpr, tpr, thres = roc_curve(y_true, y_pre,pos_label=1)
    print(fpr.shape)
    frr = 1 - tpr
    far = fpr
    frr[frr <= 0] = 1e-5
    far[far <= 0] = 1e-5
    frr[frr >= 1] = 1-1e-5
    far[far >= 1] = 1-1e-5
    return frr,far


if __name__ == "__main__":
    # 读文件获取y_true和y_score  baseline model
    y_true = np.load('./dataset/y_true.npy')
    y_score = np.load('./dataset/y_pre.npy')
    # proposed model
    y_score1 = np.load('./dataset/p_y_pre.npy')
    y_true1 = np.load('./dataset/p_y_true.npy')
    print(y_score[:20])
    print(y_score1[:20])
    
    b_frr,b_far = compute_frr_far(y_true,y_score)
    p_frr,p_far = compute_frr_far(y_true1,y_score1)

    # p_frr = np.load('./dataset/p_frr.npy')
    # p_far = np.load('./dataset/p_far.npy')
    # print(p_frr.shape)

    # 画图
    plt = plot_DET_curve()
    x, y = norm.ppf(b_frr), norm.ppf(b_far)
    plt.plot(x, y,label='baseline model')
    x1,y1 = norm.ppf(p_frr),norm.ppf(p_far)
    plt.plot(x1,y1,label='proposed model')
    # plt.plot([-40, 1], [-40, 1])
    plt.legend(fontsize=12)
    plt.xlabel('False Alarm probability(in %)', fontsize=12)
    plt.ylabel('Miss probability', fontsize=12)
    plt.show()

    
    
    eer = compute_EER(b_frr,b_far)

    min_DCF_2 = compute_minDCF2(b_frr*100,b_far*100)

    min_DCF_3 = compute_minDCF3(b_frr*100,b_far*100,min_DCF_2)

    eer = compute_EER(p_frr,p_far)

    min_DCF_2 = compute_minDCF2(p_frr*100,p_far*100)

    min_DCF_3 = compute_minDCF3(p_frr*100,p_far*100,min_DCF_2)
