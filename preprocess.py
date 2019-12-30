""" 
@ author: Qmh
@ file_name: preprocess.py
@ time: 2019:10:29:17:08
""" 

import pandas as pd
import constants as c
import numpy as np 
import librosa
import tqdm 
import os
import pickle
from python_speech_features import logfbank
import python_speech_features as psf
import glob
import constants as c
import matplotlib.pyplot as plt
import librosa.display
import math
import logging


class SilenceDetector(object):
    def __init__(self, threshold=20, bits_per_sample=16):
        self.cur_SPL = 0
        self.threshold = threshold
        self.bits_per_sample = bits_per_sample
        self.normal = pow(2.0, bits_per_sample - 1)
        self.logger = logging.getLogger('balloon_thrift')

    def is_silence(self, chunk):
        self.cur_SPL = self.soundPressureLevel(chunk)
        is_sil = self.cur_SPL < self.threshold
        # print('cur spl=%f' % self.cur_SPL)
        if is_sil:
            self.logger.debug('cur spl=%f' % self.cur_SPL)
        return is_sil
    
    def soundPressureLevel(self, chunk):
        value = math.pow(self.localEnergy(chunk), 0.5)
        value = value / len(chunk) + 1e-12
        value = 20.0 * math.log(value, 10)
        return value

    def localEnergy(self, chunk):
        power = 0.0
        for i in range(len(chunk)):
            sample = chunk[i] * self.normal
            power += sample*sample
        return power
    

# remove VAD
def VAD_audio(wav,sr,threshold = 15):
    sil_detector = SilenceDetector(threshold)
    new_wav = []
    if sr != 16000:
        wav = librosa.resample(wav, sr, 16000)
        sr = 16000
    for i in range(int(len(wav)/(sr*0.02))):
        start = int(i*sr*0.02)
        end = start + int(sr*0.02)
        is_silence = sil_detector.is_silence(wav[start:end])
        if not is_silence:
            new_wav.extend(wav[start:end])
    return new_wav

# 裁减音频
def cut_audio(src):
    samples = src.shape[0]
    n_samples = int(c.DURA * c.SR)
    print("samples=",samples)
    print("n_samples=",n_samples)
    # 裁减音频
    if samples < n_samples:
        src = np.hstack((src,np.zeros((n_samples-samples,))))
    elif samples > n_samples:
        src = src[(samples-n_samples)//2:(samples+n_samples)//2]
    return src

# compute melgram
def get_melgram(audio_path):
    src = cut_audio(audio_path)
    logam = librosa.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    # shape=(96,1366)
    ret = logam(melgram(y=src,sr=c.SR,n_fft=c.N_FFT,hop_length=c.HOP_LEN,n_mels=c.MEL_BIN))
    return ret

# 绘制音频
def draw_audio(y,sr):
    plt.figure()
    librosa.display.waveplot(y,sr)
    
# extract fbank feature
def extract_feature(audio_path):
    signal,sr = librosa.load(audio_path,sr=c.SR)
    # 去静音
    signal = np.array(VAD_audio(signal.flatten(),c.SR,15))
    # 填充音频
    signal = cut_audio(signal)    
    # extract fbank feature
    feat = psf.logfbank(signal,c.SR,nfilt=c.N_FILT)
    return feat


# 裁减音频
def cut_audio_for_test(src):
    samples = src.shape[0]
    n_samples = int(c.DURA * c.SR)
    src_list = []
    # 填充音频
    if samples < n_samples:
        src_list.append(np.hstack((src,np.zeros((n_samples-samples,)))))
    # 将音频分段
    segments = samples//n_samples

    if samples % n_samples != 0:
        segments = segments + 1

    for i in range(segments):
        # 填充最后一段
        if (i+1) * n_samples > samples:
            sub_src = src[i*n_samples:]
            sub_src = np.hstack((sub_src,np.zeros((n_samples-len(sub_src),))))
        else:
            sub_src = src[i*n_samples:(i+1)*n_samples]
        src_list.append(sub_src)

    return src_list


def extract_feature_for_test(audio_path):
    signal,sr = librosa.load(audio_path,sr=c.SR)
    # 去静音
    signal = np.array(VAD_audio(signal.flatten(),c.SR,15))
    # 填充音频
    signal_list = cut_audio_for_test(signal)    
    feat = []
    for signal in signal_list:
        # extract fbank feature
        feat.append(psf.logfbank(signal,c.SR,nfilt=c.N_FILT))
    feat = np.array(feat)  # (segments,299,64,3)
    return feat


def save_to_pickle(pickle_name,melgram):
    save_dict = {}
    save_dict['LogMel_Features'] = melgram
    with open(pickle_name,'wb') as f:
        pickle.dump(save_dict,f,protocol=-1)

# 获取数据集
def Create_Dataset(dataset_path,speaker_number=None):
    audio_dataset = os.path.join(dataset_path,'wav')
    speaker_list = os.listdir(audio_dataset)
    speaker_list.sort()
    if speaker_number:
        speaker_list = speaker_list[400:600]
    path_list = []
    label_list = []
    for sp in speaker_list:
        for x in glob.iglob(os.path.join(audio_dataset,sp)+"/*/*.wav"):
            path_list.append(x)
            label_list.append(sp)
    return (path_list,label_list)


## 预处理函数
def preprocess(dataset,save_dir,is_test=False):
    audio_paths,labels = dataset
    # 写入pickle文件
    for i in tqdm.tqdm(range(len(audio_paths))):
        try:
            # 提取特征
            if not is_test:
                # pickle_name = "_".join(audio_paths[i].split("/")[-3:]).replace("wav",'pickle')
                # pickle_path = os.path.join(save_dir,pickle_name)
                # feat = extract_feature(audio_paths[i])
                # save_to_pickle(pickle_path,feat)
                feats = extract_feature_for_test(audio_paths[i])
                for index,feat in enumerate(feats):
                    audio_name_list = audio_paths[i].split("/")[-3:]
                    audio_name_list.insert(1,str(index))
                    pickle_name = "_".join(audio_name_list).replace("wav",'pickle')
                    pickle_path = os.path.join(save_dir,pickle_name)
                    save_to_pickle(pickle_path,feat)
            else:
                pickle_name = "_".join(audio_paths[i].split("/")[-3:]).replace("wav",'pickle')
                pickle_path = os.path.join(save_dir,pickle_name)
                feat = extract_feature_for_test(audio_paths[i])
                save_to_pickle(pickle_path,feat)
        except Exception as e:
            print(e)
            exit()

    

def main(speaker_number=200,is_test=False):
    if not is_test:
        dataset = Create_Dataset(c.TRAIN_DEV_SET,speaker_number)
        save_dir = os.path.join(c.TRAIN_DEV_SET,c.TRAIN_PICKLE_NAME)
    else:
        dataset = Create_Dataset(c.TEST_SET)
        save_dir = os.path.join(c.TEST_SET,c.TEST_PICKLE_NAME)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    preprocess(dataset,save_dir=save_dir,is_test=is_test)

   


if __name__ == "__main__":
    speaker_number = 400
    is_test = False
    main(speaker_number,is_test)
    # wav_path = '/home/dsp/Documents/wav_data/SpeakerRecognition_dataset/voxceleb/data/test/wav/id10270/5sJomL_D0_g/00002.wav'
    # feat = extract_feature(wav_path)
   