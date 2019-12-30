import tensorflow as tf
import numpy as np
import python_speech_features as psf
import random
import itertools
import pickle
import time
import re
import os
import glob
import argparse
import librosa
from preprocess import VAD_audio,cut_audio


class Feeder():
    def __init__(self,hparams,mode):
        self.hparams = hparams
        self.mode = mode
        if self.mode == 'test':
            self.labels,self.wave_pairs = self.read_test_txt()
    
    def standard_normaliztion(self,x_array,epsilon=1e-12):
        return np.array([(x-np.mean(x))/max(np.std(x),epsilon) for x in x_array])

    def load_pickle(self,pk_name):
        try:
            with open(pk_name,'rb') as f:
                load_dict = pickle.load(f)
                x = load_dict["LogMel_Features"]
                # x = self.standard_normaliztion(x)  # 可以不用标准化
                # x2 = psf.delta(x1,1)
                # x3 = psf.delta(x1,2)
                # x1 = x1[:,:,np.newaxis]
                # x2 = x2[:,:,np.newaxis]
                # x3 = x3[:,:,np.newaxis]
                # x = np.concatenate((x1,x2,x3),axis=2)
            return x
        except Exception as e:
            print(e)
            return None
    
    def create_train_batch(self):
        dataset = os.path.join(self.hparams.train_dev_set,'pickle_all')
        spk_list = list(set([os.path.basename(pk).split("_")[0] for pk in glob.iglob(dataset+"/*.pickle")]))
        num_frames = int(self.hparams.segment_length * 100)
        # 从训练集中一次取k个说话人
        spk_batch = random.sample(range(len(spk_list)),k=self.hparams.num_spk_per_batch)
        # spk_batch = random.choices(range(len(spk_list)),k=self.hparams.num_spk_per_batch)
        batch_y = [spk for spk in range(self.hparams.num_spk_per_batch) for i in range(self.hparams.num_utt_per_batch)]
        batch_x = []
        for spk_id in spk_batch:
            speaker_pickle_files = [pickle for pickle in glob.iglob(dataset+"/*.pickle") if re.search(spk_list[spk_id],pickle)]
            num_pickle_per_speaker = len(speaker_pickle_files)
            if num_pickle_per_speaker < self.hparams.num_utt_per_batch:
                print("less than 10 utts")
                print(num_pickle_per_speaker)
                pass
            # 选择10句话
            utt_idx_list = random.sample(range(num_pickle_per_speaker),k=self.hparams.num_utt_per_batch)
            for utt_idx in utt_idx_list:
                utt_pickle = speaker_pickle_files[utt_idx]
                x = self.load_pickle(utt_pickle)
                if x is None:
                    continue
                # random start point for every utterance
                start_idx = random.randrange(0,x.shape[0]-num_frames)
                x = x[start_idx:start_idx+num_frames]
                batch_x.append(x)
        batch_x = np.asarray(batch_x)
        batch_y = np.asarray(batch_y)
        return batch_x,batch_y

    def extract_features(self,audio_path):
        signal,sr = librosa.load(audio_path,sr=self.hparams.sample_rate)
        # 去静音
        signal = np.array(VAD_audio(signal.flatten(),sr,15))
        # 填充音频
        signal = cut_audio(signal)    
        # extract fbank feature
        feat = psf.logfbank(signal,sr,nfilt=self.hparams.spectrogram_scale)
        num_frames = self.hparams.segment_length * 100
        num_overlap_frames = num_frames * self.hparams.overlap_ratio
        total_len = feat.shape[0]
        num_dvectors = int((total_len-num_overlap_frames)//(num_frames-num_overlap_frames))
        dvectors = []
        for dvec_idx in range(num_dvectors):
            start_idx = int((num_frames-num_overlap_frames)*dvec_idx)
            end_idx = int(start_idx+num_frames)
            dvectors.append(feat[start_idx:end_idx,:])
        dvectors = np.asarray(dvectors)
        return dvectors
    

    # 读取官方给的测试文件
    def read_test_txt(self):
        results = []
        with open(self.hparams.test_txt,'r') as f:
            results = f.read().splitlines()
        labels = []
        wave_pairs = []
        for line in results:
            label,A,B = line.split(" ")
            labels.append(label)
            wav_path = os.path.join(self.hparams.test_set,'wav')
            wave_pairs.append(((os.path.join(wav_path,A)),os.path.join(wav_path,B)))
        # reverse
        labels.reverse()
        wave_pairs.reverse()
        return labels,wave_pairs

    
    def create_test_batch(self):
        label = self.labels.pop()   # 后进先出
        wave_pair = self.wave_pairs.pop()
        # print("label=",label)
        # print("wave_pair=",wave_pair)
        wav1_dectors = self.extract_features(wave_pair[0])
        wav2_dectors = self.extract_features(wave_pair[1])
        return wav1_dectors,wav2_dectors,label
       


if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument("--train_dev_set",type=str,default='/home/dsp/Documents/wav_data/SpeakerRecognition_dataset/voxceleb/data/dev/')
    parser.add_argument("--test_set",type=str,default='/home/dsp/Documents/wav_data/SpeakerRecognition_dataset/voxceleb/data/test/')
    parser.add_argument("--segment_length",type=float,default=1.6)
    parser.add_argument("--spectrogram_scale", type=int, default=64)
    parser.add_argument("--overlap_ratio",type=float,default=0.5)
    parser.add_argument("--num_spk_per_batch",type=int,default=5)
    parser.add_argument("--num_utt_per_batch",type=int,default=10)
    parser.add_argument("--test_txt",type=str,default='./dataset/SV_dataset.txt')
    parser.add_argument("--sample_rate",type=int,default=16000)
    args = parser.parse_args()
    # feeder = Feeder(args,"train")
    # batch_x,batch_y = feeder.create_train_batch()
    # print(batch_x.shape)
    # print(batch_y.shape)
    feeder1 = Feeder(args,"test")
    print(len(feeder1.labels))
    # print(feeder1.labels[:5])
    # print(feeder1.wave_pairs[:5])
    wav1_dectors,wav2_dectors,label = feeder1.create_test_batch()
    wav1_dectors,wav2_dectors,label = feeder1.create_test_batch()
    print(len(feeder1.labels))
    print(wav1_dectors.shape)
    print(wav2_dectors.shape)
    print(label)
    
