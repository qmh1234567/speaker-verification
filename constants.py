#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# __author__: Qmh
# __file_name__: constants.py
# __time__: 2019:06:27:16:18

# MEL-SPECTROGRAM
SR = 16000
DURA = 3  # 3s
N_FFT = 512

N_MELS = 128  
HOP_LEN = 256
MEL_BIN = 96
N_FILT = 64

# DATASET
TRAIN_DEV_SET = '/home/dsp/Documents/wav_data/SpeakerRecognition_dataset/voxceleb/data/dev/'
TRAIN_PICKLE_NAME = 'pickle_600'

TEST_SET = '/home/dsp/Documents/wav_data/SpeakerRecognition_dataset/voxceleb/data/test/'
TEST_PICKLE_NAME = 'pickle_new'

# MODEL
WEIGHT_DECAY = 0.00001 #0.00001

REDUCTION_RATIO = 8 
BLOCK_NUM = 2
DROPOUT= 0.1
ALPHA = 0.1

# TRAIN
BATCH_SIZE = 64
INPUT_SHPE = (299,64,3)
# INPUT_SHPE = (160,64,3)
TRAIN_AUDIO_NUM = 100
# 每个批次的人数和句子
NUM_SPK_PER_BATCH= 32
NUM_UTT_PER_BATCH = 5
# 每个音频的分段长度
SEGMENT_LENGTH = 1.6
OVER_LAP_RATIO=0.5 # 重叠率

MODEL_DIR ='./checkpoint'
LEARN_RATE = 0.01  # 0.01
LEARN_DECAY = 0.1
MIN_LEARN_RATE = 0.00001
