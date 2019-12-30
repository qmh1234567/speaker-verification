# libri
python newpreprocess.py --in_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/Library_speech_100/test-clean/LibriSpeech/test-clean  --pk_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/Library_speech_100/test-clean/LibriSpeech/test-clean/pickle  --data_type libri

# voxceleb
python preprocess.py --in_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/voxceleb/data/dev --pk_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/voxceleb/data/dev/pickle_ref --data_type vox1

python preprocess.py --in_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/voxceleb/data/test --pk_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/voxceleb/data/test/pickle_ref --data_type vox1

# mit
python newpreprocess.py --in_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/TIMIT/lisa/data/timit/raw/TIMIT/TRAIN/wav --pk_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/TIMIT/lisa/data/timit/raw/TIMIT/TRAIN/pickle --data_type mit
python newpreprocess.py --in_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/TIMIT/lisa/data/timit/raw/TIMIT/TEST/wav --pk_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/TIMIT/lisa/data/timit/raw/TIMIT/TEST/pickle --data_type mit

# aishell
python newpreprocess.py --in_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/AIShell/trainset/wav --pk_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/AIShell/trainset/pickle --data_type aishell
python newpreprocess.py --in_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/AIShell/SV_devdataset/wav --pk_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/AIShell/SV_devdataset/pickle --data_type aishell
python newpreprocess.py --in_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/AIShell/SI_devdataset/wav --pk_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/AIShell/SI_devdataset/pickle --data_type aishell



python train.py --in_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/voxceleb/data/dev/pickle_ref --ckpt_dir /home/dsp/qmh/speaker_verification/Text-Independent-Speaker-Verification/checkpoint --mode train
python test.py --in_dir /home/dsp/Documents/wav_data/SpeakerRecognition_dataset/voxceleb/data/test/pickle_ref --ckpt_file ./checkpoint/model.ckpt-43700
