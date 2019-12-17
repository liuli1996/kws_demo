import pandas as pd
import shutil
import numpy as np
import random
import os
import librosa


def copy_enroll_file():
	csv_file = r'C:/Users/70976/PycharmProjects/kws_end2end/csv_12classes/enroll.csv'

	# cpoy files
	df = pd.read_csv(csv_file)
	for _, row in df.iterrows():
		str_list = row['filename'].split('/')
		old_path = 'E:/DataBase/speech_dataset' + '/' + str_list[-2] + '/' + str_list[-1]
		new_path = 'C:/Users/70976/Desktop/test/enrollset' + '/' + str_list[-2] + '_' + str_list[-1]
		shutil.copy(old_path, new_path)


# data augment
root = 'C:/Users/70976/Desktop/test/enrollset'
noise_root = 'E:/DataBase/noise_file'
file_list = os.listdir(root)
file_path = [os.path.join(root, file) for file in file_list]
noise_list = os.listdir(noise_root)
noise_path = [os.path.join(noise_root, noise) for noise in noise_list]

# add noise (signal)
for file in file_path:
	if os.path.isdir(file):
		continue
	else:
		file = file.replace('\\', '/')
		str_list = file.split('/')
		data = librosa.load(file, sr=16000)[0]
		print(len(data))
		if len(data) < 16000:
			data = np.pad(data, max(0, 16000-len(data)), 'constant')
		else:
			data = data[0:16000]
		noise = librosa.load(random.sample(noise_path, 1)[0], sr=16000)[0]
		data = data + noise[0:16000]
		fname = 'C:/Users/70976/Desktop/test/enrollset/noise/' + str_list[-1]
		print(fname)
		librosa.output.write_wav(fname, data, sr=16000)