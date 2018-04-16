import os, sys
sys.path.append('../')
import cv2
import numpy as np
import pandas as pd
from base_info import *


class BatchGenerator:
	def __init__(self):
		self.read_CSVs()
		self.shuffle_CSVs()
		self.numpy_data_score_1 = np.load(PATH_NUMPY_DATA_SCORE_1_CH)#, mmap_mode='r')
		self.numpy_data_score_2 = np.load(PATH_NUMPY_DATA_SCORE_2_CH)#, mmap_mode='r')
		self.numpy_data_score_3 = np.load(PATH_NUMPY_DATA_SCORE_3_CH)#, mmap_mode='r')
		self.numpy_data_score_4 = np.load(PATH_NUMPY_DATA_SCORE_4_CH)#, mmap_mode='r')
		
		self.numpy_data_score_1_labels = np.load(PATH_NUMPY_DATA_SCORE_1_labels_CH)#, mmap_mode='r')
		self.numpy_data_score_2_labels = np.load(PATH_NUMPY_DATA_SCORE_2_labels_CH)#, mmap_mode='r')
		self.numpy_data_score_3_labels = np.load(PATH_NUMPY_DATA_SCORE_3_labels_CH)#, mmap_mode='r')
		self.numpy_data_score_4_labels = np.load(PATH_NUMPY_DATA_SCORE_4_labels_CH)#, mmap_mode='r')


	def read_CSVs(self):
		self.CSV_TRAIN = pd.read_csv(PATH_CSV_TRAIN)
		self.CSV_TEST = pd.read_csv(PATH_CSV_TEST)
	

	def shuffle_CSVs(self):
		''' Shuffle the CSV using the random_seed '''
		random_seed = np.random.randint(0, 10000)
		np.random.seed(seed=random_seed)
		self.CSV_TRAIN = self.CSV_TRAIN.reindex(np.random.permutation(self.CSV_TRAIN.index))
		self.CSV_TRAIN = self.CSV_TRAIN.reset_index()
		
		random_seed = np.random.randint(0, 10000)
		np.random.seed(seed=random_seed)
		self.CSV_TEST = self.CSV_TEST.reindex(np.random.permutation(self.CSV_TEST.index))
		self.CSV_TEST = self.CSV_TEST.reset_index()


	def get_inp_image(self, path, channel=-1):
		path = os.path.join(PATH_ROOT, '{}/Perio_Frame/{}.png'.format(path.split('_')[0], path.split('_')[-1]))
		im = np.array(cv2.imread(path))
		im = cv2.GaussianBlur(im, (7, 7), 1.0)
		if channel == -1: # 3 channel
			return im
		else:
			return im[:, :, channel]


	def get_ground_truth_image(self, path):
		path = os.path.join(PATH_ROOT, '{}/masks2/{}.png'.format(path.split('_')[0], path.split('_')[-1]))
		return self.convert_img_to_one_hot_encoded(np.array(cv2.imread(path, 0)))


	def generate_training_batch(self):
		batch_X = []
		batch_Y = []
		batch_S = []
	
		s1 = np.random.choice([x for x in range(self.numpy_data_score_1.shape[0])], int(TRAIN_BATCH_SIZE/4.0))
		s2 = np.random.choice([x for x in range(self.numpy_data_score_2.shape[0])], int(TRAIN_BATCH_SIZE/4.0))
		s3 = np.random.choice([x for x in range(self.numpy_data_score_3.shape[0])], int(TRAIN_BATCH_SIZE/4.0))
		s4 = np.random.choice([x for x in range(self.numpy_data_score_4.shape[0])], int(TRAIN_BATCH_SIZE/4.0))
		
		s1_batch = self.numpy_data_score_1[s1, :, :, :]
		s2_batch = self.numpy_data_score_2[s2, :, :, :]
		s3_batch = self.numpy_data_score_3[s3, :, :, :]
		s4_batch = self.numpy_data_score_4[s4, :, :, :]
		
		s1_batch_labels = self.numpy_data_score_1_labels[s1, :, :]
		s2_batch_labels = self.numpy_data_score_2_labels[s2, :, :]
		s3_batch_labels = self.numpy_data_score_3_labels[s3, :, :]
		s4_batch_labels = self.numpy_data_score_4_labels[s4, :, :]

		batch_X = np.concatenate([s1_batch, s2_batch, s3_batch, s4_batch])	
		batch_Y = np.concatenate([s1_batch_labels, s2_batch_labels, s3_batch_labels, s4_batch_labels])
		perm = np.random.permutation(batch_X.shape[0])
		
		return batch_X[perm, :, :, :], batch_Y[perm, :, :]

	
	def convert_img_to_one_hot_encoded(self, im):
		h, w = im.shape
		one_hot = np.zeros([h, w, 2])
		one_hot_p = np.zeros([h, w])
		one_hot_n = np.zeros([h, w])
		one_hot_p[im == 255] = 255
		one_hot_n[im == 255] = 255
		one_hot[:, :, 0] = one_hot_p
		one_hot[:, :, 1] = one_hot_n
		return np.array(one_hot).astype(np.float32)		


	def get_test_data(self):
		XY = list(self.CSV_TEST['patient_frame'])
		S = list(self.CSV_TEST['score_l'])

		batch_X = []
		batch_Y = []
		
		for j in range(len(XY)):
			_channel = -1 if N_CHANNELS_IN == 3 else 0
			x = self.get_inp_image(XY[j], channel=_channel) # blue channel == 0
			y = self.convert_img_to_one_hot_encoded(self.get_ground_truth_image(XY[j]))
			batch_X.append(x)
			batch_Y.append(y)
	
		if N_CHANNELS_IN == 3:
			batch_X = np.array(batch_X).astype(np.float32)
		else:
			batch_X = np.expand_dims(np.array(batch_X).astype(np.float32), -1)
	
		batch_Y = np.array(batch_Y).astype(np.float32)
	
		return batch_X, batch_Y


if __name__ == '__main__':
	x, y = BatchGenerator().generate_training_batch()
	print(x.shape, '\t', x.max())
	print(y.shape, '\t', y.max())
	cv2.imwrite('/mas/u/arana/a.png', y[0, :, :, 0])
	cv2.imwrite('/mas/u/arana/b.png', y[0, :, :, 1])
	#test_X, test_Y = BatchGenerator().get_test_data()
	#print(test_X.max())
	#print(test_Y.max())
