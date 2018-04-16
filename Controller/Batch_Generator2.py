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
		self.numpy_data_score_1 = np.load(PATH_NUMPY_DATA_SCORE_1)#, mmap_mode='r')
		self.numpy_data_score_2 = np.load(PATH_NUMPY_DATA_SCORE_2)#, mmap_mode='r')
		self.numpy_data_score_3 = np.load(PATH_NUMPY_DATA_SCORE_3)#, mmap_mode='r')
		self.numpy_data_score_4 = np.load(PATH_NUMPY_DATA_SCORE_4)#, mmap_mode='r')
		
		self.numpy_data_score_1_labels = np.load(PATH_NUMPY_DATA_SCORE_1_labels)#, mmap_mode='r')
		self.numpy_data_score_2_labels = np.load(PATH_NUMPY_DATA_SCORE_2_labels)#, mmap_mode='r')
		self.numpy_data_score_3_labels = np.load(PATH_NUMPY_DATA_SCORE_3_labels)#, mmap_mode='r')
		self.numpy_data_score_4_labels = np.load(PATH_NUMPY_DATA_SCORE_4_labels)#, mmap_mode='r')


	def read_CSVs(self):
		self.CSV_TRAIN = pd.read_csv(PATH_CSV_TRAIN)
		self.CSV_TEST = pd.read_csv(PATH_CSV_TEST)
		self.CSV_VALIDATION_PHASE = pd.read_csv(PATH_CSV_VALIDATION_DATASET)
	

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

		random_seed = np.random.randint(0, 10000)
		np.random.seed(seed=random_seed)
		self.CSV_VALIDATION_PHASE = self.CSV_VALIDATION_PHASE.reindex(np.random.permutation(self.CSV_VALIDATION_PHASE.index))
		self.CSV_VALIDATION_PHASE = self.CSV_VALIDATION_PHASE.reset_index()


	def get_inp_image(self, path, channel=-1):
		path = os.path.join(PATH_ROOT, '{}/Perio_Frame/{}.png'.format(path.split('_')[0], path.split('_')[-1]))
		im = np.array(cv2.imread(path))
		im = cv2.GaussianBlur(im, (7, 7), 1.0)
		if channel == -1: # 3 channel
			return im
		else:
			return im[:, :, channel]

	def get_validation_image(self, _path, channel=-1):
		print(_path)
		path = os.path.join(PATH_VALIDATION_DATASET, '{}_SC_1_slow/red/{}.png'.format(_path.split('_')[0], _path.split('_')[-1]))
		im = np.array(cv2.imread(path))
		im = cv2.GaussianBlur(im, (7, 7), 1.0)
		if channel == -1: # 3 channel
			return im, os.path.join(PATH_VALIDATION_DATASET, '{}_SC_1_slow'.format(_path.split('_')[0])), '{}_pred.png'.format(_path.split('_')[-1])
		else:
			return im[:, :, channel], os.path.join(PATH_VALIDATION_DATASET, '{}_SC_1_slow'.format(_path.split('_')[0])), '{}_pred.png'.format(_path.split('_')[-1])


	def get_ground_truth_image(self, path):
		path = os.path.join(PATH_ROOT, '{}/masks2/{}.png'.format(path.split('_')[0], path.split('_')[-1]))
		return np.array(cv2.imread(path, 0))


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
		
		batch_Y = np.expand_dims(batch_Y, axis=-1).astype(np.float32)
		return batch_X[perm, :, :, :], batch_Y[perm, :, :]


	def get_test_data(self):
		XY = list(self.CSV_TEST['patient_frame'])
		S = list(self.CSV_TEST['score_l'])

		batch_X = []
		batch_Y = []
		#batch_S = []
		
		for j in range(len(XY)):
			_channel = -1 if N_CHANNELS_IN == 3 else 0
			x = self.get_inp_image(XY[j], channel=_channel) # blue channel == 0
			y = self.get_ground_truth_image(XY[j])
			#s = int(S[j])

			batch_X.append(x)
			batch_Y.append(y)
			#batch_S.append(s)

		
		if N_CHANNELS_IN == 3:
			batch_X = np.array(batch_X).astype(np.float32)
		else:
			batch_X = np.expand_dims(np.array(batch_X).astype(np.float32), -1)
		batch_Y = np.expand_dims(np.array(batch_Y).astype(np.float32), -1)
		#batch_S = np.array(batch_S).astype(np.int32)
		# print('{}\t{}\t{}'.format(batch_X.shape, batch_Y.shape, batch_S.shape))
			
		return batch_X, batch_Y#, batch_S


	def get_validation_phase_dataset(self):
		XY = list(self.CSV_VALIDATION_PHASE['patient_frame'])

		batch_X = []
		batch_X_PATH = []
		batch_X_NAMES = []
		
		for j in range(len(XY)):
			_channel = -1 if N_CHANNELS_IN == 3 else 0
			x, path, name = self.get_validation_image(XY[j], channel=_channel) # blue channel == 0

			batch_X.append(x)
			batch_X_PATH.append(path)
			batch_X_NAMES.append(name)
		
		if N_CHANNELS_IN == 3:
			batch_X = np.array(batch_X).astype(np.float32)
		else:
			batch_X = np.expand_dims(np.array(batch_X).astype(np.float32), -1)
		batch_X_PATH = np.array(batch_X_PATH)
		batch_X_NAMES = np.array(batch_X_NAMES)
			
		return batch_X, batch_X_PATH, batch_X_NAMES


if __name__ == '__main__':
	mode = sys.argv[1]
	if mode == 'train_data':
		x, y = BatchGenerator().generate_training_batch()
		print('{}\t{}'.format(x.shape, x.max()))
		print('{}\t{}'.format(y.shape, x.max()))
	if mode == 'test_data':
		test_X, test_Y = BatchGenerator().get_test_data()
		print(test_X.max())
		print(test_Y.max())
	if mode == 'validation_phase_data':
		val_X, paths, names = BatchGenerator().get_validation_phase_dataset()
		print('\n\nOUT\n\n')
		print(paths)
		print('\n\n.\n')
		print(names)
