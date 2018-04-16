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


	def read_CSVs(self):
		self.CSV_TRAIN = pd.read_csv(PATH_CSV_TRAIN)
		self.CSV_TEST = pd.read_csv(PATH_CSV_TEST)
	

	def shuffle_CSVs(self):
		''' Shuffle the CSV using the random_seed '''
		random_seed = np.random.randint(0, 10000)
		np.random.seed(seed=random_seed)
		self.CSV_TRAIN = self.CSV_TRAIN.reindex(np.random.permutation(self.CSV_TRAIN.index))
		self.CSV_TRAIN = self.CSV_TRAIN.reset_index()
		self.CSV_TEST = self.CSV_TEST.reindex(np.random.permutation(self.CSV_TEST.index))
		self.CSV_TEST = self.CSV_TEST.reset_index()

	def store_data_by_score(self):
		score_1 = self.CSV_TRAIN[self.CSV_TRAIN['score_l'] == 1]
		del score_1['index']
		del score_1['Unnamed: 0']	
		
		score_2 = self.CSV_TRAIN[self.CSV_TRAIN['score_l'] == 2]
		del score_2['index']
		del score_2['Unnamed: 0']	
		
		score_3 = self.CSV_TRAIN[self.CSV_TRAIN['score_l'] == 3]
		del score_3['index']
		del score_3['Unnamed: 0']	
		
		score_4 = self.CSV_TRAIN[self.CSV_TRAIN['score_l'] == 4]
		del score_4['index']
		del score_4['Unnamed: 0']	
		
		score_1_data = []
		score_1_masks = []
		for item in score_1['patient_frame']:
			im = self.get_inp_image(str(item))
			mask = self.get_ground_truth_image(str(item))
			score_1_data.append(im)
			score_1_masks.append(mask)
		
		score_2_data = []
		score_2_masks = []
		for item in score_2['patient_frame']:
			im = self.get_inp_image(str(item))
			mask = self.get_ground_truth_image(str(item))
			score_2_data.append(im)
			score_2_masks.append(mask)
		
		score_3_data = []
		score_3_masks = []
		for item in score_3['patient_frame']:
			im = self.get_inp_image(str(item))
			mask = self.get_ground_truth_image(str(item))
			score_3_data.append(im)
			score_3_masks.append(mask)
		
		score_4_data = []
		score_4_masks = []
		for item in score_4['patient_frame']:
			im = self.get_inp_image(str(item))
			mask = self.get_ground_truth_image(str(item))
			score_4_data.append(im)
			score_4_masks.append(mask)
		
		# creating numpy array for all
		score_1_data = np.array(score_1_data)
		score_1_masks = np.array(score_1_masks)
		score_2_data = np.array(score_2_data)
		score_2_masks = np.array(score_2_masks)
		score_3_data = np.array(score_3_data)
		score_3_masks = np.array(score_3_masks)
		score_4_data = np.array(score_4_data)
		score_4_masks = np.array(score_4_masks)
		
		# Storing the numpy arrays to disk for faster retreival afterwards instread of image by image	
		np.save(os.path.join(PATH_DATA_SCORE_1, 'im.npy'), score_1_data)
		np.save(os.path.join(PATH_DATA_SCORE_2, 'im.npy'), score_2_data)
		np.save(os.path.join(PATH_DATA_SCORE_3, 'im.npy'), score_3_data)
		np.save(os.path.join(PATH_DATA_SCORE_4, 'im.npy'), score_4_data)
		
		np.save(os.path.join(PATH_DATA_SCORE_1, 'labels.npy'), score_1_masks)
		np.save(os.path.join(PATH_DATA_SCORE_2, 'labels.npy'), score_2_masks)
		np.save(os.path.join(PATH_DATA_SCORE_3, 'labels.npy'), score_3_masks)
		np.save(os.path.join(PATH_DATA_SCORE_4, 'labels.npy'), score_4_masks)

	
	def convert_img_to_one_hot_encoded(self, im):
		h, w = im.shape
		one_hot = np.zeros([h, w, 2])
		one_hot_p = np.zeros([h, w])
		one_hot_n = np.zeros([h, w])
		one_hot_p[im == 255] = 1
		one_hot_n[im == 255] = 1
		one_hot[:, :, 0] = one_hot_p
		one_hot[:, :, 1] = one_hot_n
		return np.array(one_hot).astype(np.float32)		


	def get_inp_image(self, data, channel=-1):
		path = os.path.join(PATH_ROOT, '{}/Perio_Frame/{}.png'.format(data.split('_')[0], data.split('_')[-1]))
		im = np.array(cv2.imread(path))
		im = cv2.GaussianBlur(im, (7, 7), 1.0)
		if channel == -1: # 3 channel
			return im
		else:
			return im[:, :, channel]
	
	
	def get_ground_truth_image(self, data):
		path = os.path.join(PATH_ROOT, '{}/masks2/{}.png'.format(data.split('_')[0], data.split('_')[-1]))
		return np.array(cv2.imread(path, 0))
	

	def batch_generator(self):
		for i in range(0, len(self.CSV_TRAIN), TRAIN_BATCH_SIZE):
			batch_csv = self.CSV_TRAIN[i: i+TRAIN_BATCH_SIZE]
			batch_csv = self.CSV_TRAIN.reset_index()

			XY = batch_csv['patient_frame']
			S = batch_csv['score_l']

			batch_X = []
			batch_Y = []
			batch_S = []
			
			try:
				for j in range(TRAIN_BATCH_SIZE):
					_channel = -1 if N_CHANNELS_IN == 3 else 0
					x = self.get_inp_image(XY[j], channel=_channel) # blue channel == 0
					y = self.get_ground_truth_image(XY[j])
					s = int(S[j])

					batch_X.append(x)
					batch_Y.append(y)
					batch_S.append(s)

				if N_CHANNELS_IN == 3:
					batch_X = np.array(batch_X).astype(np.float32)
				else:
					batch_X = np.expand_dims(np.array(batch_X).astype(np.float32), -1)
				batch_Y = np.expand_dims(np.array(batch_Y).astype(np.float32), -1)
				batch_S = np.array(batch_S).astype(np.int32)
				# print('{}\t{}\t{}'.format(batch_X.shape, batch_Y.shape, batch_S.shape))
				
				yield batch_X, batch_Y, batch_S

			except:
				print('Error while generating batch !! Check')
				return


	def get_test_dataset(self):
		X = []
		Y = []
		S = []
		
		for i in range(len(self.CSV_TEST)):
			XY = self.CSV_TEST['patient_frame'][i]
			_S = self.CSV_TEST['score_l'][i]
						
			try:
				_channel = -1 if N_CHANNELS_IN == 3 else 0
				x = self.get_inp_image(XY, channel=_channel)
				y = self.get_ground_truth_image(XY)
				s = int(_S)
				
				X.append(x)
				Y.append(y)
				S.append(s)
	
			except:
				print('Error!! Check')
				return
				
		if N_CHANNELS_IN == 3:
			X = np.array(X).astype(np.float32)
		else:
			X = np.expand_dims(np.array(X).astype(np.float32), -1)
		Y = np.expand_dims(np.array(Y).astype(np.float32), -1)
		S = np.array(S).astype(np.int32)
		
		return X, Y, S
				


if __name__ == '__main__':
	gen = BatchGenerator()
	gen.store_data_by_score()
	'''
	print('-'*100)
	a, b, c = gen.next()
	print(a.max())
	print(b.max())
	print(c.max())
	
	a = np.load(os.path.join(PATH_DATA_SCORE_1, '1.npy'))
	print(a.shape)
	b = np.load(os.path.join(PATH_DATA_SCORE_2, '2.npy'))
	print(b.shape)
	c = np.load(os.path.join(PATH_DATA_SCORE_3, '3.npy'))
	print(c.shape)
	d = np.load(os.path.join(PATH_DATA_SCORE_4, '4.npy'))
	print(d.shape)
	'''
