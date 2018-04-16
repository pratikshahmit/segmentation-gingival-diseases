import os, sys
sys.path.append('../')
import cv2
import numpy as np
import pandas as pd
from base_info import *


class BatchGenerator:
	def __init__(self):
		self.read_CSV()
		self.shuffle_CSV()


	def read_CSV(self):
		self.CSV = pd.read_csv(PATH_CSV)
	

	def shuffle_CSV(self):
		''' Shuffle the CSV using the random_seed '''
		random_seed = np.random.randint(0, 10000)
		np.random.seed(seed=random_seed)
		self.CSV = self.CSV.reindex(np.random.permutation(self.CSV.index))
		self.CSV = self.CSV.reset_index()


	def get_inp_image(self, path, channel=-1):
		im = np.array(cv2.imread(path))
		im = cv2.GaussianBlur(im, (7, 7), 1.0)
		if channel == -1: # 3 channel
			return im
		else:
			return im[:, :, channel]

	def get_ground_truth_image(self, path):
		return np.array(cv2.imread(path, 0))


	def batch_generator(self):
		for i in range(0, len(self.CSV), TRAIN_BATCH_SIZE):
			batch_csv = self.CSV[i: i+TRAIN_BATCH_SIZE]
			batch_csv = batch_csv.reset_index()

			X = batch_csv['inp_img_path']
			Y = batch_csv['ground_img_path']
			S = batch_csv['score']

			batch_X = []
			batch_Y = []
			batch_S = []
			
			try:
				for j in range(TRAIN_BATCH_SIZE):
					_channel = -1 if N_CHANNELS_IN == 3 else 0
					x = self.get_inp_image(X[j], channel=_channel) # blue channel == 0
					y = self.get_ground_truth_image(Y[j])
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


if __name__ == '__main__':
	gen = BatchGenerator().batch_generator()
	print('-'*100)
	a, b, c = gen.next()
	print(a.max())
	print(b.max())
	print(c.max())
