import cv2, os
import numpy as np


PATH_DATA = '/Users/amanrana/Desktop/predicted_images'
PATH_COLOR_CODED = '/Users/amanrana/Desktop/color_coded_images'

for i in range(1, 68):
	# Reading images
	try:
		x = np.array(cv2.imread(os.path.join(PATH_DATA, '{}_perio.png'.format(i))))
		y = np.array(cv2.imread(os.path.join(PATH_DATA, '{}_gnd_color.png'.format(i))))
		pred = np.array(cv2.imread(os.path.join(PATH_DATA, '{}_pred_color.png'.format(i))))
		color_coded = np.array(cv2.imread(os.path.join(PATH_DATA, '{}_color_coded.png'.format(i))))
	except:
		# print('error')
		continue

	grid = np.zeros([960, 1280, 3])
	grid[:480, :640, :] = x
	grid[:480, 640:, :] = y
	grid[480:, :640, :] = pred
	grid[480:, 640:, :] = color_coded
	cv2.imwrite(os.path.join(PATH_COLOR_CODED, '{}._grid.png'.format(i)), grid)
