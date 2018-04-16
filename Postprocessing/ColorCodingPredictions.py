import numpy as np
import os, cv2


PATH_DATA = '/Users/amanrana/Desktop/predicted_images'

tp = 0
fp = 0
fn = 0

for idx in range(0, 67):
	x = np.array(cv2.imread(os.path.join(PATH_DATA, '{}_perio.png'.format(idx))))
	x2 = np.array(cv2.imread(os.path.join(PATH_DATA, '{}_perio.png'.format(idx))))
	x3 = np.array(cv2.imread(os.path.join(PATH_DATA, '{}_perio.png'.format(idx))))
	y = np.array(cv2.imread(os.path.join(PATH_DATA, '{}_gnd.png'.format(idx)))) / 255
	pred = np.array(cv2.imread(os.path.join(PATH_DATA, '{}_pred.png'.format(idx)))) / 255
	
	# pred_color_r = np.zeros([480, 640])
	# pred_color_g = np.zeros([480, 640])
	# pred_color_b = np.zeros([480, 640])
	# pred_color_g[pred[:, :, 0] == 1] = 204
	# pred_color_b[pred[:, :, 0] == 1] = 51
	# pred_color = np.zeros([480, 640, 3])
	# pred_color[:, :, 0] = pred_color_r
	# pred_color[:, :, 1] = pred_color_g
	# pred_color[:, :, 2] = pred_color_b

	# for i in range(480):
	# 	for j in range(640):
	# 		p1 = y[i, j, 0]
	# 		p2 = pred[i, j, 0]

	# 		r, g, b = x[i, j]

	# 		# False Positive (yellow)
	# 		if (p1 == 0 and p2 == 1):
	# 			r, g, b = 250, 226, 50
	# 			fp += 1

	# 		# False Negative (red)
	# 		if (p1 == 1 and p2 == 0):
	# 			r, g, b = 255, 100, 78
	# 			fn += 1

	# 		# True Positive (Blue)
	# 		if (p1 == 1 and p2 == 1):
	# 			r, g, b, = 0, 162, 255
	# 			tp += 1

	# 		x[i, j, 0] = b
	# 		x[i, j, 1] = g
	# 		x[i, j, 2] = r

	x2_r = x2[:, :, 0]
	x2_g = x2[:, :, 1]
	x2_b = x2[:, :, 2]
	x2_r[pred[:, :, 0] == 1] = 0
	x2_g[pred[:, :, 0] == 1] = 204
	x2_b[pred[:, :, 0] == 1] = 51
	x2_color = np.zeros([480, 640, 3])
	x2_color[:, :, 0] = x2_r
	x2_color[:, :, 1] = x2_g
	x2_color[:, :, 2] = x2_b

	# x3_r = x3[:, :, 0]
	# x3_g = x3[:, :, 1]
	# x3_b = x3[:, :, 2]
	# x3_r[y[:, :, 0] == 1] = 255
	# x3_g[y[:, :, 0] == 1] = 204
	# x3_b[y[:, :, 0] == 1] = 102
	# x3_color = np.zeros([480, 640, 3])
	# x3_color[:, :, 0] = x3_b
	# x3_color[:, :, 1] = x3_g
	# x3_color[:, :, 2] = x3_r


	# cv2.imwrite(os.path.join(PATH_DATA, '{}_color_coded.png'.format(idx)), x)
	cv2.imwrite(os.path.join(PATH_DATA, '{}_pred_color.png'.format(idx)), x2_color)
	# cv2.imwrite(os.path.join(PATH_DATA, '{}_gnd_color.png'.format(idx)), x3_color)

# print('Mean Intersection over Union for all test images: {}'.format(tp/float(tp+fn+fp)))