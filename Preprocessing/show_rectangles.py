import os, cv2
import glob, json
import matplotlib.pyplot as plt
import shutil
import numpy as np



PATH_ROOT = '/Users/amanrana/Documents/perio_frame_validation'
patient_ids = os.listdir(PATH_ROOT)
patient_ids.remove('.DS_Store')


def get_rectangles(path):
	with open(path) as f: data = json.load(f)
	return_var = []
	keyz = list(data.keys())

	## Load rectangles
	try:
		keyz.remove('overall_label')
	except:
		pass

	for key in keyz:
		rectangles = data[key]['rectangles']

		for rectangle in rectangles:
			return_var.append(rectangle)

	return return_var


for count, patient_id in enumerate(patient_ids):
	files = glob.glob(os.path.join(PATH_ROOT, patient_id, 'red/*.png'))
	img_paths = [x.split('/')[-1] for x in files]
	
	for i, img_path in enumerate(img_paths):
		rectangles = get_rectangles('{}{}'.format(files[i], '.liveuser.perio.validation.5_annotated.json'))
		img = cv2.imread(files[i])
		mask = np.zeros(img.shape)
		
		tmp_arr = np.zeros(img.shape)
		x_s, y_s, x_e, y_e = 0, 0, 0, 0
		for rectangle in rectangles:
			# img = cv2.rectangle(img, (rectangle['left'], rectangle['top']), (rectangle['left']+rectangle['width'], rectangle['top']+rectangle['height']), color=(0, 0, 0), thickness=0)
			mask[rectangle['top']: rectangle['top']+rectangle['height'], rectangle['left']: rectangle['left']+rectangle['width']] = 1
		
		if rectangle['top'] < x_s: 
			x_s = rectangle['top']
		if rectangle['left'] < y_s:
			y_s = rectangle['left']
		if rectangle['top']+rectangle['height'] > x_e:
			x_e = rectangle['top']+rectangle['height']
		if rectangle['left']+rectangle['width'] > y_e:
			y_e = rectangle['left']+rectangle['width']		
		tmp_arr[x_s: x_e, y_s: y_e] = 1
		#img[mask == 0] = 0
		#img[mask == 1] = 1
		img[tmp_arr == 0] = 0

		write_dir = os.path.join(PATH_ROOT, patient_id, 'masks_roma')
		# if os.path.exists(write_dir): shutil.rmtree(write_dir)
		if not os.path.exists(write_dir): os.mkdir(write_dir)
		cv2.imwrite(write_dir+'/{}'.format(img_path), img)




