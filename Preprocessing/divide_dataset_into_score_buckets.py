import os, shutil
import pandas as pd

PATH_ROOT = '/mas/u/arana/datasets/kumbh'
PATH_DATASET = os.path.join(PATH_ROOT, 'kumbh_labelled')
PATH_DATA = os.path.join(PATH_DATASET, 'kumbh_data')
PATH_CSV = os.path.join(PATH_DATASET, 'LOGSCSV.csv')
PATH_DISTRIBUTED_SCORE = os.path.join(PATH_ROOT, 'by_score')
if not os.path.exists(PATH_DISTRIBUTED_SCORE): os.mkdir(PATH_DISTRIBUTED_SCORE)

csv = pd.read_csv(PATH_CSV)

for row in range(len(csv)):
	row_data = csv.iloc[row, :]
	patient_frame = row_data['patient_frame']
	score = row_data['score_l']
	path_folder_score = os.path.join(PATH_DISTRIBUTED_SCORE, '{}'.format(score))
	if not os.path.exists(path_folder_score): os.mkdir(path_folder_score)
	_id, frame = patient_frame.split('_')
	path_white_orig = os.path.join(PATH_DATA, _id, 'Perio_Frame/{}.png*'.format(frame))
	path_gnd_labels = os.path.join(PATH_DATA, _id, 'masks/{}.png'.format(frame))
	new_path_perio_json = os.path.join(path_folder_score, 'perio_json')
	new_path_mask = os.path.join(path_folder_score, 'masks')
	if not os.path.exists(new_path_perio_json): os.mkdir(new_path_perio_json)
	if not os.path.exists(new_path_mask): os.mkdir(new_path_mask)
	os.system('cp {} {}'.format(path_white_orig, os.path.join(new_path_perio_json, '')))
	os.system('cp {} {}'.format(path_gnd_labels, os.path.join(new_path_mask, '')))

