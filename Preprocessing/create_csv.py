import os, glob
import json
import pandas as pd

PATH_DATASET = '/mas/u/arana/datasets/kumbh/kumbh_labelled/kumbh_data'
patient_ids = os.listdir(PATH_DATASET)
csv_file_name = 'kumbh_metadata.csv'

log = []

for i, patient in enumerate(patient_ids):
	path_data = os.path.join(PATH_DATASET, patient, 'Perio_Frame')
	path_to_gnd_img = os.path.join(PATH_DATASET, patient, 'masks')

	img_paths = glob.glob('{}/*.png'.format(path_data))

	for j, img_path in enumerate(img_paths):
		_id = img_path.split('/')[-3]
		img_name = img_path.split('/')[-1]

		path_x = img_path
		path_y = glob.glob('{}/{}'.format(path_to_gnd_img, img_name))[0]
		json_path = glob.glob('{}/{}.liveuser.perio.1_annotated.json'.format(path_data, img_name))[0]

		with open(json_path) as f:
			json_data = json.load(f)

		keyz = list(json_data.keys())

		try:
			_score = int(json_data['overall_label'][-1])
			log.append({'inp_img_path': path_x, 'ground_img_path': path_y, 'JSON_path': json_path, 'score': _score})
		except:
			print('No score provided for patirnt id: {}, frame: {}'.format(_id, img_name))


df = pd.DataFrame(log, columns=['inp_img_path', 'ground_img_path', 'JSON_path', 'score'])
df.to_csv(csv_file_name)
