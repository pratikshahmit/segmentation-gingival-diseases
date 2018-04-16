import os, glob
import json
import pandas as pd

PATH_DATASET = '/Users/amanrana/Desktop/kumbh_data'

patient_ids = os.listdir(PATH_DATASET)
rectangles_info_per_patient = {}
total_rect = 0
no_score_count_l = 0
no_score_count_r = 0

score_stat_l = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
score_stat_r = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
file = open("/Users/amanrana/Desktop/no_scores_validation.txt", "w")
log = []

for i, patient in enumerate(patient_ids):
	path_patient_data = os.path.join(PATH_DATASET, patient, 'Perio_Frame')
	imgs = [n.split('/')[-1] for n in glob.glob('{}/*.png'.format(path_patient_data))]
	lawrence_jsons_paths = glob.glob('{}/*.1_annotated.json'.format(path_patient_data))
	roma_json_paths = glob.glob('{}/*.5_annotated.json'.format(path_patient_data))
	for j, json_path in enumerate(lawrence_jsons_paths):
		with open(json_path) as f:
			_json_l = json.load(f)

		with open(roma_json_paths[j]) as f2:
			_json_r = json.load(f2)

		keyz = list(_json_l.keys())
		id = json_path.split('/')[-1][:-36]
		try:
			_score_l = int(_json_l['overall_label'][-1])
			_score_r = int(_json_r['overall_label'][-1])
		 	score_stat_l[_score_l] = score_stat_l[_score_l] + 1
		 	score_stat_r[_score_r] = score_stat_r[_score_r] + 1
		except:
			print('No score provided for {}'.format(json_path))
			no_score_count_l+= 1
			# file.write('{}\t{}\t\t{}\n'.format(no_score_count_l, patient, id))
			no_score_count_r += 1

		log.append({'patient_frame': '{}_{}'.format(patient, id), 'score_l': _score_l, 'score_r': _score_r})
		
		if 'overall_label' in keyz: keyz.remove('overall_label')

		patient_rect = 0

		for item in keyz:
			num_rectangles_for_item = len(_json_l[item]['rectangles'])
			total_rect += num_rectangles_for_item
			patient_rect += num_rectangles_for_item
		
		rectangles_info_per_patient[patient] =  patient_rect

# file.close()
			
print('Score Stats:\n{}\n'.format(score_stat_l))
print('No score count: {}\n'.format(no_score_count_l))
print('Total number of rectangles: {}\n'.format(total_rect))
df = pd.DataFrame(log)
df.to_csv('/Users/amanrana/Desktop/LOGSCSV.csv')
