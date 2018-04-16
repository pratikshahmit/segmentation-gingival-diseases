from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import os, cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ggplot import *
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

PATH_DATA_PRED = '/Users/amanrana/Desktop/predicted_images_validation'
PATH_DATA = '/Users/amanrana/Documents/perio_frame_validation'
PATH_ROCs = '/Users/amanrana/Desktop/ROC_validation'
# X = []
Y = []
PRED = []

for i in range(1, 68):
	# Reading images
	# x = np.array(cv2.imread(os.path.join(PATH_DATA, '{}_perio.png'.format(i)))).flatten()
	# X.append(x)
	y = np.array(cv2.imread(os.path.join(PATH_DATA, '{}_gnd.png'.format(i)))) / 255.0
	y = y.flatten()
	# Y.append(y)
	pred = np.array(cv2.imread(os.path.join(PATH_DATA, '{}_pred.png'.format(i)))) / 255.0
	pred = pred.flatten()
	# PRED.append(pred)
	# print('{}\t{}'.format(y.max(), pred.max()))

	# ROC Curve
	fpr, tpr, _ = metrics.roc_curve(y, pred)
	df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
	g1 = ggplot(df, aes(x='fpr', y='tpr')) + geom_line() + geom_abline(linetype='dashed')
	g1.save(os.path.join(PATH_ROCs, '{}_ROC.png'.format(i)))


# X = np.array(X).flatten()
# Y = np.array(Y).flatten()
# PRED = np.array(PRED).flatten()


## ROC Curve
# fpr, tpr, _ = metrics.roc_curve(Y, PRED)
# df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
# # g1 = ggplot(df, aes(x='fpr', y='tpr')) + geom_line() + geom_abline(linetype='dashed')
# # print(g1)

# auc = metrics.auc(fpr, tpr)
# g2 = ggplot(df, aes(x='fpr', y='tpr')) + geom_line() + geom_abline(linetype='dashed') + ggtitle('ROC Curve w/ AUC={}'.format(str(auc)))
# print(g2)

# average_precision = average_precision_score(Y, PRED)
# precision, recall, _ = precision_recall_curve(Y, PRED)
# plt.step(recall, precision, color='b', alpha=0.2,
#          where='post')
# plt.fill_between(recall, precision, step='post', alpha=0.2,
#                  color='b')

# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.title('Precision-Recall curve')
# plt.show()

# _precision_score = precision_score(Y, PRED)
# _recall_score = recall_score(Y, PRED)
# print('Precision: {}\nRecall Score: {}'.format(_precision_score, _recall_score))






'''
for i in range(1, 68):
# Reading images
try:
	x = np.array(cv2.imread(os.path.join(PATH_DATA, '{}_perio.png'.format(i)))).flatten()
	y = np.array(cv2.imread(os.path.join(PATH_DATA, '{}_gnd.png'.format(i)))) / 255.0
	y = y.flatten()
	pred = np.array(cv2.imread(os.path.join(PATH_DATA, '{}_pred.png'.format(i)))) / 255.0
	pred = pred.flatten()
	# print('{}\t{}'.format(y.max(), pred.max()))
except:
	# print('error')
	continue

fpr, tpr, _ = metrics.roc_curve(y, pred)
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
g = ggplot(df, aes(x='fpr', y='tpr')) + geom_line() + geom_abline(linetype='dashed')
print(g)




thresholds = np.linspace(1, 0, 101)
ROC = np.zeros([101, 2])

# Calculating TP, FP, TN, FN
for index in range(101):
	t = thresholds[index]
	tp = np.logical_and(pred > t, y==1).sum()
	tn = np.logical_and(pred <= t, y==0).sum()
	fp = np.logical_and(pred > t, y==0).sum()
	fn = np.logical_and(pred <= t, y==1).sum()
	
	fpr = (fp*1.0) / (fp + tn)
	tpr = (tp*1.0) / (tp + fn)
	ROC[index, 0] = fpr
	ROC[index, 1] = tpr

	fig = plt.figure(figsize=(6,6))
	plt.plot(ROC[:,0], ROC[:,1], lw=2)
	plt.xlim(-0.1,1.1)
	plt.ylim(-0.1,1.1)
	plt.xlabel('$FPR(t)$')
	plt.ylabel('$TPR(t)$')
	plt.grid()

	AUC = 0.
	for i in range(100):
	    AUC += (ROC[i+1,0]-ROC[i,0]) * (ROC[i+1,1]+ROC[i,1])
	AUC *= 0.5

	plt.title('ROC curve, AUC = %.4f'%AUC)
	plt.show()
'''
