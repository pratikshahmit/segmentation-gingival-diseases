import os, sys, cv2
os.system('clear')
import tensorflow as tf
import numpy as np
from datetime import datetime
sys.path.append('../')
from base_info import *
from Batch_Generator2 import BatchGenerator
from Models.CNN_AutoEncoder_01 import *
os.environ["CUDA_VISIBLE_DEVICES"]= '0'


class Controller:
	def __init__(self):
		pass
		

	def create_save_folder(self):
		d = datetime.now()
		date = '{}_{}_{}'.format(d.month, d.day, d.year)
		time = '{}_{}'.format(d.hour, d.minute)

		# create the date folder; check if it exists
		date_dir = os.path.join(PATH_LOGS, date)
		if not os.path.exists(date_dir): os.mkdir(date_dir)
		self.save_dir = os.path.join(date_dir, time)
		if not os.path.exists(self.save_dir): os.mkdir(self.save_dir)
		self.save_dir = os.path.join(self.save_dir, '')		


	def train(self):
		self.create_save_folder()
		gen = BatchGenerator()
		with tf.name_scope('Placeholders'):
			 # Placeholders
			x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS_IN], name='X')
			y_ae = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS_OUT], name='Y_AE')	
			LR = tf.placeholder(tf.float32, name='Learning_Rate')

		# Adding Image summaries
		tf.summary.image('Input_Image', x, max_outputs=1)
		tf.summary.image('IMAGE_Ground_Truth', y_ae, max_outputs=1)

		with tf.name_scope('Model'):
			# Output from the network
			segmented_img_out = CNN_AutoEncoder_01(verbose=True).network(x)
		tf.summary.image('Nework_Output', tf.multiply(segmented_img_out, 255.0), max_outputs=1)

		_y_ae = tf.divide(y_ae, 255.)
		
		with tf.name_scope('Loss'):
			# Loss function only during Training and Evaluation	
			epsilon = 1e-7	
			numerator = tf.reduce_sum(tf.multiply(_y_ae, segmented_img_out), axis=[1, 2, 3])
			denominator = tf.reduce_sum(tf.abs(_y_ae) + tf.abs(segmented_img_out), axis=[1, 2, 3]) + epsilon
			
			loss_ae_3_channel = 1.0 - tf.reduce_mean(tf.divide(numerator, denominator))			
			loss_ae_blue_channel = -tf.reduce_sum(tf.multiply(_y_ae, tf.log(segmented_img_out + epsilon)))	
			
			if N_CHANNELS_IN == 3:
				loss = loss_ae_3_channel
			elif N_CHANNELS_IN == 1:
				loss = loss_ae_blue_channel		
		tf.summary.scalar('Loss_Total', loss)

		with tf.name_scope('ROC'):
			tp, tp_update_op = tf.metrics.true_positives(labels=_y_ae, predictions=segmented_img_out, name='True_Positive')
			fp, fp_update_op = tf.metrics.false_positives(labels=_y_ae, predictions=segmented_img_out, name='False_Positives')
			fn, fn_update_op = tf.metrics.false_negatives(labels=_y_ae, predictions=segmented_img_out, name='False_Negatives')
			tn, tn_update_op = tf.metrics.true_positives(labels=1.-_y_ae, predictions=1.-segmented_img_out, name='True_Negatives')
			auc, auc_update_op = tf.metrics.auc(labels=_y_ae, predictions=segmented_img_out, num_thresholds=20, name='AUC')
		tf.summary.scalar('AUC', auc_update_op)	

		with tf.name_scope('Optimizer'):
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				optimizer = tf.train.AdamOptimizer(learning_rate=LR)
				train_op = optimizer.minimize(loss=loss)
		
		with tf.name_scope('Accuracy'):
			accuracy_ae, accuracy_update_op = tf.metrics.accuracy(labels=segmented_img_out, predictions=_y_ae)

		saver = tf.train.Saver()

		## Starting tf.Session()
		with tf.Session() as sess:
			init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())	
			sess.run(init)

			merged = tf.summary.merge_all()
			log_summary = tf.summary.FileWriter(self.save_dir, sess.graph)

			lr = LEARNING_RATE
			
			for iteration in range(N_ITERATIONS):
				try:
					batch_x, batch_y = gen.generate_training_batch()
				except:
					print('error')
					continue
				
				if iteration > 2 and iteration%500 == 0:
					lr = lr / 5.
						
				# Training
				_summary, _loss, _auc, _, _tp, _tn, _fp, _fn = sess.run([merged, loss, auc_update_op, train_op, tp_update_op, tn_update_op, fp_update_op, fn_update_op], feed_dict={x: batch_x, y_ae: batch_y, LR: lr})
				#_summary, _loss, _auc = sess.run([merged, loss, auc_update_op], feed_dict={x: batch_x, y_ae: batch_y, LR: lr})
				log_summary.add_summary(_summary, iteration)
				print('Iteration: {}\tLR: {} Loss: {}\tAUC: {}Precision: {}\tRecall: {}\tF1 Score: {}'.format(iteration, lr, _loss, _auc, _tp/(_tp+_fp), _tp/(_tp+_fn), (2*_tp)/((2*_tp)+_fp+_fn)))
				#print('Iteration: {}\tLR: {} Loss: {}\tAUC: {}'.format(iteration, lr, _loss, _auc))
			
			# Saving trained model
			saver.save(sess, self.save_dir)
	

	## Inference
	def inference(self):
		test_X, test_X_PATH, test_X_NAMES = BatchGenerator().get_validation_phase_dataset()
		
		with tf.Session() as sess:	
			with tf.name_scope('Placeholders'):
				 # Placeholders
				x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS_IN], name='X')
				# y_ae = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS_OUT], name='Y_AE')
			
			with tf.name_scope('Model'):
				# Output from the network
				segmented_img_out = CNN_AutoEncoder_01(verbose=True).network(x)
			
			print('Loading model')
			saver = tf.train.Saver()
			saver.restore(sess, '/mas/u/arana/LOGS/8_28_2017/14_55/')	
			print('Model loaded successfully')	
			
			print('Making predictions using trained model')
			predictions_test = sess.run(segmented_img_out, feed_dict={x: test_X})#, y_ae: test_Y})
			predictions_test = np.array(predictions_test).astype(np.int16)
			print('Storing results')		

			for i in range(predictions_test.shape[0]):
				img = 255.0 * predictions_test[i, :, :, :]
				write_folder = os.path.join(test_X_PATH[i], 'predicted')   #PATH_PREDICTED_DATA_SAVE, '{}_perio.png'.format(i)), test_X[i, :, :, :])
				print(write_folder)
				if not os.path.exists(write_folder): os.mkdir(write_folder)
				cv2.imwrite(os.path.join(write_folder, test_X_NAMES[i]), img)
			print('Predictions saved to {}'.format(PATH_VALIDATION_DATASET))

if __name__ == '__main__':
	#Controller().train()
	Controller().inference()







