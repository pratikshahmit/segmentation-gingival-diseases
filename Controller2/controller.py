import os, sys, cv2
os.system('clear')
import tensorflow as tf
import numpy as np
from datetime import datetime
sys.path.append('../')
from base_info import *
from Batch_Generator2 import BatchGenerator
from Models2.CNN_AutoEncoder_01 import *
os.environ["CUDA_VISIBLE_DEVICES"]= '1'

class Controller:

	def __init__(self):
		self.create_save_folder()
		

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


	def run(self):
		gen = BatchGenerator()
		with tf.name_scope('Placeholders'):
			 # Placeholders
			x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS_IN], name='X')
			y_ae = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS_OUT], name='Y_AE')	
			LR = tf.placeholder(tf.float32, name='Learning_Rate')

			# Adding Image summaries
		tf.summary.image('Input Image', x, max_outputs=2)
		tf.summary.image('IMAGE_Ground_Truth_0', tf.expand_dims(tf.expand_dims(y_ae[0, :, :, 0], -1), 0), max_outputs=1)
		tf.summary.image('IMAGE_Ground_Truth_1', tf.expand_dims(tf.expand_dims(y_ae[0, :, :, 1], -1), 0), max_outputs=1)

		with tf.name_scope('Model'):
			# Output from the network
			n_out = CNN_AutoEncoder_01(verbose=True).network(x)
		tf.summary.image('Nework_Output_0', tf.multiply(tf.expand_dims(tf.expand_dims(n_out[0, :, :, 0], -1), 0), 255.), max_outputs=1)
		tf.summary.image('Nework_Output_1', tf.multiply(tf.expand_dims(tf.expand_dims(n_out[0, :, :, 1], -1), 0), 255.), max_outputs=1)

		_y_ae = tf.divide(y_ae, 255.)
		
		with tf.name_scope('Loss'):
			# Loss function only during Training and Evaluation	
			epsilon = 1e-7	
			numerator_p = tf.reduce_sum(tf.multiply(_y_ae[:, :, :, 0], n_out[:, :, :, 0]), axis=[1, 2])
			denominator_p = tf.reduce_sum(tf.abs(_y_ae[:, :, :, 0]) + tf.abs(n_out[:, :, :, 0]), axis=[1, 2]) + epsilon
			numerator_n = tf.reduce_sum(tf.multiply(_y_ae[:, :, :, 1], n_out[:, :, :, 1]), axis=[1, 2])
			denominator_n = tf.reduce_sum(tf.abs(_y_ae[:, :, :, 1]) + tf.abs(n_out[:, :, :, 1]), axis=[1, 2]) + epsilon
			
			loss_ae_3_channel_p = 1.0 - tf.reduce_mean(tf.divide(numerator_p, denominator_p))			
			loss_ae_3_channel_n = 1.0 - tf.reduce_mean(tf.divide(numerator_n, denominator_n))			
			loss_ae_intensity = tf.reduce_mean(tf.square(_y_ae - n_out))	
			loss_ae_3_channel = loss_ae_3_channel_p + loss_ae_3_channel_n 
			
			if N_CHANNELS_IN == 3:
				loss = loss_ae_3_channel
			elif N_CHANNELS_IN == 1:
				loss = loss_ae_blue_channel		
		tf.summary.scalar('Loss_Total', loss)

		#with tf.name_scope('ROC'):
			#tp, tp_update_op = tf.metrics.true_positives(labels=_y_ae, predictions=n_out, name='True_Positive')
			#fp, fp_update_op = tf.metrics.false_positives(labels=_y_ae, predictions=n_out, name='False_Positives')
			#fn, fn_update_op = tf.metrics.false_negatives(labels=_y_ae, predictions=n_out, name='False_Negatives')
			#tn, tn_update_op = tf.metrics.true_positives(labels=1.-_y_ae, predictions=1.-n_out, name='True_Negatives')
			#auc, auc_update_op = tf.metrics.auc(labels=_y_ae, predictions=n_out, num_thresholds=20, name='AUC')

		with tf.name_scope('Optimizer'):
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				optimizer = tf.train.AdamOptimizer(learning_rate=LR)
				train_op = optimizer.minimize(loss=loss)
		
		#with tf.name_scope('Accuracy'):
			#accuracy_ae, accuracy_update_op = tf.metrics.accuracy(labels=n_out, predictions=_y_ae)

		## Starting tf.Session()
		with tf.Session() as sess:
			init = tf.group(tf.global_variables_initializer())#, tf.local_variables_initializer())	
			sess.run(init)

			merged = tf.summary.merge_all()
			log_summary = tf.summary.FileWriter(self.save_dir, sess.graph)

			lr = LEARNING_RATE
			
			for iteration in range(N_ITERATIONS):
				try:
					# Extracting data form batch
					batch_x, batch_y = gen.generate_training_batch()

				except:
					print('error')
					continue
				
				if iteration > 2 and iteration%1000 == 0:
					lr = lr / 5.
						
				# Training
				_summary, _loss = sess.run([merged, loss], feed_dict={x: batch_x, y_ae: batch_y, LR: lr})
				log_summary.add_summary(_summary, iteration)
				print('Iteration: {}\tLR: {} Loss: {}'.format(iteration, lr, _loss))
			
			# Saving trained model
			saver.save(sess, self.save_dir)
	

	## Inference
	def inference(self):
		test_X, test_Y = BatchGenerator().get_test_data()
		
		with tf.Session() as sess:	
			with tf.name_scope('Placeholders'):
				 # Placeholders
				x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS_IN], name='X')
				y_ae = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS_OUT], name='Y_AE')	
			
			with tf.name_scope('Model'):
				# Output from the network
				n_out = CNN_AutoEncoder_01(verbose=True).network(x)
	
			saver = tf.train.Saver()
			saver.restore(sess, PATH_LOGS+'/8_28_2017/14_55/')	

			predictions_test = sess.run(n_out, feed_dict={x: test_X, y_ae: test_Y})
			predictions_test = np.array(predictions_test).astype(np.int16)
			
			for i in range(predictions_test.shape[0]):
				img = 255.0 * predictions_test[i, :, :, :]
				cv2.imwrite(os.path.join(PATH_PREDICTED_DATA_SAVE, '{}_perio.png'.format(i)), test_X[i, :, :, :])
				cv2.imwrite(os.path.join(PATH_PREDICTED_DATA_SAVE, '{}_gnd.png'.format(i)), np.array(test_Y[i, :, :, :]).astype(np.int16))
				cv2.imwrite(os.path.join(PATH_PREDICTED_DATA_SAVE, '{}_pred.png'.format(i)), img)


if __name__ == '__main__':
	Controller().run()
	#Controller().inference()






