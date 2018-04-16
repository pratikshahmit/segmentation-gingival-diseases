import tensorflow as tf
import numpy as np
import cv2, os, sys
import pandas as pd
sys.path.append('../')
from base_info import *


class CNN_AutoEncoder_01:	

	def __init__(self, verbose=False):
		self.verbose = verbose


	def encoder_layer(self, layer_input=None, n_filters=None, k_size=3, stride=2, activation=None, padding='SAME', trainable=True, name=None):
		if n_filters == None:
			raise CustomException('Number of filter not provided to encoder layer {}'.format(name))
			return

		with tf.name_scope('name'):
			# batch_normalizaion
			#out = tf.layers.batch_normalization(inputs=layer_input, name='{}_{}'.format(name, 'batch_norm'))
			
			# conv2d
			out = tf.layers.conv2d(inputs=layer_input, filters=n_filters, kernel_size=k_size, strides=1, padding='SAME', activation=None, 
									use_bias=True, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=1234), bias_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=1234), 
									trainable=trainable, name='{}_{}'.format(name, 'conv2d'))
		
			# relu
			if activation == tf.nn.relu:
				out = tf.nn.relu(out, name='{}_{}'.format(name, 'relu'))
			elif activation == tf.sigmoid:	
				out = tf.sigmoid(out, name='{}_{}'.format(name, 'sigmoid'))
			elif activation == tf.nn.softmax:
				out = tf.nn.softmax(out, name='{}_{}'.format(name, 'softmax'))	
			
			# max_pooling2d
			out = tf.layers.max_pooling2d(inputs=out, pool_size=stride, strides=stride, padding='SAME', name='{}_{}'.format(name, 'max_pooling2d'))
			
			return out


	def flatten(self, layer_input, name):
		with tf.name_scope(name):
			sh = layer_input.get_shape().as_list()[1:]
			return tf.reshape(layer_input, shape=[-1, sh[0]*sh[1]*sh[2]])


	def decoder_layer(self, layer_input=None, is_flat=False, last_img_shape=None, n_filters=None, k_size=3, stride=2, activation=None, padding='SAME', unpooling_size=None, trainable=True, name=None):
		if n_filters == None:
			raise CustomException('Number of filter not provided to decoder layer {}'.format(name))
			return

		if is_flat == True:
			sh2 = last_img_shape[1:]
			layer_input = tf.reshape(layer_input, shape=[-1, sh2[0], sh2[1], sh2[2]])
		
		# batch_normalizaion
		#out = tf.layers.batch_normalization(inputs=layer_input, name='{}_{}'.format(name, 'batch_norm'))
		
		# Upsampling
		out = tf.image.resize_images(layer_input, size=unpooling_size)

		# conv2d_transpose
		out = tf.layers.conv2d_transpose(inputs=out, filters=n_filters, kernel_size=k_size, strides=1, padding='SAME', activation=None, use_bias=True, 
									kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=1234), bias_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=1234), 
									trainable=trainable, name='{}_{}'.format(name, 'conv2d'))

		# relu
		if activation == tf.nn.relu:
			out = tf.nn.relu(out, name='{}_{}'.format(name, 'relu'))
		elif activation == tf.sigmoid:	
			out = tf.sigmoid(out, name='{}_{}'.format(name, 'sigmoid'))
		elif activation == tf.nn.softmax:
			out = tf.nn.softmax(out, name='{}_{}'.format(name, 'softmax'))
		
		return out
	

	def add_layers(self, a, b):
		return tf.concat([a, b], axis=-1)


	def network(self, network_input=None):
		# Encoder
		encode1 = self.encoder_layer(layer_input=network_input, n_filters=FILTERS[0], activation=tf.nn.relu, trainable=True, name='Encoder__L1')
		if self.verbose: print('INPUT -> {},\t\tLayer Output Shape:\t{}'.format('Encode1', encode1.get_shape().as_list()))
		encode2 = self.encoder_layer(layer_input=encode1, n_filters=FILTERS[1], activation=tf.nn.relu, trainable=True, name='Encoder__L2')
		if self.verbose: print('{} -> {},\t\tLayer Output Shape:\t{}'.format('Encode1', 'Encode2', encode2.get_shape().as_list()))
		encode3 = self.encoder_layer(layer_input=encode2, n_filters=FILTERS[2], activation=tf.nn.relu, trainable=True, name='Encoder__L3')
		if self.verbose: print('{} -> {},\t\tLayer Output Shape:\t{}'.format('Encode2', 'Encode3', encode3.get_shape().as_list()))
		encode4 = self.encoder_layer(layer_input=encode3, n_filters=FILTERS[3], activation=tf.nn.relu, trainable=True, name='Encoder__L4')
		if self.verbose: print('{} -> {},\t\tLayer Output Shape:\t{}'.format('Encode3', 'Encode4', encode4.get_shape().as_list()))
		encode5 = self.encoder_layer(layer_input=encode4, n_filters=FILTERS[4], activation=tf.nn.relu, trainable=True, name='Encoder__L5')
		if self.verbose: print('{} -> {},\t\tLayer Output Shape:\t{}'.format('Encode4', 'Encode5', encode5.get_shape().as_list()))

		# Decoder
		decode1 = self.decoder_layer(layer_input=encode5, is_flat=False, last_img_shape=encode5.get_shape().as_list(), n_filters=FILTERS[3], activation=tf.nn.relu, unpooling_size=[30, 40], trainable=True, name='Decoder__L1')
		decode1 = self.add_layers(decode1, encode4)		
		if self.verbose: print('{} -> {},\t\tLayer Output Shape:\t{}'.format('Encode5', 'Decode1', decode1.get_shape().as_list()))
		decode2 = self.decoder_layer(layer_input=decode1, is_flat=False, last_img_shape=None, n_filters=FILTERS[2], activation=tf.nn.relu, unpooling_size=[60, 80], trainable=True, name='Decoder__L2')
		decode2 = self.add_layers(decode2, encode3)
		if self.verbose: print('{} -> {},\t\tLayer Output Shape:\t{}'.format('Decode1', 'Decode2', decode2.get_shape().as_list()))
		decode3 = self.decoder_layer(layer_input=decode2, is_flat=False, last_img_shape=None, n_filters=FILTERS[1], activation=tf.nn.relu, unpooling_size=[120, 160], trainable=True, name='Decoder__L3')
		decode3 = self.add_layers(decode3, encode2)
		if self.verbose: print('{} -> {},\t\tLayer Output Shape:\t{}'.format('Encode2', 'Decode3', decode3.get_shape().as_list()))
		decode4 = self.decoder_layer(layer_input=decode3, is_flat=False, last_img_shape=None, n_filters=FILTERS[0], activation=tf.nn.relu, unpooling_size=[240, 320], trainable=True, name='Decoder__L4')
		decode4 = self.add_layers(decode4, encode1)
		if self.verbose: print('{} -> {},\t\tLayer Output Shape:\t{}'.format('Encode3', 'Decode4', decode4.get_shape().as_list()))
		decode5 = self.decoder_layer(layer_input=decode4, is_flat=False, last_img_shape=None, n_filters=NUM_NN_IMAGE_OUTPUTS, activation=tf.sigmoid, unpooling_size=[480, 640], trainable=True, name='Decoder__L5')
		if self.verbose: print('{} -> {},\t\tLayer Output Shape:\t{}'.format('Encode4', 'Decode5', decode5.get_shape().as_list()))

		return decode5









