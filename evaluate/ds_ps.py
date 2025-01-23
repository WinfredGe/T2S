from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import pickle
import mgzip
import argparse
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import time
import warnings
warnings.filterwarnings("ignore")

def discriminative_score_metrics (ori_data, generated_data, iterations = 2000, rnn_name = 'gru'):
	"""Use post-hoc RNN to classify original data and synthetic data

	Args:
	- ori_data: original data
	- generated_data: generated synthetic data

	Returns:
	- discriminative_score: np.abs(classification accuracy - 0.5)
	"""
	# Initialization on the Graph
	tf.reset_default_graph()

	# Basic Parameters
	no, seq_len, dim = np.asarray(ori_data).shape    

	# Set maximum sequence length and each sequence length
	ori_time, ori_max_seq_len = extract_time(ori_data)
	generated_time, generated_max_seq_len = extract_time(ori_data)
	max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
	 
	## Builde a post-hoc RNN discriminator network 
	# Network parameters
	hidden_dim = int(dim/2)
	# iterations = 2000
	batch_size = 128

	# Input place holders
	# Feature
	X = tf.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x")
	X_hat = tf.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x_hat")

	T = tf.placeholder(tf.int32, [None], name = "myinput_t")
	T_hat = tf.placeholder(tf.int32, [None], name = "myinput_t_hat")

	# discriminator function
	def discriminator (x, t, rnn_name = 'gru'):
		"""
	Simple discriminator function.
	Args:
	  - x: time-series data
	  - t: time information
	Returns:
	  - y_hat_logit: logits of the discriminator output
	  - y_hat: discriminator output
	  - d_vars: discriminator variables
	  """
		with tf.variable_scope("discriminator", reuse = tf.AUTO_REUSE) as vs:
			# d_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'd_cell')
			if (rnn_name == 'gru'):
				d_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'd_cell')
			# LSTM
			elif (rnn_name == 'lstm'):
				d_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'd_cell')
			# LSTM Layer Normalization
			elif (rnn_name == 'lstmLN'):
				d_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'd_cell')


			d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, x, dtype=tf.float32, sequence_length = t)
			y_hat_logit = tf.contrib.layers.fully_connected(d_last_states, 1, activation_fn=None) 
			y_hat = tf.nn.sigmoid(y_hat_logit)
			d_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]

		return y_hat_logit, y_hat, d_vars

	y_logit_real, y_pred_real, d_vars = discriminator(X, T, rnn_name = 'gru')
	y_logit_fake, y_pred_fake, _ = discriminator(X_hat, T_hat, rnn_name = 'gru')
	    
	# Loss for the discriminator
	d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit_real, 
	                                                                   labels = tf.ones_like(y_logit_real)))
	d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit_fake, 
	                                                                   labels = tf.zeros_like(y_logit_fake)))
	d_loss = d_loss_real + d_loss_fake

	# optimizer
	d_solver = tf.train.AdamOptimizer().minimize(d_loss, var_list = d_vars)
	    
	## Train the discriminator   
	# Start session and initialize
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	# Train/test division for both original and generated data
	train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
	train_test_divide(ori_data, generated_data, ori_time, generated_time)

	# Training step
	for itt in range(iterations):
		# Batch setting
		X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
		X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)
		      
		# Train discriminator
		_, step_d_loss = sess.run([d_solver, d_loss], 
		                          feed_dict={X: X_mb, T: T_mb, X_hat: X_hat_mb, T_hat: T_hat_mb})            

	## Test the performance on the testing set    
	y_pred_real_curr, y_pred_fake_curr = sess.run([y_pred_real, y_pred_fake], 
	                                            feed_dict={X: test_x, T: test_t, X_hat: test_x_hat, T_hat: test_t_hat})

	y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis = 0))
	y_label_final = np.concatenate((np.ones([len(y_pred_real_curr),]), np.zeros([len(y_pred_fake_curr),])), axis = 0)

	# Compute the accuracy
	acc = accuracy_score(y_label_final, (y_pred_final>0.5))
	discriminative_score = np.abs(0.5-acc)

	return discriminative_score  


# ====================================================
def predictive_score_metrics (ori_data, generated_data, iterations = 5000, rnn_name = 'gru'):
	"""Report the performance of Post-hoc RNN one-step ahead prediction.

	Args:
	- ori_data: original data
	- generated_data: generated synthetic data

	Returns:
	- predictive_score: MAE of the predictions on the original data
	"""
	# Initialization on the Graph
	tf.reset_default_graph()

	# Basic Parameters
	no, seq_len, dim = np.asarray(ori_data).shape

	# Set maximum sequence length and each sequence length
	ori_time, ori_max_seq_len = extract_time(ori_data)
	generated_time, generated_max_seq_len = extract_time(ori_data)
	max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
	 
	## Builde a post-hoc RNN predictive network 
	# Network parameters
	hidden_dim = int(dim/2)
	# iterations = 5000
	batch_size = 128

	# Input place holders
	X = tf.placeholder(tf.float32, [None, max_seq_len-1, dim-1], name = "myinput_x")
	T = tf.placeholder(tf.int32, [None], name = "myinput_t")    
	Y = tf.placeholder(tf.float32, [None, max_seq_len-1, 1], name = "myinput_y")

	# Predictor function
	def predictor (x, t, rnn_name = 'gru'):
		"""Simple predictor function.

		Args:
		  - x: time-series data
		  - t: time information
		  
		Returns:
		  - y_hat: prediction
		  - p_vars: predictor variables
		"""
		with tf.variable_scope("predictor", reuse = tf.AUTO_REUSE) as vs:
			# p_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'p_cell')
			if (rnn_name == 'gru'):
				p_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'p_cell')
			# LSTM
			elif (rnn_name == 'lstm'):
				p_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'p_cell')
			# LSTM Layer Normalization
			elif (rnn_name == 'lstmLN'):
				p_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'p_cell')


			p_outputs, p_last_states = tf.nn.dynamic_rnn(p_cell, x, dtype=tf.float32, sequence_length = t)
			y_hat_logit = tf.contrib.layers.fully_connected(p_outputs, 1, activation_fn=None) 
			y_hat = tf.nn.sigmoid(y_hat_logit)
			p_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]

		return y_hat, p_vars
    
	y_pred, p_vars = predictor(X, T, rnn_name = 'gru')
	# Loss for the predictor
	p_loss = tf.losses.absolute_difference(Y, y_pred)
	# optimizer
	p_solver = tf.train.AdamOptimizer().minimize(p_loss, var_list = p_vars)
	    
	## Training    
	# Session start
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	# Training using Synthetic dataset
	for itt in range(iterations):
		      
		# Set mini-batch
		idx = np.random.permutation(len(generated_data))
		train_idx = idx[:batch_size]     
		        
		X_mb = list(generated_data[i][:-1,:(dim-1)] for i in train_idx)
		T_mb = list(generated_time[i]-1 for i in train_idx)
		Y_mb = list(np.reshape(generated_data[i][1:,(dim-1)],[len(generated_data[i][1:,(dim-1)]),1]) for i in train_idx)        
		      
		# Train predictor
		_, step_p_loss = sess.run([p_solver, p_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})        

	## Test the trained model on the original data
	idx = np.random.permutation(len(ori_data))
	train_idx = idx[:no]

	X_mb = list(ori_data[i][:-1,:(dim-1)] for i in train_idx)
	T_mb = list(ori_time[i]-1 for i in train_idx)
	Y_mb = list(np.reshape(ori_data[i][1:,(dim-1)], [len(ori_data[i][1:,(dim-1)]),1]) for i in train_idx)

	# Prediction
	pred_Y_curr = sess.run(y_pred, feed_dict={X: X_mb, T: T_mb})

	# Compute the performance in terms of MAE
	MAE_temp = 0
	for i in range(no):
		MAE_temp = MAE_temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i,:,:])

	predictive_score = MAE_temp / no

	return predictive_score

# ====================================================
def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate = 0.8):
	"""Divide train and test data for both original and synthetic data.

	Args:
	- data_x: original data
	- data_x_hat: generated data
	- data_t: original time
	- data_t_hat: generated time
	- train_rate: ratio of training data from the original data
	"""
	# Divide train/test index (original data)
	no = len(data_x)
	idx = np.random.permutation(no)
	train_idx = idx[:int(no*train_rate)]
	test_idx = idx[int(no*train_rate):]

	train_x = [data_x[i] for i in train_idx]
	test_x = [data_x[i] for i in test_idx]
	train_t = [data_t[i] for i in train_idx]
	test_t = [data_t[i] for i in test_idx]      

	# Divide train/test index (synthetic data)
	no = len(data_x_hat)
	idx = np.random.permutation(no)
	train_idx = idx[:int(no*train_rate)]
	test_idx = idx[int(no*train_rate):]

	train_x_hat = [data_x_hat[i] for i in train_idx]
	test_x_hat = [data_x_hat[i] for i in test_idx]
	train_t_hat = [data_t_hat[i] for i in train_idx]
	test_t_hat = [data_t_hat[i] for i in test_idx]

	return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time (data):
	"""Returns Maximum sequence length and each sequence length.

	Args:
	- data: original data

	Returns:
	- time: extracted time information
	- max_seq_len: maximum sequence length
	"""
	time = list()
	max_seq_len = 0
	for i in range(len(data)):
		max_seq_len = max(max_seq_len, len(data[i][:,0]))
		time.append(len(data[i][:,0]))

	return time, max_seq_len

def batch_generator(data, time, batch_size):
	"""Mini-batch generator.

	Args:
	- data: time-series data
	- time: time information
	- batch_size: the number of samples in each batch

	Returns:
	- X_mb: time-series data in each batch
	- T_mb: time information in each batch
	"""
	no = len(data)
	idx = np.random.permutation(no)
	train_idx = idx[:batch_size]     
	        
	X_mb = list(data[i] for i in train_idx)
	T_mb = list(time[i] for i in train_idx)

	return X_mb, T_mb



# ====================================================
if __name__=="__main__":
	parser = argparse.ArgumentParser(description='manual to this script')
	parser.add_argument('--method_name', type=str, default = None)
	parser.add_argument('--dataset_name', type=str, default = None)
	parser.add_argument('--dataset_state', type=str, default = None)
	parser.add_argument('--gpu_id', type=str, default = None)
	parser.add_argument('--gpu_fraction', type=float, default = None)
	args = parser.parse_args()

	method_name = args.method_name
	dataset_name = args.dataset_name
	dataset_state = args.dataset_state
	gpu_id = args.gpu_id
	gpu_fraction = args.gpu_fraction


	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id 
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



	with mgzip.open('./data/' + dataset_name + '_' + dataset_state + '.pkl', 'rb') as f:
		ori_data = pickle.load(f)


	with mgzip.open('./data/' + method_name + '/' + dataset_name + '_' + dataset_state + '_gen.pkl', 'rb') as f:
		generated_data = pickle.load(f)
	generated_data = np.array(generated_data)

	print(ori_data.shape, generated_data.shape)

	# ===========================================
	rnn_name = 'lstm'
	iter_disc = 2000
	iter_pred = 5000

	disc_all = []
	time_all = []
	for i in range(0,5):
		start = time.time()
		temp_disc = discriminative_score_metrics(ori_data, generated_data, iterations = iter_disc, rnn_name = rnn_name)
		end = time.time()
		disc_all.append(temp_disc)
		time_all.append(end-start)


	pred_all = []
	for i in range(0,5):
		start = time.time()
		temp_pred = predictive_score_metrics(ori_data, generated_data, iterations = iter_pred, rnn_name = rnn_name)
		end = time.time()
		pred_all.append(temp_pred)
		time_all.append(end-start)

	disc_all = np.array(disc_all)
	pred_all = np.array(pred_all)
	time_all = np.array(time_all)

	with open('../data/' + method_name + '/' + dataset_name + '_' + dataset_state + '_eval_model.pkl', 'wb') as f:
		pickle.dump([disc_all, pred_all, time_all], f)
