import matplotlib.pyplot as plt
import numpy as np
import pickle
import random as r
import tensorflow as tf
import time

import data_load as dl

# Preprocessing of data done in data_load

# Model: NN with one input layer (784 nodes), two hidden layers (100 nodes each), output layer (10 nodes)

class Layer:
	def __init__(self, prev_nodes, n_nodes):
		self.n_nodes = n_nodes
		self.prev_nodes = prev_nodes

		# Represents the weight and bias matrices that precede the current layer
		# np.random.normal provides a normal distribution over a given standard deviation
		self.W = tf.convert_to_tensor(np.random.normal(scale=0.1, size=(prev_nodes, n_nodes)).reshape(prev_nodes, n_nodes), dtype=tf.float32)
		self.bias = tf.zeros([n_nodes, 1])

		# Create hidden layer from number of nodes
		self.H = tf.zeros([n_nodes, 1])
		self.Hs = []

		# Store each gradient for each variable over a batch of (batch_size) samples (size determined in training)
		self.W_gradients = []
		self.bias_gradients = []
		self.H_gradients = []

		self.W_gradient = None
		self.bias_gradient = None

		# Store average gradients as well for batch training
		self.W_gradient_avg = None
		self.bias_gradient_avg = None

	def computeGradientAvgs(self):
		self.W_gradient_avg = tf.add_n(self.W_gradients) / len(self.W_gradients)
		self.bias_gradient_avg = tf.add_n(self.bias_gradients) / len(self.bias_gradients)

	def getRegularizationMats(self):
		max_index = tf.argmax(tf.reduce_sum(tf.abs(self.W), axis=0)).numpy()
		regmat_W = np.zeros(self.W.shape)
		regmat_W[np.arange(self.prev_nodes), max_index] = tf.sign(self.W[:, max_index])

		regmat_bias = tf.fill([self.bias.shape[0], self.bias.shape[1]], np.sign(np.sum(np.abs(self.bias))))
		

		return (tf.convert_to_tensor(regmat_W, dtype=tf.float32), regmat_bias)


class Model:
	# Sets up the structure of the model, no data yet
	def __init__(self, input_size, layer_params, output_size):
		self.N = input_size
		self.K = output_size
		self.L = len(layer_params)

		self.layers = [None] * self.L
		self.layers[0] = Layer(self.N, layer_params[0])
		self.layers[1] = Layer(layer_params[0], layer_params[1])
		
		# For the output layer, H = y_hat (classification)
		self.output_layer = Layer(layer_params[-1], self.K)

	# Calculate the nodes for all layers until the output
	def forward_prop(self, input_data, activation='ReLU', output='softmax'):
		# First hidden layer (H_0 = activation((W^0^T X) + W^0_0))
		self.layers[0].H = tf.nn.relu(tf.matmul(self.layers[0].W, input_data, transpose_a=True) + self.layers[0].bias)
		self.layers[0].Hs.append(self.layers[0].H)


		# Next L-1 hidden layers (H_(l) = activation((W^l^T H_(l-1)) + W^l_0))
		self.layers[1].H = tf.nn.relu(tf.matmul(self.layers[1].W, self.layers[0].H, transpose_a=True) + self.layers[1].bias)
		self.layers[1].Hs.append(self.layers[1].H)

		# Output layer (H_L = output((W^L^T H_(L-1)) + W^L_0))
		self.output_layer.H = tf.matmul(self.output_layer.W, self.layers[1].H, transpose_a=True) + self.output_layer.bias
		self.output_layer.Hs.append(self.output_layer.H)

	def forward_prop_batch(self, input_data, batch_size, activation='ReLU', output='softmax'):
		# for i in range(self.L):
		# 	layer_nodes = tf.nn.relu(tf.matmul(self.layers[i].W, layer_nodes, transpose_a=True) + self.layers[i].bias)
		layer_nodes = tf.nn.relu(tf.matmul(self.layers[0].W, input_data, transpose_a=True) + tf.repeat(self.layers[0].bias, batch_size, axis=1))
		layer_nodes = tf.nn.relu(tf.matmul(self.layers[1].W, layer_nodes, transpose_a=True) + tf.repeat(self.layers[1].bias, batch_size, axis=1))
		#Y_hat = tf.nn.softmax(tf.matmul(self.output_layer.W, layer_nodes, transpose_a=True) + tf.repeat(self.output_layer.bias, batch_size, axis=1), axis=0)
		Y_hat = tf.matmul(self.output_layer.W, layer_nodes, transpose_a=True) + tf.repeat(self.output_layer.bias, batch_size, axis=1)

		return Y_hat

	# Calculate the gradient of each layer and weight matrix, starting with the output layer
	def backward_prop(self, train_data, train_labels):
		# Output layer gradients
		self.output_layer.H_gradient = -(tf.cast(train_labels, dtype=tf.float64) - self.output_layer.H)
		true_index = tf.argmax(train_labels)[0]

		prev_H = self.layers[-1].H
		z_vec = tf.matmul(self.output_layer.W, prev_H, transpose_a=True) + self.output_layer.bias
		softmax_z = tf.nn.softmax(z_vec, axis=0)

		bias_gradient = softmax_z * softmax_z[true_index] * self.output_layer.H_gradient[true_index]

		#bias_gradient = tf.convert_to_tensor(bias_gradient)
		self.output_layer.bias_gradients.append(bias_gradient)
		self.layers[-1].H_gradient = tf.matmul(self.output_layer.W, bias_gradient)
		self.output_layer.W_gradients.append(tf.matmul(prev_H, bias_gradient, transpose_b=True))

		# Hidden layer gradients
		layer_nodes = self.layers[0].H
		bias_gradient = tf.nn.relu(self.layers[1].H_gradient)
		W_gradient = tf.matmul(layer_nodes, bias_gradient, transpose_b=True)

		self.layers[1].bias_gradients.append(bias_gradient)
		self.layers[0].H_gradient = tf.matmul(self.layers[1].W, bias_gradient)
		self.layers[1].W_gradients.append(W_gradient)


		layer_nodes = tf.cast(train_data, dtype=tf.float64)
		bias_gradient = tf.nn.relu(self.layers[0].H_gradient)
		W_gradient = tf.matmul(layer_nodes, bias_gradient, transpose_b=True)

		self.layers[0].bias_gradients.append(bias_gradient)
		self.layers[0].W_gradients.append(W_gradient)

	def backward_prop_batch(self, train_data, train_labels, batch_size):
		# Convert set of Hs to tensor for use in back prop
		H_vecs_output = tf.reshape(tf.convert_to_tensor(self.output_layer.Hs), [self.K, batch_size])
		H_vecs_l1 = tf.reshape(tf.convert_to_tensor(self.layers[1].Hs), [100, batch_size])
		H_vecs_l0 = tf.reshape(tf.convert_to_tensor(self.layers[0].Hs), [100, batch_size])

		# Output layer gradients
		self.output_layer.H_gradients = -(1./50.) * (train_labels - tf.nn.softmax(H_vecs_output, axis=0))

		self.output_layer.bias_gradient = tf.matmul(self.output_layer.H_gradients, tf.ones([batch_size, 1]))
		self.output_layer.W_gradient = tf.matmul(H_vecs_l1, self.output_layer.H_gradients, transpose_b=True)
		self.layers[1].H_gradients = tf.matmul(self.output_layer.W, self.output_layer.H_gradients)

		# Hidden layer gradients
		self.layers[1].bias_gradient = tf.matmul(tf.math.multiply(self.layers[1].H_gradients, tf.round(H_vecs_l1 / (H_vecs_l1 + 0.000000001))), tf.ones([batch_size, 1]))
		self.layers[1].W_gradient = tf.matmul(H_vecs_l0, tf.math.multiply(self.layers[1].H_gradients, tf.round(H_vecs_l1 / (H_vecs_l1 + 0.000000001))), transpose_b=True)
		self.layers[0].H_gradients = tf.matmul(self.layers[1].W, tf.math.multiply(self.layers[1].H_gradients, tf.round(H_vecs_l1 / (H_vecs_l1 + 0.000000001))))

		self.layers[0].bias_gradient = tf.matmul(tf.math.multiply(self.layers[0].H_gradients, tf.round(H_vecs_l0 / (H_vecs_l0 + 0.000000001))), tf.ones([batch_size, 1]))
		self.layers[0].W_gradient = tf.matmul(train_data, tf.math.multiply(self.layers[0].H_gradients, tf.round(H_vecs_l0 / (H_vecs_l0 + 0.000000001))), transpose_b=True)

	def update_weights(self, alpha, reg):
		# Update the weights using the computed gradients and regularization matrices

		# With regularization
		#regmats = self.layers[0].getRegularizationMats()
		self.layers[0].W = self.layers[0].W - alpha * (self.layers[0].W_gradient + reg * tf.sign(self.layers[0].W))
		self.layers[0].bias = self.layers[0].bias - alpha * (self.layers[0].bias_gradient + reg * tf.sign(self.layers[0].bias))

		#regmats = self.layers[1].getRegularizationMats()
		self.layers[1].W = self.layers[1].W - alpha * (self.layers[1].W_gradient + reg * tf.sign(self.layers[1].W))
		self.layers[1].bias = self.layers[1].bias - alpha * (self.layers[1].bias_gradient + reg * tf.sign(self.layers[1].bias))

		#regmats = self.output_layer.getRegularizationMats()
		self.output_layer.W = self.output_layer.W - alpha * (self.output_layer.W_gradient + reg * tf.sign(self.output_layer.W))
		self.output_layer.bias = self.output_layer.bias - alpha * (self.output_layer.bias_gradient + reg * tf.sign(self.output_layer.bias))

		# Without regularization
		# self.layers[0].W = self.layers[0].W - (alpha * self.layers[0].W_gradient)
		# self.layers[0].bias = self.layers[0].bias - (alpha * self.layers[0].bias_gradient)

		# self.layers[1].W = self.layers[1].W - (alpha * self.layers[1].W_gradient)
		# self.layers[1].bias = self.layers[1].bias - (alpha * self.layers[1].bias_gradient)

		# self.output_layer.W = self.output_layer.W - (alpha * self.output_layer.W_gradient)
		# self.output_layer.bias = self.output_layer.bias - (alpha * self.output_layer.bias_gradient)

	# Compute cross-entropy loss (L = -sum_k=1^K Y_k*log(Y_hat_k))
	def compute_loss_accuracy(self, data, labels):
		Y_hat = tf.nn.softmax(self.forward_prop_batch(data, tf.shape(data)[1].numpy()), axis=0)
		loss = tf.reduce_mean(tf.reduce_sum(-tf.math.multiply(labels, tf.math.log(Y_hat)), axis=0))
		# predict vector
		predict = tf.one_hot(tf.math.argmax(Y_hat, axis=0), self.K, dtype=tf.float32, axis=0)
		# accuracy
		accuracy = tf.reduce_mean(tf.reduce_sum(tf.math.multiply(labels, predict), axis=0))
		#print(train_labels, self.output_layer.H, loss)
		return loss.numpy(), accuracy.numpy()

	# Compute # of incorrect classifications / # of total occurences of digit parameter
	def getClassificationError(self, pred_labels, true_labels, digit):
		correct_class = np.zeros((self.K, 1))
		correct_class[digit] = 1
		correct_class = tf.convert_to_tensor(correct_class, dtype=tf.float32)
		correct_vec = tf.matmul(true_labels, correct_class, transpose_a=True)

		occurences = tf.reduce_sum(correct_vec)
		incorrect = occurences - tf.reduce_sum(tf.matmul(tf.math.multiply(pred_labels, true_labels), correct_class, transpose_a=True))

		return incorrect.numpy() / occurences.numpy()

	def train_and_plot(self, train_data, train_labels, test_data, test_labels, batch_size, alpha=0.001, reg=0.001, epochs=1000):
		# Do forward propagation, then back propagation, then update weights in each layer for each batch of (batch_size) data points
		prev_loss = 0
		train_losses = []
		test_losses = []
		train_accs = []
		test_accs = []

		train_class_errors = {}
		test_class_errors = {}
		for k in range(self.K):
			train_class_errors[k] = []
			test_class_errors[k] = []
		
		train_loss, train_accuracy = self.compute_loss_accuracy(tf.transpose(train_data[:, :-1]), tf.transpose(train_labels))
		print("End of epoch 0", "(loss:", train_loss, ", accuracy:", train_accuracy, ")")

		# alpha = 0.1 * tf.reduce_mean(tf.norm(self.output_layer.H_gradients, axis=0))

		interval = 10
		# lr_interval = 2
		# update_t = 0

		t = -1
		while t < epochs-1:
			start = time.time()
			t += 1
			total_loss = 0
			full_indeces = list(range(len(train_data)))
			# Loops through each batch of data
			for i in range(0, len(train_data), batch_size):
				# set random seed
				r.seed(i)
				# generate batch indices
				id_list = r.sample(full_indeces, batch_size)
				# take batch data
				batch_data = tf.gather(train_data, id_list, axis=0)
				# take batch labels
				batch_labels = tf.gather(train_labels, id_list, axis=0)

				batch_data = tf.transpose(batch_data[:, :-1])
				batch_labels = tf.transpose(batch_labels)

				# Reset batch gradients for each new batch
				self.layers[0].Hs = []
				self.layers[0].W_gradients = []
				self.layers[0].H_gradients = []
				self.layers[0].bias_gradients = []

				self.layers[1].Hs = []
				self.layers[1].W_gradients = []
				self.layers[1].H_gradients = []
				self.layers[1].bias_gradients = []

				self.output_layer.Hs = []
				self.output_layer.W_gradients = []
				self.output_layer.H_gradients = []
				self.output_layer.bias_gradients = []

				j = 0
				while j < batch_size:
					sample_data = tf.reshape(batch_data[:, j], [self.N, 1])
					sample_label = tf.reshape(batch_labels[:, j], [self.K, 1])

					self.forward_prop(sample_data)
					j += 1

				self.backward_prop_batch(batch_data, batch_labels, batch_size)
				self.update_weights(alpha, reg)


				loss, accuracy = self.compute_loss_accuracy(batch_data, batch_labels)
				#print(loss, accuracy)


				# Confirmation: do forward prop again and confirm that new loss is less than previous loss
				# new_loss = 0
				# for j in range(batch_size):
				# 	Y_hat = self.forward_prop(tf.reshape(batch_data[:, j], [batch_data.shape[0], 1]))
				# 	new_loss += self.compute_loss(tf.reshape(batch_labels[:, j], [batch_labels.shape[0], 1]))
				# print(loss, new_loss)

				# Compute total loss to determine average loss after each epoch
				total_loss += loss

				# Timer output
				# print("Single batch training:", end - start)
			
			train_loss, train_accuracy = self.compute_loss_accuracy(tf.transpose(train_data[:, :-1]), tf.transpose(train_labels))
			print("End of epoch", t+1, "(loss:", train_loss, ", accuracy:", train_accuracy, ", learning rate:", alpha, ")")			

			# update_t
			# if t == update_t:
			# 	alpha = alpha/2.0
			# 	update_t = t + lr_interval
			# 	lr_interval += 1

			if (t+1) % 2 == 0:
				alpha = alpha/2.0

			if (t+1) % interval == 0:
				test_loss, test_accuracy = self.compute_loss_accuracy(tf.transpose(test_data[:, :-1]), tf.transpose(test_labels))
				train_losses.append(train_loss)
				train_accs.append(train_accuracy)
				test_losses.append(test_loss)
				test_accs.append(test_accuracy)

				# alpha = alpha/float(t+1)

				for k in range(self.K):
					train_class_errors[k].append(self.getClassificationError(self.classify(tf.transpose(train_data[:, :-1])), tf.transpose(train_labels), k))
					test_class_errors[k].append(self.getClassificationError(self.classify(tf.transpose(test_data[:, :-1])), tf.transpose(test_labels), k))

			end = time.time()

			# Timer output over each epoch
			# print("Single epoch of training:", end - start)
		
		# Plot loss and accuracy vectors over number of iterations
		e_list = list(range(interval, epochs+interval, interval))
		self.plot_epochs(train_losses, e_list, "Training Loss")
		self.plot_epochs(test_losses, e_list, "Testing Loss")
		self.plot_epochs(train_accs, e_list, "Training Accuracy")
		self.plot_epochs(test_accs, e_list, "Testing Accuracy")

		# Plot classification error for each digit as well
		for k in range(self.K):
			self.plot_epochs(train_class_errors[k], e_list, "Training Error " + str(k))
			self.plot_epochs(test_class_errors[k], e_list, "Testing Error " + str(k))

		#print(prev_loss, total_loss)
		test_loss, test_accuracy = self.compute_loss_accuracy(tf.transpose(test_data[:, :-1]), tf.transpose(test_labels))
		print("Final test classification average error:", 1 - test_accuracy)

	def classify(self, test_data):
		Y_hat = self.forward_prop_batch(test_data, tf.shape(test_data)[1].numpy())
		return tf.one_hot(tf.math.argmax(Y_hat, axis=0), self.K, dtype=tf.float32, axis=0)

	def plot_epochs(self, data, epochs, name):
		plt.figure()
		plt.plot(epochs, data, color='green')
		plt.title(name + " vs. Epochs")
		plt.xlabel("epochs")
		plt.ylabel(name)
		plt.savefig(name + '.png')
		plt.show()

	def saveWeights(self, file):
		# Store weight and bias of each layer into theta
		theta = []
		for i in range(self.L):
			theta.append(self.layers[i].W.numpy())
			theta.append(self.layers[i].bias.numpy())
		theta.append(self.output_layer.W.numpy())
		theta.append(self.output_layer.bias.numpy())

		# Save theta in file using script provided in assignment
		filehandler = open(file, 'wb')
		pickle.dump(theta, filehandler, protocol=2)
		filehandler.close()

# Process data to remove extra 1 from data and transpose for model purpose
train_data = dl.training_set
train_labels = dl.training_labels
test_data = dl.testing_set
test_labels = dl.testing_labels


nn_model = Model(784, [100, 100], 10)
nn_model.train_and_plot(train_data, train_labels, test_data, test_labels, batch_size=50, epochs=50)
nn_model.saveWeights('nn_parameters.txt')


