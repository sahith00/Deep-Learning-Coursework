import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

EPOCHS = 70

# Data shape: (M x 32 x 32 x 3)
# Labels shape: (M x 1 x 10)
def preprocess_data(data):
    new_data = tf.convert_to_tensor(data / 255.0, dtype=tf.float32)
    return new_data

def preprocess_labels(labels):
	new_labels = tf.reshape(tf.one_hot(labels, 10), [len(labels), 10])
	return new_labels

def load_data():
	training_data = np.load('./Prog3_data_Spring22/training_data.npy')
	training_labels = np.load('./Prog3_data_Spring22/training_label.npy')
	testing_data = np.load('./Prog3_data_Spring22/testing_data.npy')
	testing_labels = np.load('./Prog3_data_Spring22/testing_label.npy')

	# Not sparse
	training_data = preprocess_data(training_data)
	training_labels = preprocess_labels(training_labels)
	testing_data = preprocess_data(testing_data)
	testing_labels = preprocess_labels(testing_labels)

	# Sparse
	# training_data = preprocess_data(training_data)
	# testing_data = preprocess_data(testing_data)

	return training_data, training_labels, testing_data, testing_labels


# Model:
# Conv Layer with 16 5x5 filters and no zero padding with stride of 1
# Max pooling layer with 2x2 pooling and stride of 2
# Conv Layer with 32 5x5 filters and no zero padding with stride of 1
# Max pooling layer with 2x2 pooling and stride of 2
# Conv Layer with 64 3x3 filters and no zero padding with stride of 1
# Fully connected layer of 500 nodes with ReLU activation
# Output layer of 10 nodes with softmax output function
# ReLU activation on all conv maps
class CNN_Model(tf.keras.Model):
	def __init__(self):
		super(CNN_Model, self).__init__()
		self.conv_layer_1 = tf.keras.layers.Conv2D(16, 5, activation='relu', padding='valid', strides=1)
		self.dropout_layer_1 = tf.keras.layers.Dropout(0.2)
		self.pool_layer_1 = tf.keras.layers.MaxPool2D(2, 2)
		self.conv_layer_2 = tf.keras.layers.Conv2D(32, 5, activation='relu', padding='valid', strides=1)
		self.dropout_layer_2 = tf.keras.layers.Dropout(0.2)
		self.pool_layer_2 = tf.keras.layers.MaxPool2D(2, 2)
		self.conv_layer_3 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='valid', strides=1)
		self.dropout_layer_3 = tf.keras.layers.Dropout(0.2)
		self.flatten_layer = tf.keras.layers.Flatten()
		self.hidden_layer = tf.keras.layers.Dense(500, activation='relu')
		self.dropout_layer_4 = tf.keras.layers.Dropout(0.2)
		self.output_layer = tf.keras.layers.Dense(10, activation='softmax')
		

	def call(self, input_data, training=True):
		outputs = self.conv_layer_1(input_data)
		if training:
			outputs = self.dropout_layer_1(outputs, training=True)
		outputs = self.pool_layer_1(outputs)
		outputs = self.conv_layer_2(outputs)
		if training:
			outputs = self.dropout_layer_2(outputs, training=True)
		outputs = self.pool_layer_2(outputs)
		outputs = self.conv_layer_3(outputs)
		if training:
			outputs = self.dropout_layer_3(outputs, training=True)
		outputs = self.flatten_layer(outputs)
		outputs = self.hidden_layer(outputs)
		if training:
			outputs = self.dropout_layer_4(outputs, training=True)
		outputs = self.output_layer(outputs)

		return outputs

# Retrieves saved model if load = True
# Else, creates new CNN Model
def getModel(load_file=None, lr=0.001):
	if load_file is not None:
		new_model = tf.keras.models.load_model(load_file)
		return new_model

	model = CNN_Model()
	# Adagrad and Adadelta converges later, run with many more iterations (maybe 200, maybe load existing model and continue training)
	model.compile(
		optimizer=tf.keras.optimizers.Adamax(learning_rate=lr),
		loss=tf.keras.losses.CategoricalCrossentropy(),
		metrics=[tf.keras.metrics.CategoricalAccuracy()]
	)
	return model

# Plots provided loss and accuracy over e epochs
def plotLossAccuracy(loss, acc, e, name):
	epochs = list(range(e))

	plt.figure(1)
	plt.plot(epochs, loss, color='green')
	plt.title(name + " Loss vs. Epochs")
	plt.xlabel("epochs")
	plt.ylabel(name + " Loss")
	plt.savefig(name + ' Loss.png')

	plt.figure(2)
	plt.plot(epochs, acc, color='green')
	plt.title(name + " Accuracy vs. Epochs")
	plt.xlabel("epochs")
	plt.ylabel(name + " Accuracy")
	plt.savefig(name + ' Accuracy.png')
	plt.show()

# Computes classification error of pred_labels compared to true_labels for a single digit
def getClassificationError(pred_labels, true_labels, digit):
	correct_class = np.zeros((10, 1))
	correct_class[digit] = 1
	correct_class = tf.convert_to_tensor(correct_class, dtype=tf.float32)
	correct_vec = tf.matmul(true_labels, correct_class)

	occurences = tf.reduce_sum(correct_vec)
	incorrect = occurences - tf.reduce_sum(tf.matmul(tf.math.multiply(tf.one_hot(tf.math.argmax(pred_labels, axis=1), 10, axis=1), true_labels), correct_class))

	return incorrect.numpy() / occurences.numpy()

# Creates visualizations of features
def visualize(data):
	# Normalize data
	data_min = data.min()
	data = (data - data_min) / (data.max() - data_min)

	n_filters = data.shape[3]

	for i in range(n_filters):
		plt.figure()
		plt.title("Visualized filter #" + str(i))
		plt.imshow(data[:, :, :, i])
		plt.savefig("Visualized filter " + str(i) + ".png")
		plt.show()

training_data, training_labels, testing_data, testing_labels = load_data()

# Test various learning rates
# lrs = np.linspace(0.0001, 0.001, 5)
# avg_errors = []
# for lr in lrs:
# 	model = getModel(lr=lr)
# 	history = model.fit(training_data, training_labels, epochs=EPOCHS, validation_data=(testing_data, testing_labels))

# 	pred_labels = model.predict(testing_data)
# 	avg_error = 0.0
# 	for k in range(10):
# 		class_error = getClassificationError(pred_labels, testing_labels, k)
# 		avg_error += class_error
# 	avg_error = avg_error/10.0
# 	avg_errors.append(avg_error)

# plt.figure()
# plt.title("Average classification error vs initial learning rate")
# plt.plot(lrs, avg_errors, color='green')
# plt.xlabel("learning rate")
# plt.ylabel("error")
# plt.savefig("Learning rate testing.png")
# plt.show()



# Create and train model with training data
model = getModel(lr=0.001)
history = model.fit(training_data, training_labels, epochs=EPOCHS, validation_data=(testing_data, testing_labels))

# Record and plot loss and accuracy
train_loss = history.history['loss']
train_acc = history.history['categorical_accuracy']
plotLossAccuracy(train_loss, train_acc, EPOCHS, "Training")

# Evaluate model on test data and plot loss and accuracy
# Also record classification error for each class and average classification error
test_loss = history.history['val_loss']
test_acc = history.history['val_categorical_accuracy']
# print('Model accuracy: {:5.2f}%'.format(100 * test_acc))
plotLossAccuracy(test_loss, test_acc, EPOCHS, "Testing")


# Get classification error for each class and average classification error overall on test data
pred_labels = model.predict(testing_data)
# Add softmax to predicted labels if not in output layer of model
avg_error = 0.0
for k in range(10):
	class_error = getClassificationError(pred_labels, testing_labels, k)
	avg_error += class_error
	print("Classification error of class", k, ":", class_error)

avg_error = avg_error/10.0
print("Average classification error:", avg_error)

# Visualize filters of first convolutional layer
visualize(model.trainable_variables[0].numpy())


# Save model
model.save("trained_model")



# Evaluate existing model
# model = getModel("trained_model")
# loss, acc = model.evaluate(testing_data, testing_labels, verbose=2)
# print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))



# Layer examples from instructions
# tf.keras.layers.Conv2D(filters, kernel_size, padding, name)
# tf.keras.layers.MaxPool2D(pool_size, strides, padding, name)
# tf.keras.layers.Dense(output_units, activation, name)
# tf.keras.layers.Flatten()
# tf.nn.softmax_cross_entropy_with_logits()

# Softmax with logits used before activation

# Compare results with different optimizers

# Save model
# your_model.save("trained_model")

# Reload saved model
# new_model = tf.keras.models.load_model('trained_model')
# loss, acc = new_model.evaluate(testing_data, testing_label, verbose=2)
# print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

# Back propagation from instructions
# def back_prop(input_data):
# 	with tf.GradientTape(persistent = True) as T:
# 		output = self.forward(input_data)
# 		loss = compute_loss_function(output, input_labels)
# 	model_gradient = T.gradient(loss, your_model.trainable_variables)
# 	your_optimizer.apply_gradients(zip(model_gradient, your_model.trainable_variables))
