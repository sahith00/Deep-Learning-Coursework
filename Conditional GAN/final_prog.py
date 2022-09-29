import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from math import floor
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from scipy.linalg import sqrtm
from skimage.transform import resize

EPOCHS = 2
batch_size = 202

real_images = np.load('./FinalProg_Spring22/images.npy')
attributes = np.load('./FinalProg_Spring22/attributes5.npy')

# (202599, 64, 64, 3), (202599, 5)
print(real_images.shape)
print(attributes.shape)

# assumes images have the shape 299x299x3, pixels in [0,255]
# Taken from machinelearningmastery.com (citation in report)
def calculate_inception_score(images, n_split=10, eps=1E-16):
	model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
	# predict class probabilities for images
	yhat = model.predict(images)
	# enumerate splits of images/predictions
	scores = list()
	n_part = floor(images.shape[0] / n_split)
	for i in range(n_split):
		# retrieve p(y|x)
		ix_start, ix_end = i * n_part, i * n_part + n_part
		p_yx = yhat[ix_start:ix_end]
		# calculate p(y)
		p_y = np.expand_dims(p_yx.mean(axis=0), 0)
		# calculate KL divergence using log probabilities
		kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
		# sum over classes
		sum_kl_d = kl_d.sum(axis=1)
		# average over images
		avg_kl_d = np.mean(sum_kl_d)
		# undo the log
		is_score = np.exp(avg_kl_d)
		# store
		scores.append(is_score)
	# average across images
	is_avg = np.mean(scores)
	return is_avg

# calculate frechet inception distance
# Taken from machinelearningmastery.com (citation in report)
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

# scale an array of images to a new size
# Taken from machinelearningmastery.com (citation in report)
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return np.asarray(images_list)

# Retrieves saved model if load = True
# Else, creates new Deep Dynamic Model
def getModel(load_file=None, lr=0.0001):
	if load_file is not None:
		new_model = tf.keras.models.load_model(load_file)
		return new_model

	model = cGAN()
	# Add metrics in project description to compile call
	model.compile(
		dis_optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5),
		gen_optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5),
		loss='binary_crossentropy'
	)
	return model

################################################################################################
class FID(tf.keras.metrics.Metric):
	def __init__(self):
		super(FID, self).__init__()
		self.fid = self.add_weight(name='fid')

	def update_state(self, true_val, predict_val, sample_weight=None):
		self.fid = tf.math.reduce_mean(tf.math.reduce_euclidean_norm((true_val - predict_val), axis=3)) * 1000

	def result(self):
		return self.fid

class CustomCallback(tf.keras.callbacks.Callback):
	def __init__(self, validation_data):
		super(CustomCallback, self).__init__()
		self.use_images, self.use_attributes = validation_data

	def on_epoch_end(self, epoch, logs=None):
		# Notice `training` is set to False.
		# This is so all layers run in inference mode (batchnorm).
		z_input = np.random.rand(16*100)
		z_input = tf.cast(tf.reshape(z_input, (16, 1, 1, 100)), dtype=tf.float32)
		y_input = np.random.randint(2, size=16*5)
		y_input = tf.cast(tf.reshape(y_input, (16, 1, 1, 5)), dtype=tf.float32)

		predictions = self.model.generator((z_input, y_input), training=False)

		fig = plt.figure(figsize=(4, 4))

		for i in range(predictions.shape[0]):
			plt.subplot(4, 4, i+1)
			plt.imshow((predictions[i] + 1) / 2.0)
			plt.axis('off')

		plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
		plt.show(block=False)
		plt.pause(1)
		plt.close()

		# Validation test (FID Score)
		num_images = batch_size
		z_input = np.random.rand(num_images*100)
		z_input = tf.cast(tf.reshape(z_input, (num_images, 1, 1, 100)), dtype=tf.float32)
		# y_input = np.random.randint(2, size=num_images*5)
		y_input = self.use_attributes
		y_input = tf.cast(tf.reshape(y_input, (num_images, 1, 1, 5)), dtype=tf.float32)
		gen_images = self.model.generator((z_input, y_input), training=False)

		gen_images = gen_images.numpy()
		gen_images = scale_images(gen_images, (299,299,3))
		gen_images = preprocess_input(gen_images)

		model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

		print("\nFID:", calculate_fid(model, gen_images, self.use_images), "IS:", calculate_inception_score(gen_images))

# cGAN Model
class Generator(tf.keras.Model):
	def __init__(self):
		super(Generator, self).__init__()
		start_filters = 256
		self.deconv1 = tf.keras.layers.Conv2DTranspose(start_filters, (4, 4), activation='relu')
		self.batch_norm1 = tf.keras.layers.BatchNormalization()
		self.deconv2 = tf.keras.layers.Conv2DTranspose(start_filters, (4, 4), activation='relu')
		self.batch_norm2 = tf.keras.layers.BatchNormalization()
		self.concat = tf.keras.layers.Concatenate(axis=-1)
		self.deconv3 = tf.keras.layers.Conv2DTranspose(start_filters, (5, 5), strides=(2, 2), padding='same', activation='relu')
		self.batch_norm3 = tf.keras.layers.BatchNormalization()
		self.deconv4 = tf.keras.layers.Conv2DTranspose(start_filters/2, (5, 5), strides=(2, 2), padding='same', activation='relu')
		self.batch_norm4 = tf.keras.layers.BatchNormalization()
		self.deconv5 = tf.keras.layers.Conv2DTranspose(start_filters/4, (5, 5), strides=(2, 2), padding='same', activation='relu')
		self.batch_norm5 = tf.keras.layers.BatchNormalization()
		self.deconv6 = tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh')

	def call(self, inputs, training=False):
		input_noise, input_labels = inputs
		output_noise = self.deconv1(input_noise)
		output_noise = self.batch_norm1(output_noise, training=training)
		output_labels = self.deconv2(input_labels)
		output_labels = self.batch_norm2(output_labels, training=training)
		output = self.concat((output_noise, output_labels))
		output = self.deconv3(output)
		output = self.batch_norm3(output, training=training)
		output = self.deconv4(output)
		output = self.batch_norm4(output, training=training)
		output = self.deconv5(output)
		output = self.batch_norm5(output, training=training)
		output = self.deconv6(output)

		return output

class Discriminator(tf.keras.Model):
	def __init__(self):
		super(Discriminator, self).__init__()
		start_filters = 32
		self.conv1 = tf.keras.layers.Conv2D(start_filters, (2, 2), (2, 2))
		self.batch_norm1 = tf.keras.layers.BatchNormalization()
		self.conv2 = tf.keras.layers.Conv2D(start_filters, (2, 2), (2, 2))
		self.batch_norm2 = tf.keras.layers.BatchNormalization()
		self.concat = tf.keras.layers.Concatenate(axis=-1)
		self.conv3 = tf.keras.layers.Conv2D(start_filters*4, (2, 2), strides=(2, 2), padding='same')
		self.batch_norm3 = tf.keras.layers.BatchNormalization()
		self.conv4 = tf.keras.layers.Conv2D(start_filters*8, (2, 2), strides=(2, 2), padding='same')
		self.batch_norm4 = tf.keras.layers.BatchNormalization()
		self.conv5 = tf.keras.layers.Conv2D(start_filters*16, (2, 2), strides=(2, 2), padding='same')
		self.batch_norm5 = tf.keras.layers.BatchNormalization()
		self.flatten = tf.keras.layers.Flatten()
		self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
		self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)

	def call(self, inputs, training=False):
		start_filters = 32
		input_image, input_labels = inputs
		input_labels = tf.repeat(input_labels, repeats=[4096], axis=1)
		output_labels = tf.reshape(input_labels, (tf.shape(input_labels)[0], 64, 64, 5))
		output_image = self.conv1(input_image)
		output_image = self.leaky_relu(output_image)
		output_image = self.batch_norm1(output_image, training=training)
		output_labels = self.conv2(output_labels)
		output_labels = self.leaky_relu(output_labels)
		output_labels = self.batch_norm2(output_labels, training=training)
		output = self.concat((output_image, output_labels))
		output = self.conv3(output)
		output = self.leaky_relu(output)
		output = self.batch_norm3(output, training=training)
		output = self.conv4(output)
		output = self.leaky_relu(output)
		output = self.batch_norm4(output, training=training)
		output = self.conv5(output)
		output = self.leaky_relu(output)
		output = self.batch_norm5(output, training=training)
		output = self.flatten(output)
		output = self.output_layer(output)

		return output		

class cGAN(tf.keras.Model):
	def __init__(self):
		super(cGAN, self).__init__()
		#self.generator = Generator()
		self.generator = tf.keras.models.load_model("trained_generator")
		self.discriminator = Discriminator()
		self.data_size = batch_size

	def compile(self, dis_optimizer, gen_optimizer, loss):
		super(cGAN, self).compile(loss=loss)
		self.dis_optimizer = dis_optimizer
		self.gen_optimizer = gen_optimizer

	def train_step(self, inputs):
		# Create all data for training
		# Extract real images and real labels from inputs
		input_images, input_labels = inputs[0]

		# Create random input noise for generator based on d=100
		input_noise = np.random.rand(self.data_size * 100)
		input_noise = tf.cast(tf.reshape(input_noise, (self.data_size, 1, 1, 100)), dtype=tf.float32)

		gen_labels = tf.reshape(input_labels, (self.data_size, 1, 1, 5))

		# Run forward and loss computation with gradient tape
		with tf.GradientTape() as dis_tape, tf.GradientTape() as gen_tape:
			fake_images = self.generator((input_noise, gen_labels), training=True)

			real_pred = self.discriminator((input_images, input_labels), training=True)
			fake_pred = self.discriminator((fake_images, input_labels), training=True)

			gen_loss = self.compiled_loss(tf.ones(self.data_size, 1), fake_pred)
			dis_loss = self.compiled_loss(tf.concat((tf.ones(self.data_size, 1), tf.zeros(self.data_size, 1)), axis=0), tf.concat((real_pred, fake_pred), axis=0))

		# Run backward prop with tape and trainable variables on discriminator
		dis_gradients = dis_tape.gradient(dis_loss, self.discriminator.trainable_variables)
		self.dis_optimizer.apply_gradients(zip(dis_gradients, self.discriminator.trainable_variables))

		# Run backward prop on generator
		gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
		self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

		return {"dis_loss": dis_loss, "gen_loss": gen_loss}


################################################################################################

if __name__ == '__main__':
	# Preprocess data_size portion of the entire image and attribute set
	data_size = 202000
	real_images = tf.divide(tf.cast(real_images[:data_size], dtype=tf.float32) - 127.5, 127.5)
	attributes = tf.cast(attributes[:data_size], dtype=tf.float32)

	# Get validation data to send to model training
	num_images = batch_size
	use_images = real_images.numpy()
	use_attributes = attributes.numpy()
	temp_y = np.random.randint(2, size=5)
	temp_y = tf.reshape(temp_y, (1, 5))
	idx = np.all(use_attributes == temp_y, axis=1)
	use_images = use_images[idx][:num_images]
	use_images = scale_images(use_images, (299,299,3))
	use_images = preprocess_input(use_images)
	use_attributes = use_attributes[idx][:num_images]

	# Create cGAN model that shapes actual inputs to above input shapes
	model = getModel(lr=0.0002)
	history = model.fit((real_images, attributes), batch_size=batch_size, epochs=EPOCHS, callbacks=[CustomCallback(validation_data=(use_images, use_attributes))])

	# Generate random image
	z_input = np.random.rand(100)
	z_input = tf.cast(tf.reshape(z_input, (1, 1, 1, 100)), dtype=tf.float32)
	y_input = attributes[0]
	y_input = tf.cast(tf.reshape(y_input, (1, 1, 1, 5)), dtype=tf.float32)
	gen_images = model.generator((z_input, y_input))
	gen_image = (gen_images[0] + 1) / 2.0
	plt.imshow(gen_image)
	plt.savefig("Final Image.png")
	plt.show()

	model.generator.save("trained_generator")




