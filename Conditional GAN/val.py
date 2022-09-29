import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from math import floor
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from scipy.linalg import sqrtm
from skimage.transform import resize

real_images = np.load('./FinalProg_Spring22/images.npy')
attributes = np.load('./FinalProg_Spring22/attributes5.npy')

# calculate frechet inception distance
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

# assumes images have the shape 299x299x3, pixels in [0,255]
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

# scale an array of images to a new size
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
		dis_optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
		gen_optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
		loss='binary_crossentropy'
	)
	return model

def plotImages(generator, y_input, real_images, run_num="1"):
	z_input = np.random.rand(16*100)
	z_input = tf.cast(tf.reshape(z_input, (16, 1, 1, 100)), dtype=tf.float32)

	predictions = generator((z_input, y_input))

	fig = plt.figure(figsize=(4, 4))

	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, i+1)
		plt.imshow((predictions[i] + 1) / 2.0)
		plt.axis('off')

	plt.savefig("gen_images" + run_num + ".png")
	plt.show()

	predictions = predictions.numpy()
	predictions = scale_images(predictions, (299,299,3))
	predictions = preprocess_input(predictions)

	use_images = scale_images(real_images, (299,299,3))
	use_images = preprocess_input(use_images)

	model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

	print("\nFID:", calculate_fid(model, predictions, use_images), "IS:", calculate_inception_score(predictions, n_split=4))

def doExperiment(generator):
	# Create random attribute and load set of matching real images
	y_rand = np.random.randint(2, size=5)
	y_rand = tf.reshape(y_rand, (1, 5))
	print(y_rand)
	use_images = real_images[np.all(attributes == y_rand, axis=1)][:16]
	y_rand = tf.repeat(y_rand, repeats=16, axis=0)
	y_rand = tf.cast(tf.reshape(y_rand, (16, 1, 1, 5)), dtype=tf.float32)

	plotImages(generator, y_rand, use_images)

	# Repeat two more times
	y_rand = np.random.randint(2, size=5)
	y_rand = tf.reshape(y_rand, (1, 5))
	print(y_rand)
	use_images = real_images[np.all(attributes == y_rand, axis=1)][:16]
	y_rand = tf.repeat(y_rand, repeats=16, axis=0)
	y_rand = tf.cast(tf.reshape(y_rand, (16, 1, 1, 5)), dtype=tf.float32)

	plotImages(generator, y_rand, use_images, run_num="2")

	y_rand = np.random.randint(2, size=5)
	y_rand = tf.reshape(y_rand, (1, 5))
	print(y_rand)
	use_images = real_images[np.all(attributes == y_rand, axis=1)][:16]
	y_rand = tf.repeat(y_rand, repeats=16, axis=0)
	y_rand = tf.cast(tf.reshape(y_rand, (16, 1, 1, 5)), dtype=tf.float32)

	plotImages(generator, y_rand, use_images, run_num="3")


def doVal(generator, y_label):
	num_images = 1000

	# d = 100 for z random noise
	z_input = np.random.rand(num_images * 100)
	z_input = tf.cast(tf.reshape(z_input, (num_images, 1, 1, 100)), dtype=tf.float32)
	y_input = tf.reshape(y_label, (1, 5))
	y_input = tf.repeat(y_label, repeats=num_images, axis=0)
	y_input = tf.cast(tf.reshape(y_input, (num_images, 1, 1, 5)), dtype=tf.float32)
	gen_images = generator((z_input, y_input))
	gen_images = (gen_images + 1) / 2.0

	fig = plt.figure(figsize=(10, 10))

	for i in range(100):
		plt.subplot(10, 10, i+1)
		plt.imshow(gen_images[i])
		plt.axis('off')

	plt.savefig('val_images.png')
	plt.show(block=False)
	plt.pause(1)
	plt.close()

	print("Getting FID and IS scores...")

	gen_images = gen_images.numpy()
	gen_images = scale_images(gen_images, (299,299,3))
	gen_images = preprocess_input(gen_images)

	use_images = real_images[np.all(attributes == y_label, axis=1)][:num_images]
	use_images = scale_images(use_images, (299,299,3))
	use_images = preprocess_input(use_images)

	model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

	print("FID:", calculate_fid(model, gen_images, use_images), "IS:", calculate_inception_score(gen_images))


# Get generator model
generator = getModel(load_file="./models/trained_generator")

# Model evaluation requirement for project report
doExperiment(generator)

# Val.py requirement
# y label input is a numpy array of shape (5,)
y_label = np.random.randint(2, size=5)

doVal(generator, y_label)







