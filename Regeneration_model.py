"""
The railroad surface
Regneration model

H. Kim and S. Han, 
"Regeneration of a defective Railroad Surface for defect detection with Deep Convolution Neural Networks," Journal of Internet Computing and Services, vol. 21, no. 6, pp. 23-31, 2020.
DOI: 10.7472/jksii.2020.21.6.23.

"""
import os, time
import numpy as np
from datetime import datetime as date

# tensorflow
import tensorflow as tf
# future warning disable
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# keras
from keras.models import Model, load_model
from keras.layers import Input
from keras import losses
from keras import backend as K

# util about data
from load_data import *
from load_data import _load_data


# PATH difine
PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PRAMS_SAVE_DIR = os.path.join(PATH, 'GAN_MODEL_LOG')
GENERATED_IMAGE_DIR = os.path.join(PATH, 'GAN_GENERATED_test')
# directory that saves generated image each run times
NOW_DIR = os.path.join(GENERATED_IMAGE_DIR, date.now().strftime('%dday_%Hhour_%Mmin_'))
#NOW_DIR = os.path.join(GENERATED_IMAGE_DIR, 'NOW')
# makedir
if not os.path.exists(MODEL_PRAMS_SAVE_DIR):
	os.mkdir(MODEL_PRAMS_SAVE_DIR)
if not os.path.exists(GENERATED_IMAGE_DIR):
	os.mkdir(GENERATED_IMAGE_DIR)
"""
if not os.path.exists(NOW_DIR):
	os.mkdir(NOW_DIR)
"""
def preprocess(data:dict, rescale = False, normalize = False, center = False, 
	only_target = False, ratio = 1, target = 'None', shuffle = True, concate = True,
	equal_to_contrast = False):
	"""
		data : data dictionary
		 - dict keys : 'train_img', 'train_label', 'test_img', 'test_label'
		rescale : data range set to 0~1
		target : 'background' or 'defect' or 'None'
	"""
	# concatenation train, test
	if concate:
		images = np.concatenate((data['train_img'], data['test_img']), axis = 0)
		labels = np.concatenate((data['train_label'], data['test_label']), axis = 0)
	else:
		images = data['train_img']
		labels = data['train_label']

	if target != 'None' and only_target:
		images, labels = find_contain_target(images, labels, target)
		print('only contain to target, ', target)
		
	if not only_target:
		# if only_target was false, (background : defect)ratio set to same ratio or func's argument ratio.
		defect_img, defect_label = find_contain_target(images, labels, 'defect')
		n_defect = len(defect_img)
		use_n_image = int(n_defect * ratio)
		back_img, back_label = find_contain_target(images, labels, 'background')
		n_back_images = len(back_img)

		if n_back_images < (n_defect + use_n_image):
			raise ValueError('ratio error')

		random_index = np.random.choice((len(back_img)-use_n_image), 1)[0]
		back_img = back_img[random_index:random_index+use_n_image]
		back_label = back_label[random_index:random_index+use_n_image]

		images = np.concatenate((defect_img, back_img), axis = 0)
		labels = np.concatenate((defect_label, back_label), axis = 0)

	if shuffle:
		images, labels, _ = shuffle_with_sameindex_img_label(images, labels)

	if equal_to_contrast:
		images = equal_contrast_all(images)

	#save_img('origin', np.expand_dims(images[:10], axis = -1), name = '_original_', rescale = False)
	images_mean = 0
	images_std = 1
	images_center = 0
	if rescale:
		images = images / 255
	if normalize:
		images_mean = np.mean(images)
		images_std = np.std(images)
		images = (images - images_mean) / images_std
		print(' >>> images mean, std : ', images_mean, images_std)
		if center:
			if rescale:
				images_center = 0.5
			else:
				images_center = 255/2
			images += images_center
	
	images = np.expand_dims(images, axis = -1)
	labels = labeling(labels, axis=-1)

	print(' >> images set shape : ', images.shape)
	print(' >> labels set shape : ', labels.shape)
	#labels.astype(np.float32)
	return images, labels, images_mean, images_std, images_center

def postprocess(data, mean, std, center_v, rescaling = False, normalize = False, center = False):
	if center:
		data = data - center_v
	if normalize:
		data = (data*std)+mean
	if rescaling:
		x_max = np.max(data)
		x_min = np.min(data)
		data = (x_max - x_min) * ((data - np.min(data))/ (np.max(data)-np.min(data))) + x_min
	return data

def save_img(data_type:str, data:np.ndarray, path = '', name = '_generated_', rescale = False, mode = 'L'):
	try:
		img_name = os.path.join(path, data_type + name)
	except:
		img_name = os.path.join(NOW_DIR, data_type + name)

	length = data.shape[0]
	data = np.squeeze(data, axis = 3)
	for i in range(length):
		save_name = img_name + str(i)
		img_save(data[i], save_name, rescale, mode)

class GAN:
	def __init__(self, type_name, images, images_mean, images_std, labels, style_img, now, model_params,
		rescale = False, normalize = False, center = False, epochs = 1001, batch_size = 64, learning_rate = 2e-4, dilated = 2, patches = 7):
		# images save dir
		self.now = now
		self.model_params = model_params
		# data
		self.name = type_name
		self.data = images
		self.data_size = self.data.shape
		self.set_data_dims = 1
		self.labels = labels 
		self.style_img = style_img
		self.images_mean = images_mean
		self.images_std = images_std
		# training params
		self.epochs = epochs
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.rescale = rescale
		self.normalize = normalize
		self.center = center
		self.dilation_rate = (dilated, dilated)
		self.dilation_double = (int(dilated), int(dilated))
		self.patch_size = patches
		# Input tensor
		self.image_tensor = Input(shape=(self.data_size[1], self.data_size[2], self.set_data_dims,))
		self.label_tensor = Input(shape=(self.data_size[1], self.data_size[2], 2,))
		self.back_tensor = Input(shape=(self.data_size[1], self.data_size[2], 2,))
		self.style_tensor = Input(shape=(self.data_size[1], self.data_size[2], self.set_data_dims,))
		self.noise_tensor = Input(shape=(int(self.data_size[1]/4), int(self.data_size[2]/4), self.set_data_dims,))
		self.real_labels = Input(shape=(int(self.data_size[1]/4-self.patch_size+1), int(self.data_size[2]/4-self.patch_size+1), self.set_data_dims,))
		self.fake_labels = Input(shape=(int(self.data_size[1]/4-self.patch_size+1), int(self.data_size[2]/4-self.patch_size+1), self.set_data_dims,))
		# model, variable initialize
		self.variable_init()
		self.model_init()
		# trainable variable
		self.G, self.D = self.trainable_list_set()
		# model saver
		self.saver = tf.train.Saver(self.G)

	def variable_init(self, initializer = 'basic'):
		if initializer == 'basic':
			self.weight_init = tf.random_normal_initializer(0, 0.02)
		elif initializer == 'glorot' or 'xavier':
			self.weight_init = tf.glorot_uniform_initializer()
		else:
			self.weight_init = tf.glorot_uniform_initializer()
		self.bias_init = tf.zeros_initializer()

		# # of mini-batch
		self.n_batch = int(self.data_size[0] / self.batch_size)
		print(' # of batch : ', self.n_batch+1)
		# parameters
		self.filter_min = 32
		self.filter_size = 3
		self.G_var = {
			'W1' : tf.get_variable('g_w1', [self.filter_size, self.filter_size, int(self.filter_min*4), 2],
				dtype=tf.float32, initializer=self.weight_init),
			'BN1_gamma' : tf.get_variable('g_bn1_g', [1], initializer = tf.ones_initializer()),
			'BN1_beta' : tf.get_variable('g_bn1_b', [1], initializer = tf.zeros_initializer()),		
			'W2_res' : tf.get_variable('g_w2_res', [3, 3, int(self.filter_min*2), 2],
				dtype=tf.float32, initializer=self.weight_init),
			'W2_1' : tf.get_variable('g_w2_1', [self.filter_size, self.filter_size, int(self.filter_min*4), int(self.filter_min*2)],
				dtype=tf.float32, initializer=self.weight_init),
			'W2_2' : tf.get_variable('g_w2_2', [5, 5, int(self.filter_min*4), int(self.filter_min*2)],
				dtype=tf.float32, initializer=self.weight_init),
			'BN2_gamma' : tf.get_variable('g_bn2_g', [1], initializer = tf.ones_initializer()),
			'BN2_beta' : tf.get_variable('g_bn2_b', [1], initializer = tf.zeros_initializer()),	
			'W3' : tf.get_variable('g_w3', [self.filter_size, self.filter_size, self.filter_min, int(self.filter_min*2)],
				dtype=tf.float32, initializer=self.weight_init),
			'BN3_gamma' : tf.get_variable('g_bn3_g', [1], initializer = tf.ones_initializer()),
			'BN3_beta' : tf.get_variable('g_bn3_b', [1], initializer = tf.zeros_initializer()),	
			'W4_res' : tf.get_variable('g_w4_res', [3, 3, 1, int(self.filter_min*2)],
				dtype=tf.float32, initializer=self.weight_init),
			'W4_1' : tf.get_variable('g_w4_1', [self.filter_size, self.filter_size, self.filter_min, 1],
				dtype=tf.float32, initializer=self.weight_init),
			'W4_2' : tf.get_variable('g_w4_2', [5, 5, self.filter_min, 1],
				dtype=tf.float32, initializer=self.weight_init),
			'B1' : tf.get_variable('g_b1', [self.batch_size, int(self.data_size[1]/2), int(self.data_size[2]/2), int(self.filter_min*4)], initializer=self.bias_init),
			'B2' : tf.get_variable('g_b2', [self.batch_size, int(self.data_size[1]/2), int(self.data_size[2]/2), int(self.filter_min*2)], initializer=self.bias_init),
			'B3' : tf.get_variable('g_b3', [self.batch_size, int(self.data_size[1]), int(self.data_size[2]), self.filter_min], initializer=self.bias_init),
			'B4' : tf.get_variable('g_b4', [self.batch_size, int(self.data_size[1]), int(self.data_size[2]), 1], initializer=self.bias_init),
			}
		self.D_var = {
			'W1' : tf.get_variable('d_w1', [self.filter_size, self.filter_size, 1, self.filter_min],
				dtype=tf.float32, initializer=self.weight_init),
			'W2' : tf.get_variable('d_w2', [self.filter_size, self.filter_size, self.filter_min, int(self.filter_min*2)],
				dtype=tf.float32, initializer=self.weight_init),
			'BN2_gamma' : tf.get_variable('d_bn2_g', [1], initializer = tf.ones_initializer()),
			'BN2_beta' : tf.get_variable('d_bn2_b', [1], initializer = tf.zeros_initializer()),
			'W3' : tf.get_variable('d_w3', [self.filter_size, self.filter_size, int(self.filter_min*2), int(self.filter_min*4)],
				dtype=tf.float32, initializer=self.weight_init),
			'BN3_gamma' : tf.get_variable('d_bn3_g', [1], initializer = tf.ones_initializer()),
			'BN3_beta' : tf.get_variable('d_bn3_b', [1], initializer = tf.zeros_initializer()),		
			'W4' : tf.get_variable('d_w4', [self.filter_size, self.filter_size, int(self.filter_min*4), 2],
				dtype=tf.float32, initializer=self.weight_init),
			'BN4_gamma' : tf.get_variable('d_bn4_g', [1], initializer = tf.ones_initializer()),
			'BN4_beta' : tf.get_variable('d_bn4_b', [1], initializer = tf.zeros_initializer()),
			'W5' : tf.get_variable('d_w5', [self.patch_size, self.patch_size, 2, 1], dtype=tf.float32, initializer=self.weight_init),
			'B1' : tf.get_variable('d_b1', [self.filter_min], initializer=self.bias_init),
			'B2' : tf.get_variable('d_b2', [int(self.filter_min*2)], initializer=self.bias_init),
			'B3' : tf.get_variable('d_b3', [int(self.filter_min*4)], initializer=self.bias_init),
			'B4' : tf.get_variable('d_b4', [2], initializer=self.bias_init),
			'B5' : tf.get_variable('d_b5', [1], initializer=self.bias_init)
			}
		self.style_def()

	def style_def(self):
		style = np.zeros((self.batch_size, self.data_size[1], self.data_size[2]))
		style_n = self.style_img.shape[0]-1
		style_indices = 0
		for i in range(len(style)):
			style[i] = self.style_img[style_indices]
			style_indices += 1
			if style_indices > style_n:
				style_indices = 0
		if self.rescale:
			style = style/255
		if self.normalize:
			style = (style - np.mean(style)) / np.std(style)
			if self.center:
				if self.rescale:
					style = style + 0.5
				else:
					style = style + (255/2)

		self.style = np.expand_dims(style, axis = -1)

	def trainable_list_set(self):
		G_list = []
		D_list = []
		for G_val in self.G_var.values():
			G_list.append(G_val)
		for D_val in self.D_var.values():
			D_list.append(D_val)
		return G_list, D_list

	def resize_image(self, data, ratio = 4):
		resized = tf.image.resize_images(data, size = (int(self.data_size[1]/ratio), int(self.data_size[2]/ratio)))

		return resized

	def minmax_rescale(self, x, x_min = 0, x_max = 255):
		x_max = np.max(self.data)
		x_min = np.min(self.data)
		y = (x_max - x_min) * ((x - K.min(x))/ (K.max(x)-K.min(x))) + x_min
		return y 
		
	def generator_net(self, inputs, labels):
		""" 
		z = X | L
		z1 = conv^-1(z)
		z2 = bn(z1)
		z3 = relu(z2) 
		...
		y = rescale_as_image(z3)
		"""
		inputs = inputs * self.resize_image(labels)
		conv1 = K.conv2d_transpose(inputs, self.G_var['W1'], 
			output_shape = (self.batch_size, int(self.data_size[1]/2), int(self.data_size[2]/2), int(self.filter_min*4)),
			strides = (2,2), padding = 'same') + self.G_var['B1']
		bn1 = tf.nn.batch_normalization(conv1, K.mean(conv1), K.var(conv1),
			self.G_var['BN1_beta'], self.G_var['BN1_gamma'], variance_epsilon = 5e-5)
		relu1 = K.relu(bn1)
		
		conv2_res = K.conv2d_transpose(inputs, self.G_var['W2_res'], 
			output_shape = (self.batch_size, int(self.data_size[1]/2), int(self.data_size[2]/2), int(self.filter_min*2)),
			strides = (2,2), padding = 'same')
		
		conv2 = (K.conv2d(relu1, self.G_var['W2_1'], padding = 'same') + \
			K.conv2d(relu1, self.G_var['W2_2'], padding = 'same')) + self.G_var['B2']
		bn2 = tf.nn.batch_normalization(conv2, K.mean(conv2), K.var(conv2),
			self.G_var['BN2_beta'], self.G_var['BN2_gamma'], variance_epsilon = 5e-5)
		relu2 = K.relu(bn2) + conv2_res

		conv3 = K.conv2d_transpose(relu2, self.G_var['W3'], 
			output_shape = (self.batch_size, int(self.data_size[1]), int(self.data_size[2]), self.filter_min),
			strides = (2,2), padding = 'same') + self.G_var['B3']
		bn3 = tf.nn.batch_normalization(conv3, K.mean(conv3), K.var(conv3),
			self.G_var['BN3_beta'], self.G_var['BN3_gamma'], variance_epsilon = 5e-5)
		relu3 = K.relu(bn3)
		
		conv4_res = K.conv2d_transpose(relu2, self.G_var['W4_res'],
			output_shape = (self.batch_size, int(self.data_size[1]), int(self.data_size[2]), 1),
			strides = (2,2), padding = 'same')
		
		conv4 = (K.conv2d(relu3, self.G_var['W4_1'], padding = 'same') + \
			K.conv2d(relu3, self.G_var['W4_2'], padding = 'same')) + self.G_var['B4']
		y = K.relu(conv4) + conv4_res
		y = K.tanh(y)
		y = self.minmax_rescale(y)
		#y = conv4

		return y

	def discriminator_net(self, inputs):
		"""
		z = conv(x)
		z1 = bn(z)
		z2 = lerelu(z1) 
		...
		y = sigmoid(dot(z2, w))
		"""
		conv1 = K.conv2d(inputs, self.D_var['W1'], padding = 'same') + self.D_var['B1']
		lrelu1 = tf.nn.leaky_relu(conv1)

		conv2 = K.conv2d(lrelu1, self.D_var['W2'], padding = 'same', strides = (2,2)) + self.D_var['B2']
		bn2 = tf.nn.batch_normalization(conv2, K.mean(conv2), K.var(conv2),
			self.D_var['BN2_beta'], self.D_var['BN2_gamma'], variance_epsilon = 5e-5)
		lrelu2 = tf.nn.leaky_relu(bn2)
		
		conv3 = K.conv2d(lrelu2, self.D_var['W3'], padding = 'same') + self.D_var['B3']
		bn3 = tf.nn.batch_normalization(conv3, K.mean(conv3), K.var(conv3),
			self.D_var['BN3_beta'], self.D_var['BN3_gamma'], variance_epsilon = 5e-5)
		lrelu3 = tf.nn.leaky_relu(bn3)
		
		conv4 = K.conv2d(lrelu3, self.D_var['W4'], padding = 'same', strides = (2,2)) + self.D_var['B4']
		bn4 = tf.nn.batch_normalization(conv4, K.mean(conv4), K.var(conv4),
			self.D_var['BN4_beta'], self.D_var['BN4_gamma'], variance_epsilon = 5e-5)
		y = tf.nn.leaky_relu(bn4)
		#y = K.sigmoid(K.batch_flatten(y))
		#y = K.sigmoid(K.dot(K.batch_flatten(y), self.D_var['W5']) + self.D_var['B5'])
		y = K.sigmoid(K.conv2d(y, self.D_var['W5']) + self.D_var['B5'])

		return y

	def model_init(self):
		self.generator = self.generator_net(self.noise_tensor, self.label_tensor)
		self.real = self.discriminator_net(self.image_tensor)
		self.fake = self.discriminator_net(self.generator)
		#self.style_generator = self.generator_net(self.noise_tensor, self.back_tensor)
		self.style_real = self.discriminator_net(self.style_tensor)
		#self.style_fake = self.discriminator_net(self.style_generator)
		print(' >> discriminator output shape : ', self.fake.get_shape())

	def create_noise_normal(self, mean = 0.5, stddev = 0.1, calculated = False, save = False):
		if calculated:
			data_mean = np.mean(self.data)
			stddev = np.std(self.data)
			print(' >> data mean : ', data_mean , ' | data stddev : ', stddev)
		else:
			data_mean = mean
			stddev = stddev

		noise = np.random.normal(data_mean, stddev,
			size = (self.data_size[0], int(self.data_size[1]/4), int(self.data_size[2]/4), self.set_data_dims))
		array_save_to_pickle(noise, 'trained_noise', self.model_params)
		if save:
			# save to digital images file. (.png)
			save_img('test', noise[:10], name = '_created_noise_', rescale = self.rescale)
	
		return noise

	def mini_batch(self, batch_index, x, y, z):
		if batch_index == self.n_batch:
			true_img = x[-self.batch_size:]
			true_labels = y[-self.batch_size:]
			fake_noise = z[-self.batch_size:]
		else:
			true_img = x[0 + (batch_index*self.batch_size): ((batch_index + 1) * self.batch_size)]
			true_labels = y[0 + (batch_index*self.batch_size): ((batch_index + 1) * self.batch_size)]
			fake_noise = z[0 + (batch_index*self.batch_size): ((batch_index + 1) * self.batch_size)]
			
		return true_img, true_labels, fake_noise

	def training(self, test_save = False, style_alpha = 0.7, recovery_beta = 0.7, smooth_labels = 0.1, \
			model_params = 'exp'):
		# Loss func
		"""
		- another loss function considered

		origin_discriminator_loss = tf.reduce_mean(
			(1/2)*losses.mean_squared_error(self.fake_labels, self.fake) + \
			(1/2)*losses.mean_squared_error(self.real_labels, self.real))
		origin_generator_loss = tf.reduce_mean(
			(1/2)*K.mean(losses.mean_squared_error(self.real_labels, self.fake), axis = (1,2)) + \
			(1/2)*K.mean(K.abs(self.generator - self.image_tensor), axis = (1,2,3)))

		style_discirminator_loss = tf.reduce_mean(
			(1/2)*losses.mean_squared_error(self.fake_labels, self.style_fake) + \
			(1/2)*losses.mean_squared_error(self.real_labels, self.style_real))
		style_generator_loss = tf.reduce_mean(
			(1/2)*K.mean(losses.mean_squared_error(self.real_labels, self.style_fake), axis = (1,2)) + \
			(1/2)*K.mean(K.abs(self.style_generator - self.style_tensor), axis = (1,2,3)))

		discriminator_loss = style_alpha*style_discirminator_loss + recovery_beta*origin_discriminator_loss
		generator_loss = style_alpha*style_generator_loss + recovery_beta*origin_generator_loss
		"""
		"""
		discriminator_loss = tf.reduce_mean(
			losses.mean_squared_error(self.fake_labels, self.fake) + \
			(style_alpha*losses.mean_squared_error(self.real_labels, self.style_real)
				+ recovery_beta*losses.mean_squared_error(self.real_labels, self.real)))
		generator_loss = tf.reduce_mean(
			K.mean(losses.mean_squared_error(self.real_labels, self.fake), axis = (1,2)) + \
			(K.mean(style_alpha*K.abs(self.generator - self.style_tensor), axis = (1,2,3))
				+ K.mean(recovery_beta*K.abs(self.generator - self.image_tensor), axis = (1,2,3))))
		"""
		
		fake_loss = losses.mean_squared_error(self.fake_labels, self.fake)
		real_style_loss = losses.mean_squared_error(self.real_labels, self.style_real)
		real_origin_loss = losses.mean_squared_error(self.real_labels, self.real)
		discriminator_loss = tf.reduce_mean((1/2)*fake_loss + (1/2)*K.mean(style_alpha*real_style_loss+recovery_beta*real_origin_loss))

		noise_real_loss = losses.mean_squared_error(self.real_labels, self.fake)
		to_style_loss = K.abs(self.generator - self.style_tensor)
		to_origin_loss = K.abs(self.generator - self.image_tensor)
		generator_loss = tf.reduce_mean((1/2)*noise_real_loss + (1/2)*K.mean(style_alpha*to_style_loss + recovery_beta*to_origin_loss))
		
		# optimizer
		discriminator_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(discriminator_loss, var_list = self.D)
		generator_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(generator_loss, var_list = self.G)
		
		init = tf.global_variables_initializer()
		noise = self.create_noise_normal(calculated = False, mean = 0, stddev = 1)
		"""
		real_labels = np.ones((self.batch_size, int(int(self.data_size[1]/4) * int(self.data_size[2]/4)))) - 0.1
		fake_labels = np.zeros((self.batch_size, int(int(self.data_size[1]/4) * int(self.data_size[2]/4)))) + 0.1
		"""
		real_labels = np.ones((self.batch_size, int(self.data_size[1]/4 - self.patch_size+1), int(self.data_size[2]/4 - self.patch_size+1), 1)) - smooth_labels
		fake_labels = np.zeros((self.batch_size, int(self.data_size[1]/4 - self.patch_size+1), int(self.data_size[2]/4 -self.patch_size+1), 1))
		
		discriminator_loss_list = []
		generator_loss_list = []

		print(' >>> style info, ', np.max(self.style), np.min(self.style), np.mean(self.style))
		print(' >>> image info, ', np.max(self.data), np.min(self.data), np.mean(self.data))

		with tf.Session() as sess:
			sess.run(init)
			print('# :', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
			for epoch in range(self.epochs):
				epoch_time = 0
				discriminator_loss_sum = 0
				generator_loss_sum = 0

				for batch_index in range(self.n_batch+1):
					start_time = time.time()

					true_img, true_labels, fake_noise = self.mini_batch(batch_index, self.data, self.labels, noise)
					#background_labels = np.zeros_like(true_labels)
					#background_labels[:,:,:,0] = true_labels[:,:,:,0]
					style_batch = self.style * np.expand_dims(true_labels[:,:,:,0], axis = -1)
					#style_batch = self.style

					_, d_loss_val = sess.run([discriminator_op, discriminator_loss],
						feed_dict = {self.image_tensor:true_img,
							self.label_tensor:true_labels, self.noise_tensor:fake_noise,
							self.real_labels:real_labels, self.fake_labels:fake_labels,
							self.style_tensor:style_batch})
					discriminator_loss_sum += d_loss_val

					_, g_loss_val = sess.run([generator_op, generator_loss],
						feed_dict = {self.image_tensor:true_img, 
							self.label_tensor:true_labels, self.noise_tensor:fake_noise,
							self.real_labels:real_labels,
							self.style_tensor:style_batch})
					generator_loss_sum += g_loss_val

					run_times = time.time() - start_time
					epoch_time += run_times
					print(' -->> %i / %i epochs | %i/%i | discriminator loss : %f, generator_loss : %f | %0.3fs'
						%(epoch + 1, self.epochs, batch_index*self.batch_size, \
							self.data_size[0], d_loss_val, g_loss_val, epoch_time), end = '\r')
					
				print('')

				discriminator_loss_list.append(discriminator_loss_sum/(self.n_batch+1))
				generator_loss_list.append(generator_loss_sum/(self.n_batch+1))

				if test_save and epoch%1000 == 0:
					self.saver.save(sess, self.model_params + "/GAN.ckpt", epoch)
					"""
					generated_list = []
					generated = sess.run(self.generator, feed_dict = {self.label_tensor:true_labels, self.noise_tensor:fake_noise})
					generated_list.extend(generated)
					gen = np.asarray(generated_list)
					if not self.rescale:
						gen = gen*self.images_std + self.images_mean
					#print(gen.shape)
					save_img('test', gen[:2], name = '_test_generated_' + str(epoch) + '_', rescale = self.rescale)
					"""
			# save params
			if not test_save:
				self.saver.save(sess, self.model_params + "/GAN.ckpt", epoch)

		loss_dict = {'epochs' : self.epochs, 'bs' : self.batch_size, 'lr' : self.learning_rate,
			'G_loss' : generator_loss_list, 'D_loss' : discriminator_loss_list}
		with open(os.path.join(self.now, 'TRAINING_LOGS.txt'), 'w') as f:
			f.write(str(loss_dict))

	def make_label(self, path, name, zero_label, number_of = 64, save = False):
		# if zero_label True, all label set to zero array, i.e. in case, none defect.
		path = os.path.join(path, self.name)
		if zero_label:
			data = np.zeros((number_of, self.data_size[1], self.data_size[2], 1))
		else:
			data = load_custom_label(path, name)
			data = data / 255
			data = data[:number_of]
		if save:
			save_img(self.name, np.expand_dims(data, axis = -1), name = '_custom_labels_', rescale = self.rescale)
		#data = data+1
		return data

	def generate(self, data, noise_indices = 0, n_data = -1, custom_label = True, zero_label = False, 
			save_labels = False, at_epoch = '/GAN.ckpt'):
		"""
		custom_label true : make custom label
			label_set false : make to zeros
			else : make custom_label
		"""
		noise = load_from_pickle('trained_noise', self.model_params)
		
		if not n_data == -1:
			self.n_batch = int((n_data-1)/self.batch_size)
			print(' # of minibatchs', self.n_batch)
		
		generated_list = []
		if custom_label:
			if n_data == -1:
				n_data = data.shape[0]
				#print(n_data)
			labels = self.make_label(path = 'labels_sample', name = 'custom_labels_', number_of = n_data, zero_label = zero_label, save = save_labels) #ex. camoflagu
			labels = labeling(labels, axis = -1)
			print(' >> loaded labels shape, ', labels.shape )
			if len(labels) > len(noise):
				raise ValueError('n_data > len(noise)')		
		else:
			labels = self.labels
		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)
			try:
				model_params = os.path.join(self.model_params, at_epoch)
				self.saver.restore(sess, model_params)
			except:
				ckpt = tf.train.latest_checkpoint(self.model_params)
				self.saver.restore(sess, ckpt)
			print(" -> feature generate.")
			for batch_index in range(self.n_batch + 1): 
				_, label_batch, noise_batch = self.mini_batch(batch_index, data, labels, noise)
				generated = sess.run(self.generator, feed_dict = {self.label_tensor:label_batch, self.noise_tensor:noise_batch})
				generated_list.extend(generated)
		print(' => gened # of ',len(generated_list))
		return np.asarray(generated_list)

	def generate_repeat_from_customLabels(self, data, n_data=-1, zero_label = False, at_epoch = '/GAN.ckpt'):
		loaded_labels = self.make_label(path = 'labels_sample', name = 'custom_labels_', number_of = n_data, zero_label = zero_label)
		loaded_labels = labeling(loaded_labels, axis = -1)
		if n_data == -1:
			n_data = data.shape[0]
			self.n_batch = int((n_data-1)/self.batch_size)
			print(' # of minibatchs', self.n_batch)

		print(' >> loaded labels shape, ', loaded_labels.shape )
		noise = load_from_pickle('trained_noise', self.model_params)
		splitpoint = (self.n_batch+1) * self.batch_size
		#print(splitpoint)
		repeat = int(n_data/splitpoint)
		if repeat < 1:
			repeat = 1
		print('generation ', repeat, ' times.' )
		generated_list = []
		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)
			try:
				model_params = os.path.join(self.model_params, at_epoch)
				self.saver.restore(sess, model_params)
			except:
				ckpt = tf.train.latest_checkpoint(self.model_params)
				self.saver.restore(sess, ckpt)
			print(" -> feature generate.")
			for re in range(repeat):
				labels = loaded_labels[0+int(re*splitpoint):int((re+1)*splitpoint)]
				for batch_index in range(self.n_batch + 1): 
					_, label_batch, noise_batch = self.mini_batch(batch_index, data, labels, noise)
					generated = sess.run(self.generator, feed_dict = {self.label_tensor:label_batch, self.noise_tensor:noise_batch})
					generated_list.extend(generated)
		print(' => gened # of ',len(generated_list))
		return np.asarray(generated_list)

def GAN_run(data_type = 'type1', style_alpha = 1, recovery_beta = 1, epochs = 4001, custom_name = 'exp1_', style = 'type1'):
	# 0. data prepare (selected by contain target or All)
	
	#dataset = load_dataset()
	subset_path = os.path.join('RSDDs_dataset', 'sub_samples_pickles') 
	dataset = load_dataset(subset_path, subsets = True)
	concat = False
	#dataset = load_dataset()
	if data_type == 'type1':
		data = dataset['type1']
		n_gen = 1024
		only_target = True
		center = False
	elif data_type == 'type2':
		print('type2, require eliminated 3pixel of width..')
		data = type2_data_process(dataset['type2'])
		n_gen = 1152
		only_target = True
		center = False
	else:
		raise ValueError(' type is \'type1\' or \'type2\' ')
	rescale = True
	normalize = False
	equal_to_contrast = False
	

	# 1. preprocess
	# if need only specific target, only target = True, target : 'background' or 'defect'
	images, labels, images_mean, images_std, images_center = preprocess(data, rescale = rescale, center = center, normalize = normalize, ratio = 1,
	 	only_target = only_target, target = 'defect', concate = concat, equal_to_contrast=equal_to_contrast)
	style_path = os.path.join('gan_style', style)
	style_img = list_to_arr(_load_data(style_path))
	print('>>> style sample # : ', len(style_img))

	# 2. generate adversarial network
	now = NOW_DIR + '_' + custom_name
	if not os.path.exists(now):
		os.mkdir(now)
	model_params = os.path.join(MODEL_PRAMS_SAVE_DIR, custom_name)
	if not os.path.exists(model_params):
		os.mkdir(model_params)
	GAN_model = GAN(data_type, images, images_mean, images_std, labels, style_img, now = now,
		rescale = rescale, normalize = normalize, center = center, epochs = epochs,
		batch_size = 16, learning_rate = 2e-4, dilated = 2, patches = 7, model_params = model_params)
	
	generate_epoch = 2000
	generate_epoch_path = "GAN.ckpt-" + str(generate_epoch)
	
	GAN_model.training(test_save = True, style_alpha = style_alpha,
	 recovery_beta = recovery_beta, smooth_labels = 0.1)
	
	generated = GAN_model.generate(data = images, zero_label = False, custom_label = False, n_data = -1, at_epoch = generate_epoch_path) 
	generated = postprocess(generated, images_mean, images_std, images_center, normalize = normalize, center = center)
	save_img(data_type, generated, name = '_generated_to_origin_'+custom_name, rescale = rescale, path = now)
	
	generated_1 = GAN_model.generate(data = images, zero_label = False, custom_label = True, n_data = -1, save_labels = True, at_epoch = generate_epoch_path) # custom label 64
	generated_1 = postprocess(generated_1, images_mean, images_std, images_center, normalize = normalize, center = center)
	save_img(data_type, generated_1, name = '_generated_to_custom_'+custom_name, rescale = rescale, path = now)
		
	image_origin = postprocess(images, images_mean, images_std, images_center, center = center, normalize = normalize)
	save_img(data_type, image_origin, name = '_original_'+custom_name, rescale = rescale, path = now)
	#save_img(data_type, generated, name = '_generated_', rescale = rescale)
	#save_img(data_type, generated[:100], name = '_generated_', rescale = rescale)
	#save_img(data_type, np.expand_dims(labels[:,:,:,1], axis = -1), name = '_labels_', rescale = rescale, path = now)
	"""
	generated = GAN_model.generate_repeat_from_customLabels(data=images, zero_label = False, n_data = n_gen, at_epoch = generate_epoch_path)
	save_img(data_type, generated, name = '_generated_for_repeat', rescale = rescale, path = now)
	"""

def GAN_run_for_test(data_type = 'type1'):
	# 1. data prepare(selected for test) & preprocessing(rescale to 0~1)
	rescale = True

	images, labels, index, nd_img = selected_dataset(data_type = data_type, rescale = rescale, defect_ratio = 1)
	data = {'train_img':images, 'train_label': labels, 'test_img':nd_img}

	# 2. generate adversarial network
	GAN_model = GAN(data_type, images, labels, now = NOW_DIR,
		rescale = rescale, epochs = 101, batch_size = 32, learning_rate = 2e-4)
	#GAN_model.training(test_save = False)
	generated = GAN_model.generate(data = nd_img, noise_indices = index, 
		noise_indexing = True, label_set = True, custom_label = True)
	save_img(data_type, generated[index], name = '_generated_', rescale = rescale)
	#save_img(data_type, labels[index], name = '_generated_L', rescale = rescale)

if __name__ == "__main__":
	#GAN_run(data_type = 'type1')
	GAN_run(data_type = 'type1', style_alpha = 1, recovery_beta = 1, epochs = 2001, custom_name = 'other_1style', style = 'type1')
	K.clear_session()
	"""
	GAN_run(data_type = 'type1', style_alpha = 0, recovery_beta = 1, epochs = 4001, custom_name = 'original_style')
	K.clear_session()
	GAN_run(data_type = 'type1', style_alpha = 1, recovery_beta = 1, epochs = 4001, custom_name = 'unnormal_style', style = 'custom_style')
	K.clear_session()
	"""
	#GAN_run_for_test(data_type = 'type1')

	#K.clear_session()






