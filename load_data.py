import os, os.path
import pickle
import numpy as np
from PIL import Image
import collections
import pandas as pd

"""
RDSSs_datasets, Rail road
published by :
Jinrui Gan, Q.Y. Li , et al. A Hierarchical Extractor-Based Visual Rail Surface Inspection System.
IEEE Sensors Journal, 2017, 17(23):7935-7944.
'http://icn.bjtu.edu.cn/Visint/resources/RSDDs.aspx'
"""
ROOT = 'RSDDs_dataset'
TYPE1 = 'Type-I_RSDDs_dataset'
TYPE2 = 'Type-II_RSDDs_dataset'

PATH = os.path.dirname(os.path.abspath(__file__))
LABELS = 'GroundTruth'
IMAGES = 'Rail_surface_images'
PICKLE_ROOT = os.path.join(ROOT, 'pickle_batch')
PREDICT = os.path.join(ROOT, 'predict')
if not os.path.exists(PICKLE_ROOT):
	os.mkdir(PICKLE_ROOT)
if not os.path.exists(PREDICT):
	os.mkdir(PREDICT)

def merge_path(x, y):
	data_path = os.path.join(ROOT, x, y)
	if not os.path.exists(data_path):
		raise ValueError(" data path not exist.")
	return data_path

def _load_data(path:os.path):
	data = []
	file_list = os.listdir(path)
	file_list.sort()
	for filename in file_list:
		with Image.open(os.path.join(path, filename)) as img:
			images = np.asarray(img)
			data.append(images)
	return data

def load_data_all():
	##	return 4 attributes of list
	type1_img = _load_data(merge_path(TYPE1, IMAGES))
	type1_label = _load_data(merge_path(TYPE1, LABELS))
	type2_img = _load_data(merge_path(TYPE2, IMAGES))
	type2_label = _load_data(merge_path(TYPE2, LABELS))
	return type1_img, type1_label, type2_img, type2_label

def _load_data_with_sort(path:os.path, name):
	data = []
	for i in range(len(os.listdir(path))):
		filename = name + str(i) + '.png'
		with Image.open(os.path.join(path, filename)) as img:
			image = np.asarray(img)
			data.append(image)
	return data

def check_data(path:os.path):
	columns = list['name', 'size']
	arr_list = []
	for filename in os.listdir(path):
		with Image.open(os.path.join(path, filename)) as img:
			arr = np.asarray(img)
			arr_list.append(filename, arr.shape)
	df = pd.DataFrame(arr_list, columns = columns)
	df['size'].value_counts()
	return df

def label_masking(label_list, threshold = 127):
	length = len(label_list)
	for i in range(length):
		print("> %i / %i masking.."%(i+1, length), end= '\r')
		ones = np.ones(label_list[i].shape, dtype = np.uint8)
		mask = (label_list[i] > threshold) * ones
		label_list[i] = mask
	print('\n')
	return label_list

## specific process.
def type1_makeup(type1_img, type1_label, v1 = 100, v2 = 160, masking = True):
	"""
		v1, v2 : desired output image size
		return : 
			newimg list, newlabel list
	"""
	if masking:
		type1_label = label_masking(type1_label)

	## elimination some images with 1282 * y to 1280 * y
	for i in range(len(type1_img)):
		if type1_img[i].shape[0] > 1280:
			type1_img[i] = type1_img[i][:1280, :]
			#newarr = type1_img[i][:1280, :]
			#type1_img[i] = newarr

	newimg = []
	newlabel = []
	for i in range(len(type1_img)):
		print(' > expanding type1 data.. %i / %i '%(i+1, len(type1_img)), end = '\r')
		x = type1_img[i].shape[0]
		if x > v1:
			n_part = int(x / v1)
			for j in range(n_part):
				newimg.append(type1_img[i][(j * v1):v1 + (j * v1),:])
				newlabel.append(type1_label[i][(j * v1):v1 + (j * v1), :])
		else:
			newimg.append(type1_img[i])
			newlabel.append(type1_label[i])
	print('\n-> type1 expanded shape : (%i %i %i)'%(len(newimg), v1, v2), end = '\n')
	return newimg, newlabel

def type2_makeup(type2_img, type2_label, v1 = 100, v2 = 55, masking = True):
	if masking:
		type2_label = label_masking(type2_label)

	newimg = []
	newlabel = []
	for i in range(len(type2_img)):
		print(' > expanding type2 data.. %i / %i '%(i+1, len(type2_img)), end = '\r')
		x = type2_img[i].shape[0]
		if x > v1:
			n_part = int(x / v1)
			for j in range(n_part):
				newimg.append(type2_img[i][(j * v1): v1 + (j * v1),:])
				newlabel.append(type2_label[i][(j * v1): v1 + (j * v1), :])
		else:
			newimg.append(type2_img[i])
			newlabel.append(type2_label[i])
	print('\n-> type2 expanded shape : (%i %i %i)'%(len(newimg), v1, v2), end = '\n')
	return newimg, newlabel

def list_to_arr(arr_list:list, arr_shape_check = False):
	# array shape in the list is must same.
	number, v1, v2 = len(arr_list), arr_list[0].shape[0], arr_list[0].shape[1]
	#print(' > list to array to (%i %i %i)'%(number, v1, v2))
	if arr_shape_check:
		for i in range(len(arr_list)):
			if arr_list[i].shape[0] != v1 or arr_list[i].shape[1] != v2:
				print("array shape in the list not same.")
				return
	newarr = np.stack(arr_list)
	return newarr

def shuffle_with_sameindex_img_label(images, labels, memorial_n = 0):
	"""
		input : list 
		memorial_n : how many to store index(from end)
		output : array
	"""
	length = len(images)
	if memorial_n > length:
		raise ValueError('..')
	reIndex = np.random.permutation(length)
	replaced_images = np.zeros_like(images)
	replaced_labels = np.zeros_like(labels)
	for i in range(length):
		replaced_images[reIndex[i]] = images[i]
		replaced_labels[reIndex[i]] = labels[i]
	return replaced_images, replaced_labels, reIndex[length-memorial_n:]

def split_train_test(images, labels, train_ratio = 0.9, shuffle = True):
	"""
		images : all images array
		labels : all ground truth array
		train_ratio : train sets and test sets separation ratio
		shuffle : doing shuffle data(to array)
		return : separated sets dict(to array)
	"""
	length = len(images)
	train_num = int(length * train_ratio)
	#test_num = length - train_num

	if shuffle:
		images, labels, _ = shuffle_with_sameindex_img_label(images, labels)
	else:
		images = list_to_arr(images)
		labels = list_to_arr(labels)

	return {'train_img':images[:train_num], 'train_label':labels[:train_num],
	'test_img':images[train_num:], 'test_label':labels[train_num:]}

def pickle_save(data:dict, pickle_root, batch = 2):
	if not os.path.exists(pickle_root):
		os.mkdir(pickle_root)
	for t in ('type1', 'type2'):
		for key in ('train_img', 'train_label', 'test_img', 'test_label'): 
			batch_size = int(data[t][key].shape[0] / batch)
			for b in range(batch):
				pickle_name = t +"-" + key + '_' + str(b+1) + '.pkl'
				with open(os.path.join(pickle_root, pickle_name), 'wb') as pkl:
					if b == (batch-1):
						pickle.dump(data[t][key][b * batch_size :], pkl, -1)
					else:
						pickle.dump(data[t][key][(b*batch_size) : ((b+1)*batch_size)], pkl, -1)
			print(' > saved ' + pickle_name)

def array_save_to_pickle(arr:np.ndarray, file_name, root = 'pickle_file'):
	file_name = file_name + '.pkl'
	if not os.path.exists(root):
		os.mkdir(root)
	with open(os.path.join(root, file_name), 'wb') as pkl:
		pickle.dump(arr, pkl, -1)

def load_from_pickle(file_name, root = 'pickle_file'):
	try:
		file_name = file_name + '.pkl'
		with open(os.path.join(root, file_name), 'rb') as pkl:
			data = pickle.load(pkl)
		return data
	except:
		raise ValueError('errorrororroro')


def all_load(save_pickle = True, shuffle = True, train_ratio = 0.9):
	"""
		dictionary structure : 
		dict - type1 - train_img
					 - train_label
					 - test_img
					 - test_label
			 - type2 - train_img
			 		 - train_label
			 		 - test_img
			 		 - test_label
		return : dict
	"""
	type1_img, type1_label, type2_img, type2_label = load_data_all()
	type1_imgs, type1_labels = type1_makeup(type1_img, type1_label, v1 = 100, v2 = 160, masking = True)
	type2_imgs, type2_labels = type2_makeup(type2_img, type2_label, v1 = 100, v2 = 55, masking = True)
	type1 = split_train_test(type1_imgs, type1_labels, train_ratio = train_ratio, shuffle = shuffle)
	type2 = split_train_test(type2_imgs, type2_labels, train_ratio = train_ratio, shuffle = shuffle)
	if shuffle:
		print(' > dataset is shuffled.')

	data = collections.defaultdict(dict)
	data['type1']['train_img'] = type1['train_img']
	data['type1']['train_label'] = type1['train_label']
	data['type1']['test_img'] = type1['test_img']
	data['type1']['test_label'] =type1['test_label']
	data['type2']['train_img'] = type2['train_img']
	data['type2']['train_label'] = type2['train_label']
	data['type2']['test_img'] = type2['test_img']
	data['type2']['test_label'] = type2['test_label']

	if save_pickle:
		pickle_save(data, PICKLE_ROOT)

	return data

def load_dataset(pickle_root = None, used_pickle = True, batch = 2, subsets = False):
	dataset = collections.defaultdict(dict)
	if used_pickle:
		if pickle_root == None:
			pickle_root = PICKLE_ROOT
		pickle_file = os.path.join(pickle_root, 'type1-train_img_1.pkl')
		print(pickle_root)
		if not os.path.exists(pickle_file):
			if subsets:
				dataset = subsets_makeup(save_pickle = True, trainingratio = 0.5)
			else:
				dataset = all_load()
		else:
			n_index = int(batch * 8)
			index = 0
			for t in ('type1', 'type2'):
				for key in ('train_img', 'train_label', 'test_img', 'test_label'):
					datastack = []
					for b in range(batch):
						index += 1
						pickle_name = t +"-" + key + '_' + str(b+1) + '.pkl'
						with open(os.path.join(pickle_root, pickle_name), 'rb') as pkl:
							data = pickle.load(pkl)
							datastack.append(data)
						dataset[t][key] = np.concatenate(datastack)
						print(' > reading pickle files %i/%i '%(index, n_index), end ='\r')
			print('\n')
	else:
		if subsets:
			dataset = subsets_makeup(save_pickle = False, trainingratio = 0.5)
		else:
			dataset = all_load(save_pickle = False)
		print(' loading new dataset from direct-root.')

	return dataset


def flatten(array:np.ndarray):
	n, v1, v2 = array.shape
	return array.reshape([-1, v1*v2])

def expand_ch3(array:np.ndarray):
	return np.stack([array, array, array], axis=3)

def labeling(array:np.ndarray, axis:int):
	ones = np.ones(array.shape, dtype = np.uint8)
	target = (array == 1) * ones
	background = (array == 0) * ones
	return np.stack([background, target], axis=axis)

def expand_all(data):
	X_train, y_train = expand_ch3(data['train_img']), labeling(data['train_label'], axis=3)
	X_test, y_test = expand_ch3(data['test_img']), labeling(data['test_label'], axis=3)
	return X_train, y_train, X_test, y_test

def img_save(array:np.ndarray, name, rescale = True, mode = None, root = 'custom'):
	img_name = name + '.png'
	if rescale:
		image = Image.fromarray(np.uint8(array * 255), mode)
	else:
		image = Image.fromarray(np.uint8(array), mode)
	if root == None:
		image.save(os.path.join(PREDICT, img_name))
	elif root == 'custom':
		image.save(img_name)
	else:
		if not os.path.exists(root):
			os.mkdir(root)
		image.save(os.path.join(root, img_name))


def get_only_target():
	"""
		only use before preprocessing!!
	"""
	type1_img, type1_label, type2_img, type2_label = load_data_all()
	type1_imgs, type1_labels = type1_makeup(type1_img, type1_label, v1 = 100, v2 = 160, masking = True)
	type2_imgs, type2_labels = type2_makeup(type2_img, type2_label, v1 = 100, v2 = 55, masking = True)

	new_type1_imgs, new_type1_labels = find_contain_target(type1_imgs, type1_labels)
	new_type2_imgs, new_type2_labels = find_contain_target(type2_imgs, type2_labels)

	return {'type1_img' : new_type1_imgs, 'type1_label' : new_type1_labels,
		'type2_img':new_type2_imgs, 'type2_label':new_type2_labels}

def find_contain_target(X, y, target = 'defect'):
	"""
		X, y : list
	"""
	data_len = len(X)

	newarr_img_list = []
	newarr_label_list = []

	for i in range(data_len):
		non_zero_count = np.count_nonzero(y[i])
		if target == 'defect':
			if not non_zero_count == 0:
				# then X[i] is contain target
				newarr_img_list.append(X[i])
				newarr_label_list.append(y[i])
		elif target == 'background':
			if non_zero_count == 0:
				newarr_img_list.append(X[i])
				newarr_label_list.append(y[i])
		else:
			raise ValueError('target missing')

	newarr_img = list_to_arr(newarr_img_list)
	newarr_label = list_to_arr(newarr_label_list)
	print(' contain target : ', target, ' image shape : ', newarr_img.shape, end = '\n')
	return newarr_img, newarr_label

def get_defect_ratio(X, y, percentage = False):
	data_len = len(X)
	defect_ratio_arr = np.zeros(data_len)

	for i in range(data_len):
		full_size = y[i].size
		non_zero_count = np.count_nonzero(y[i])
		defect_ratio = non_zero_count/full_size
		if percentage:
			defect_ratio *= 100
		defect_ratio_arr[i] = defect_ratio
	return defect_ratio_arr

def get_images_from_gan(data, img_path, labels_path, data_type = 'type1', labels_dir = 'predict'):
	labels_dir = os.path.join(labels_path, 'predict')

	gened_images = list_to_arr(_load_data(img_path))
	gened_labels = list_to_arr(_load_data(labels_dir))

	print(gened_images.shape)
	print(gened_labels.shape)

	data['train_img'] = np.vstack((data['train_img'], gened_images))
	data['train_label'] = np.vstack((data['train_label'], gened_labels))

	print(data['train_img'].shape)
	print(data['train_label'].shape)

	return data

def images_savetofile(img, name, rescale, root = None):
	length = len(img)
	for i in range(length):
		savename = name + str(i) 
		img_save(img[i], savename, rescale, 'L', root)

def defect_save(data_type = 'type1', name = 'defect', concate = True):
	dataset = load_dataset()

	if data_type == 'type1':
		data = dataset['type1']
	elif data_type == 'type2':
		data = dataset['type2']
	else:
		raise ValueError(' type is \'type1\' or \'type2\' ')

	if concate:
		images = np.concatenate((data['train_img'], data['test_img']), axis = 0)
		label = np.concatenate((data['train_label'], data['test_label']), axis = 0)
	else:
		images = data['train_img']
		label = data['train_label']

	X, _ = find_contain_target(images, label, target = 'defect')
	save_name = os.path.join('defects', name)
	images_savetofile(X, save_name, rescale = False)

def load_custom_label(path, name):
	path = os.path.join(PATH, path)
	if not os.path.exists(path):
		raise ValueError(path, 'path not exists.')

	custom_label = np.asarray(_load_data_with_sort(path, name))
	print(' custom label shape : ', custom_label[0].shape)

	if custom_label[0].ndim == 3:
		custom_label = np.asarray(Image.convert('L', custom_label))
	return custom_label

def selected_dataset(train_ratio = 0.9, defect_ratio = 1, rescale = True, \
	shuffle = True, set_defect_ratio = True, data_type = 'type1', used_pickle = True):
	"""
		custom dataset setting.
		+ preprocess(rescale)
	"""
	pkl_file_root = 'selected_dataset'
	if used_pickle:
		try:
			images = load_from_pickle('train_images', pkl_file_root)
			labels = load_from_pickle('train_labels', pkl_file_root)
			index = load_from_pickle('specific_indices', pkl_file_root)
			nd_img = load_from_pickle('not_defect_image_for_test', pkl_file_root)
		except:
			print('pickle file not exist, data prosessing..')
			images, labels, index, nd_img = selected_dataset(train_ratio, defect_ratio, rescale, shuffle, set_defect_ratio, data_type, False)

	else:
		type1_img, type1_label, type2_img, type2_label = load_data_all()
		if data_type == 'type1':
			type1_imgs, type1_labels = type1_makeup(type1_img, type1_label, v1 = 1000, v2 = 160, masking = True)
			type1 = split_train_test(type1_imgs, type1_labels, train_ratio = train_ratio, shuffle = shuffle)
			# again
			train_img, train_label  = type1_makeup(type1['train_img'], type1['train_label'], v1 = 100, masking = False)
			test_img, test_label = type1_makeup(type1['test_img'], type1['test_label'], v1 = 100, masking = False)
			
		elif data_type == 'type2':
			type2_imgs, type2_labels = type2_makeup(type2_img, type2_label, v1 = 1250, v2 = 55, masking = True)
			type2 = split_train_test(type2_imgs, type2_labels, train_ratio = train_ratio, shuffle = shuffle)

			train_img, train_label = type2_makeup(type2['train_img'], type2['train_label'], v1 = 100, masking = False)
			test_img, test_label = type2_makeup(type2['test_img'], type2['test_label'], v1 = 100, masking = False)

		else:
			raise ValueError('..')

		# find not defect surface in sub-set
		nd_img, nd_lb = find_contain_target(test_img, test_label, target = 'background')

		if set_defect_ratio:
			# if only_target was false, (background : defect)ratio set to same ratio or func's argument ratio.
			defect_img, defect_label = find_contain_target(train_img, train_label, 'defect')
			n_defect = len(defect_img)
			use_n_image = int(n_defect * defect_ratio)
			#print('  # of contain target, ', n_defect) 
			back_img, back_label = find_contain_target(train_img, train_label, 'background')
			n_back_images = len(back_img)

			if n_back_images < (n_defect + use_n_image):
				raise ValueError('ratio error')

			random_index = np.random.choice((len(back_img)-use_n_image), 1)[0]
			back_img = back_img[random_index:random_index+use_n_image]
			back_label = back_label[random_index:random_index+use_n_image]

			images = np.concatenate((defect_img, back_img), axis = 0)
			labels = np.concatenate((defect_label, back_label), axis = 0)
		else:
			images = train_img
			labels = train_label

		concat_images = np.concatenate((images, nd_img), axis = 0)
		concat_labels = np.concatenate((labels, nd_lb), axis = 0)

		images, labels, index = shuffle_with_sameindex_img_label(concat_images, concat_labels, memorial_n = nd_img.shape[0]) 

		if rescale:
			images = images / 255
			nd_img = nd_img / 255
		images = np.expand_dims(images, axis = 3)
		labels = np.expand_dims(labels, axis = 3)
		nd_img = np.expand_dims(nd_img, axis = 3)
		print(' >> images set shape : ', images.shape)

		array_save_to_pickle(images, 'train_images', pkl_file_root)
		array_save_to_pickle(labels, 'train_labels', pkl_file_root)
		array_save_to_pickle(index, 'specific_indices', pkl_file_root)
		array_save_to_pickle(nd_img, 'not_defect_image_for_test', pkl_file_root)
		nd_img = np.squeeze(nd_img, axis = -1)
		for i in range(len(nd_img)):
			nd_img_name = 'nd_img' + str(i)
			img_save(nd_img[i], nd_img_name, rescale = True, mode = 'L', root = 'GAN_TEST_SAMPLE')

	return images, labels, index, nd_img

def load_bysubsets(n_subsets = 3, name = 'type1'):
	subset_path = os.path.join(ROOT, 'sub_samples')
	images_list = []
	labels_list = []
	subsets_length = []
	images_len = 0
	for idx in range(n_subsets):
		img_name = name + "_img_subset" + str(idx)
		lbs_name = name + "_lbs_subset" + str(idx)
		images = load_from_pickle(img_name, subset_path)
		labels = load_from_pickle(lbs_name, subset_path)
		images_list.append(images)
		labels_list.append(labels)
		subsets_length.append(len(images))
		images_len += len(images) 
	return images_list, labels_list, subsets_length, images_len

def subsamples_data_init(n_clusters = 3, name = 'type1', trainingratio = 0.1):
	try:
		images_subsets, labels_subsets, subsets_length, images_length = load_bysubsets(n_clusters, name)
		print(' saved pickle exist, loaded by .pkl')
	except:
		print(' saved pickle file not exist, cluster re-structuring..')
		from clustering import run_whole_data
		run_whole_data(n_clusters = n_clusters)
		images_subsets, labels_subsets, subsets_length, images_length = load_bysubsets(n_clusters, name)
	traininglength = int(images_length * trainingratio)
	print(subsets_length, traininglength)
	training_subset_idx = np.argmin(np.abs(np.asarray(subsets_length) - traininglength))
	print(' training indices : ', training_subset_idx, ', ', images_subsets[training_subset_idx].shape[0],' of samples')
	train_images = []
	train_labels = []
	test_images = []
	test_labels = []
	for i in range(n_clusters):
		if not i == training_subset_idx:
			test_images.extend(images_subsets[i])
			test_labels.extend(labels_subsets[i])
	train_images.extend(images_subsets[training_subset_idx])
	train_labels.extend(labels_subsets[training_subset_idx])
	#print('->', type(images_subsets[training_subset_idx]), type([labels_subsets[training_subset_idx]]), type(test_images), type(test_labels))
	return train_images, train_labels, test_images, test_labels

def subsets_makeup(save_pickle = True, trainingratio = 0.5, n_clusters = 3):
	t1_train_img, t1_train_lbl, t1_test_img, t1_test_lbl = subsamples_data_init(n_clusters=n_clusters,name='type1',trainingratio=trainingratio)	
	t1_train_img, t1_train_lbl = type1_makeup(t1_train_img, t1_train_lbl, v1=100,v2=160,masking=True)
	t1_test_img, t1_test_lbl = type1_makeup(t1_test_img, t1_test_lbl, v1=100,v2=160,masking=True)
	t1_train_img, t1_train_lbl, _ = shuffle_with_sameindex_img_label(t1_train_img, t1_train_lbl)

	t2_train_img, t2_train_lbl, t2_test_img, t2_test_lbl = subsamples_data_init(n_clusters=n_clusters,name='type2',trainingratio=trainingratio)
	t2_train_img, t2_train_lbl = type2_makeup(t2_train_img, t2_train_lbl, v1=100,v2=55,masking=True)
	t2_test_img, t2_test_lbl = type2_makeup(t2_test_img, t2_test_lbl, v1=100,v2=55,masking=True)
	t2_train_img, t2_train_lbl, _ = shuffle_with_sameindex_img_label(t2_train_img, t2_train_lbl)

	data = collections.defaultdict(dict)
	data['type1']['train_img'] = t1_train_img
	data['type1']['train_label'] = t1_train_lbl
	data['type1']['test_img'] = list_to_arr(t1_test_img)
	data['type1']['test_label'] = list_to_arr(t1_test_lbl)
	data['type2']['train_img'] = t2_train_img
	data['type2']['train_label'] = t2_train_lbl
	data['type2']['test_img'] = list_to_arr(t2_test_img)
	data['type2']['test_label'] = list_to_arr(t2_test_lbl)

	if save_pickle:
		pickle_save(data, os.path.join(ROOT, 'sub_samples_pickles'))

	return data

def subsets_loaded_generated_images(data_type = 'type1', concate = False, used_generated_for_train = True, used_origin_for_test = False):
	# test sets by pickle
	subset_path = os.path.join('RSDDs_dataset', 'sub_samples_pickles') 
	dataset = load_dataset(subset_path, subsets = True)
	generated_t1img_path = os.path.join('images_sample', data_type)
	generated_t1lbs_path = os.path.join('labels_sample', data_type)
	generated_t2img_path = os.path.join('images_sample', 'type2')
	generated_t2lbs_path = os.path.join('labels_sample', 'type2')
	
	if used_generated_for_train:
		if data_type == 'type1':
			train_images = _load_data_with_sort(generated_t1img_path, 'type1_generated_for_repeat')
			train_labels = _load_data_with_sort(generated_t1lbs_path, 'custom_labels_')
		elif data_type == 'type2':
			train_images = _load_data_with_sort(generated_t2img_path, 'type2_generated_for_repeat')
			train_labels = _load_data_with_sort(generated_t2lbs_path, 'custom_labels_')

		else:
			raise ValueError('data type error')
		if concate:
			train_img = np.concatenate((list_to_arr(train_images), dataset[data_type]['train_img']), axis = 0)
			train_lbl = np.concatenate((list_to_arr(train_labels)/255, dataset[data_type]['train_label']), axis = 0)
		else:
			if used_origin_for_test:
				dataset[data_type]['test_img'] = dataset[data_type]['train_img']
				dataset[data_type]['test_label'] = dataset[data_type]['train_label']
			train_img = list_to_arr(train_images)
			train_lbl = list_to_arr(train_labels)/255
	else:
		train_img = dataset[data_type]['train_img']
		train_lbl = dataset[data_type]['train_label']

	
	train_img, train_lbl, _ = shuffle_with_sameindex_img_label(train_img, train_lbl)
	dataset[data_type]['train_img'] = train_img
	dataset[data_type]['train_label'] = train_lbl

	"""
	t2_train_images = _load_data(generated_t2img_path)
	t2_train_labels = _load_data(generated_t2lbs_path)
	dataset['type2']['train_img'] = list_to_arr(t2_train_images)
	dataset['type2']['train_label'] = list_to_arr(t2_train_labels)
	"""
	print(dataset[data_type]['train_img'].shape, dataset[data_type]['train_label'].shape,
		dataset[data_type]['test_img'].shape, dataset[data_type]['test_label'].shape)
	return dataset[data_type]

def type2_data_process(dataset, width = 52):
	# for even-odd problem
	keys = ['train_img', 'test_img', 'train_label', 'test_label']
	for key in keys:
		dataset[key] = dataset[key][:,:,:width]
	return dataset

def equal_contrast(x):
	hist, bins = np.histogram(x.flatten(), 256, [0,256])
	cdf = hist.cumsum()
	cdf_m = np.ma.masked_equal(cdf, 0)
	cdf_equal = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
	cdf_equal = np.ma.filled(cdf_equal, 0).astype('uint8')
	y = cdf_equal[x]
	return y

def equal_contrast_all(x):
	length = len(x)
	new_arr = np.zeros_like(x, dtype = np.uint8)
	for i in range(length):
		new_arr[i] = equal_contrast(x[i])
	return new_arr





	





