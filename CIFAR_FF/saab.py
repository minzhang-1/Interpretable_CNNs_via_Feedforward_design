import numpy as np
from skimage.util.shape import view_as_windows

from sklearn.decomposition import PCA
from numpy import linalg as LA
from skimage.measure import block_reduce

import matplotlib.pyplot as plt

def parse_list_string(list_string):
	"""Convert the class string to list."""
	elem_groups=list_string.split(",")
	results=[]
	for group in elem_groups:
		term=group.split("-")
		if len(term)==1:
	  		results.append(int(term[0]))
		else:
	  		start=int(term[0])
	  		end=int(term[1])
	  		results+=range(start, end+1)
	return results

# convert responses to patches representation
def window_process(samples, kernel_size, stride):
	'''
	Create patches
	:param samples: [num_samples, feature_height, feature_width, feature_channel]
	:param kernel_size: int i.e. patch size
	:param stride: int
	:return patches: flattened, [num_samples, output_h, output_w, feature_channel*kernel_size^2]

	'''
	n, h, w, c = samples.shape
	output_h = (h - kernel_size)//stride + 1
	output_w = (w - kernel_size)//stride + 1
	patches = view_as_windows(np.ascontiguousarray(samples), (1, kernel_size, kernel_size, c), step=(1, stride, stride, c))
	patches = patches.reshape(n, output_h, output_w, c*kernel_size*kernel_size)
	return patches

def remove_mean(features, axis):
	'''
	Remove the dataset mean.
	:param features [num_samples,...]
	:param axis the axis to compute mean
	
	'''
	feature_mean=np.mean(features,axis=axis,keepdims=True)
	feature_remove_mean=features-feature_mean
	return feature_remove_mean,feature_mean

def select_balanced_subset(images, labels, use_num_images, use_classes):
	'''
	select equal number of images from each classes
	'''
	# Shuffle
	num_total=images.shape[0]
	shuffle_idx=np.random.permutation(num_total)
	images=images[shuffle_idx]
	labels=labels[shuffle_idx]

	num_class=len(use_classes)
	num_per_class=int(use_num_images/num_class)
	selected_images=np.zeros((use_num_images,images.shape[1],images.shape[2],images.shape[3]))
	selected_labels=np.zeros(use_num_images)
	for i in range(num_class):
		# images_in_class=images[labels==i]
		idx=(labels==i)
		index=np.where(idx==True)[0]
		images_in_class=np.zeros((index.shape[0],images.shape[1],images.shape[2],images.shape[3]))
		for j in range(index.shape[0]):
			images_in_class[j]=images[index[j]]
		selected_images[i*num_per_class:(i+1)*num_per_class]=images_in_class[:num_per_class]
		selected_labels[i*num_per_class:(i+1)*num_per_class]=np.ones((num_per_class))*i

	# Shuffle again
	shuffle_idx=np.random.permutation(num_per_class*num_class)
	selected_images=selected_images[shuffle_idx]
	selected_labels=selected_labels[shuffle_idx]
	# For test
	# print(selected_images.shape)
	# print(selected_labels[0:10])
	# plt.figure()
	# for i in range (10):
	# 	img=selected_images[i,:,:,0]
	# 	plt.imshow(img)
	# 	plt.show()
	return selected_images,selected_labels

def remove_zero_patch(samples):
	std_var=(np.std(samples, axis=1)).reshape(-1,1)
	ind_bool=(std_var==0)
	ind=np.where(ind_bool==True)[0]
	print('zero patch shape:',ind.shape)
	samples_new=np.delete(samples,ind,0)
	return samples_new

def find_kernels_pca(sample_patches, num_kernels, energy_percent):
	'''
	Do the PCA based on the provided samples.
	If num_kernels is not set, will use energy_percent.
	If neither is set, will preserve all kernels.

	:param samples: [num_samples, feature_dimension]
	:param num_kernels: num kernels to be preserved
	:param energy_percent: the percent of energy to be preserved
	:return: kernels, sample_mean
	'''
	# Remove patch mean
	sample_patches_centered, dc=remove_mean(sample_patches, axis=1)
	sample_patches_centered=remove_zero_patch(sample_patches_centered)
    # Remove feature mean (Set E(X)=0 for each dimension)
	training_data, feature_expectation=remove_mean(sample_patches_centered, axis=0)

	pca=PCA(n_components=training_data.shape[1], svd_solver='full',whiten=True)
	pca.fit(training_data)

	# fig=plt.figure(1)
	# x=np.arange(1,training_data.shape[1]+1)
	# y=np.log(pca.explained_variance_)
	# y=np.delete(y,y.shape[0]-1,0)
	# num_channels=sample_patches.shape[-1]
	# largest_ev=[np.var(dc*np.sqrt(num_channels))]
	# print('Largest eigenvalues:',largest_ev)
	# var=np.log(largest_ev)
	# y=np.concatenate((var, y), axis=0)
	# plt.plot(x,y,'bo-')
	# plt.xlabel('components number')
	# plt.ylabel('log of energy')
	# plt.title('PCA energy')
	# plt.savefig('PCA_energy_5.png')
	# plt.show()

	# Compute the number of kernels corresponding to preserved energy
	if  energy_percent:
		energy=np.cumsum(pca.explained_variance_ratio_)
		num_components=np.sum(energy<energy_percent)+1
	else:
		num_components=num_kernels

	kernels=pca.components_[:num_components,:]
	mean=pca.mean_

	num_channels=sample_patches.shape[-1]
	largest_ev=[np.var(dc*np.sqrt(num_channels))]
	dc_kernel=1/np.sqrt(num_channels)*np.ones((1,num_channels))/np.sqrt(largest_ev)
	kernels=np.concatenate((dc_kernel, kernels), axis=0)

	print("Num of kernels: %d"%num_components)
	print("Energy percent: %f"%np.cumsum(pca.explained_variance_ratio_)[num_components-1])
	return kernels, mean

def multi_Saab_transform(images, labels, kernel_sizes, num_kernels, energy_percent, use_num_images, use_classes):
	'''
	Do the PCA "training".
	:param images: [num_images, height, width, channel]
	:param labels: [num_images]
	:param kernel_sizes: list, kernel size for each stage,
	       the length defines how many stages conducted
	:param num_kernels: list the number of kernels for each stage,
	       the length should be equal to kernel_sizes.
	:param energy_percent: the energy percent to be kept in all PCA stages.
	       if num_kernels is set, energy_percent will be ignored.
    :param use_num_images: use a subset of train images
    :param use_classes: the classes of train images
    return: pca_params: PCA kernels and mean
    '''

	num_total_images=images.shape[0]
	if use_num_images<num_total_images and use_num_images>0:
		sample_images, selected_labels=select_balanced_subset(images, labels, use_num_images, use_classes)
	else:
		sample_images=images
	# sample_images=images
	num_samples=sample_images.shape[0]
	num_layers=len(kernel_sizes)
	pca_params={}
	pca_params['num_layers']=num_layers
	pca_params['kernel_size']=kernel_sizes

	for i in range(num_layers):
		print('--------stage %d --------'%i)
    	# Create patches
		sample_patches=window_process(sample_images,kernel_sizes[i],1) # overlapping
		h=sample_patches.shape[1]
		w=sample_patches.shape[2]
    	# Flatten
		sample_patches=sample_patches.reshape([-1, sample_patches.shape[-1]])

    	# Compute PCA kernel
		if not num_kernels is None:
			num_kernel=num_kernels[i]
		kernels, mean=find_kernels_pca(sample_patches, num_kernel, energy_percent)
		
		num_channels=sample_patches.shape[-1]
		if i==0:
			# Transform to get data for the next stage
			transformed=np.matmul(sample_patches, np.transpose(kernels))
		else:
	    	# Compute bias term
			bias=LA.norm(sample_patches, axis=1)
			bias=np.max(bias)
			pca_params['Layer_%d/bias'%i]=bias
			# Add bias
			sample_patches_centered_w_bias=sample_patches+1/np.sqrt(num_channels)*bias
			# Transform to get data for the next stage
			transformed=np.matmul(sample_patches_centered_w_bias, np.transpose(kernels))
	    	# Remove bias
			e=np.zeros((1, kernels.shape[0]))
			e[0,0]=1
			transformed-=bias*e

    	# Reshape: place back as a 4-D feature map
		sample_images=transformed.reshape(num_samples, h, w,-1)

		# Maxpooling
		sample_images=block_reduce(sample_images, (1,2,2,1), np.max)


		print('Sample patches shape after flatten:', sample_patches.shape)
		print('Kernel shape:', kernels.shape)
		print('Transformed shape:', transformed.shape)
		print('Sample images shape:', sample_images.shape)
    	
		pca_params['Layer_%d/kernel'%i]=kernels
		pca_params['Layer_%d/pca_mean'%i]=mean

	return pca_params

# Initialize
def initialize(sample_images, pca_params):

	num_layers=pca_params['num_layers']
	kernel_sizes=pca_params['kernel_size']

	for i in range(num_layers):
		print('--------stage %d --------'%i)
		# Extract parameters
		kernels=pca_params['Layer_%d/kernel'%i]
		mean=pca_params['Layer_%d/pca_mean'%i]

    	# Create patches
		sample_patches=window_process(sample_images,kernel_sizes[i],1) # overlapping
		h=sample_patches.shape[1]
		w=sample_patches.shape[2]
    	# Flatten
		sample_patches=sample_patches.reshape([-1, sample_patches.shape[-1]])

		num_channels=sample_patches.shape[-1]
		if i==0:
			# Transform to get data for the next stage
			transformed=np.matmul(sample_patches, np.transpose(kernels))
		else:
			bias=pca_params['Layer_%d/bias'%i]
			# Add bias
			sample_patches_centered_w_bias=sample_patches+1/np.sqrt(num_channels)*bias
			# Transform to get data for the next stage
			transformed=np.matmul(sample_patches_centered_w_bias, np.transpose(kernels))
	    	# Remove bias
			e=np.zeros((1, kernels.shape[0]))
			e[0,0]=1
			transformed-=bias*e
    	
    	# Reshape: place back as a 4-D feature map
		num_samples=sample_images.shape[0]
		sample_images=transformed.reshape(num_samples, h, w,-1)

		# Maxpooling
		sample_images=block_reduce(sample_images, (1,2,2,1), np.max)

		print('Sample patches shape after flatten:', sample_patches.shape)
		print('Kernel shape:', kernels.shape)
		print('Transformed shape:', transformed.shape)
		print('Sample images shape:', sample_images.shape)
	return sample_images




