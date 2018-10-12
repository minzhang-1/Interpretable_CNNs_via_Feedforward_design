import pickle
import numpy as np
import data
import saab
import keras
import sklearn

def main():
	# load data
	fr=open('pca_params.pkl','rb')  
	pca_params=pickle.load(fr)
	fr.close()

	# read data
	train_images, train_labels, test_images, test_labels, class_list = data.import_data("0-9")
	print('Training image size:', train_images.shape)
	print('Testing_image size:', test_images.shape)
	
	# Training
	print('--------Training--------')
	feature=saab.initialize(train_images, pca_params) 
	feature=feature.reshape(feature.shape[0],-1)
	print("S4 shape:", feature.shape)
	print('--------Finish Feature Extraction subnet--------')
	feat={}
	feat['feature']=feature
	
	# save data
	fw=open('feat.pkl','wb')    
	pickle.dump(feat, fw)    
	fw.close()

if __name__ == '__main__':
	main()
