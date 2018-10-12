# Interpretable_CNNs_via_Feedforward_design
This is an implementation of the paper [Interpretable Convolutional Neural Networks via Feedforward Design](https://arxiv.org/abs/1810.02786),
maintained by Min Zhang and Jiali Duan.<br>
### Table of Content
- [Dataset] ( Hand-written digits classification)
    * [MNIST] ( train set: 60000, 28x28. Downloaded from Keras)
    * [CIFAR10] ( train set: 50000, 32*32. Downloaded from Keras)
- [Installation] (rensorflow, keras, pickle, sklearn and skimage)
    * [Sklearn Installation] Refer to http://scikit-learn.org/stable/install.html)
    * [Skimage Installation] (Refer to http://scikit-image.org/docs/dev/install.html)
    * [Optional: Jupyter Notebook] (Refer to http://jupyter.org/install.html)
- [Feedforward Steps] 
    * Command `python Getkernel.py`, getting convolutional layers kernels
    * Command `python Getfeature.py`, getting feature after convolution
    * Command `python Getweight.py`, getting fully connected layers kernels and training accuracy
    * Command `python mnist_test.py` or `python test.py`, getting test accuracy
  
### Contact me
Jiali Duan (Email: jialidua@usc.edu)<br>
Min Zhang (Email: zhan980@usc.edu)
