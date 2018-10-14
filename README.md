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
    * [adv_attack] Cleverhans (Refer to https://github.com/tensorflow/cleverhans)
    * [Required]  pickle

- [Feedforward Steps] 
    * Command `python Getkernel.py`, getting convolutional layers kernels
    * Command `python Getfeature.py`, getting feature after convolution
    * Command `python Getweight.py`, getting fully connected layers kernels and training accuracy
    * Command `python mnist_test.py` or `python test.py`, getting test accuracy

#### Adversarial attack
- [Function]
    * BP/ff models are provided for cifar10 and mnist dataset under folder `dataset_structre_model`
    * Models can be trained from scratch if no filename is not specified
    * By changing adversarial attack methods, different algorithms can be tested
    * Refer to `show_sample.ipynb` to visualize generated adversarial samples

- [Usage]
    * `python cifar_keras.py -train_dir cifar_BP_model -filename cifar.ckpt -method FGSM`
    * `python cifar_keras.py -train_dir cifar_ff_model -filename FF_init_model.ckpt -method BIM`

```
cifar_keras.py:
  --batch_size: Size of training batches
    (default: '128')
    (an integer)
  --filename: Checkpoint filename.
    (default: 'FF_init_model.ckpt')
  --learning_rate: Learning rate for training
    (default: '0.001')
    (a number)
  --[no]load_model: Load saved model or train.
    (default: 'true')
  --method: Adversarial attack method
    (default: 'FGSM')
  --nb_epochs: Number of epochs to train model
    (default: '40')
    (an integer)
  --train_dir: Directory where to save model.
    (default: 'cifar_ff_model')
```

### Contact me
Jiali Duan (Email: jialidua@usc.edu)<br>
Min Zhang (Email: zhan980@usc.edu)
