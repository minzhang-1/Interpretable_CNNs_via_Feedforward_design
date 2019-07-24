# Interpretable_CNN

This part contains the code for adversarial attack in the paper **Interpretable Convolutional Neural Networks via Feed Forward Design**,
maintained by Jiali Duan and Min Zhang.

### Table of Content

- [Requirements]
    * Python3, keras, tensorflow, cleverhans (Refer to https://github.com/tensorflow/cleverhans), pickle

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