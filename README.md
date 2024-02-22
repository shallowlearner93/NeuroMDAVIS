# NeuroMDAVIS
Maitra, C., Seal, D.B., Das, V. and De, R.K., 2024. **NeuroMDAVIS: Visualization of single-cell multi-omics data under deep learning framework**

## Architecture of NeuroMDAVIS. 
![Figure1](https://github.com/shallowlearner93/NeuroMDAVIS/assets/113589317/5cb6d1ad-84e3-4a6d-9fbd-babac912a85b)

**Figure 1**: he NeuroMDAVIS network architecture developed for visualization of high-dimensional multi-omics datasets.

## Requirements
To run **NeuroMDAVIS**, one needs to install `numpy`, `pandas`, `sklearn`, `scipy`, and `tensorflow` packages. Installation codes are as follows:
+ `pip install numpy`
+ `pip install pandas`
+ `pip install scikit-learn`
+ `pip install scipy`
+ `pip install tensorflow`

## Running NeuroMDAVIS.
To run **NeuroMDAVIS**, import `NeuroMDAVIS.py` from the `Scripts` directory and run the function `NeuroMDAVIS`. All the parameters are mentioned below for better understanding. One can also follow a `*_NeuroMDAVIS.ipynb` file from the folders: `Notebooks`.

### Parameters 
All input parameters are as follows: `X`, `dim`, `lambda_act`, `lambda_weight`, `num_neuron`, `bs`, `epoch`, `sd`, `verbose`
+ `X`: List of input data matrices for training. [The input data matrices should be in the form of cells x features.]
+ `dim`: Dimension onto which the data is being projected. 
+ `lambda_act`: Activity regularizer parameter. 
+ `lambda_weight`: Kernel regularizer parameter.
+ `num_neuron`: List of neurons in the shared hidden layer and modality specific hidden layer (modality wise). 
+ `bs`: Training batch size.
+ `epoch`: Total number of iteration for training.
+ `sd`: To reproduce the results set a seed value.
+ `verbose`: 1 for printing the output.


### Code to run NeuroMDAVIS
To run **NeuroMDAVIS**, one needs to import the script NeuroMDAVIS (within the `Scripts` directory) first. An example is provided below. Let `x1` [cells x features] and `x2` [cells x features] be two training datasets, coming from two different omics modalities.
```
X = [x1, x2]
dim = 2
lambda_act = 0
lambda_weight = 0
num_neuron = [64, [128, 16]]
bs=128
epoch=500
sd=0
verbose=1

X_embedding = MV.NeuroMDAVIS(X, dim, lambda_act, lambda_weight, num_neuron, bs, epoch, sd, verbose)
```
# Dataset Source
--------------
The datasets used in this work can be downloaded from the following link.
https://zenodo.org/records/10623932

