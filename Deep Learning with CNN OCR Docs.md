# Data Collection
Import Library and Datasets
```
# import tensorflow
import tensorflow as tf

# normalization
from tensorflow.keras.utils import normalize

import matplotlib.pyplot as plt
import numpy as np

# import mnist dataset
# contain 28x28 handwritten images digits from 0-9
from tensorflow.keras.datasets import mnist
```

## Load Dataset
```
# load data digit
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# check data train and testing dimension
print(f'Shape of x train : {x_train.shape}')
print(f'Shape of y train : {y_train.shape}')
print(f'Shape of x test : {x_test.shape}')
print(f'Shape of y test : {y_test.shape}')
```

## Load Dataset using Tensorflow Dataset Library if using Pipeline 
```
# import tensorflow dataset library
import tensorflow_datasets as tfds

# load data using tensorflow dataset
(ds_train, ds_test), ds_info = tfds.load(
    name = 'mnist', 
    split=['train', 'test'], 
    shuffle_files=True,
    as_supervised=True, with_info=True
)
```


# Data Transformation
```
# reshape data menjadi [samples][width][heights][channels]
# sesuai dimensi yang diminta layer Conv2D
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
```

# Data Normalization

# Model Building

# Model Training

# Model Evaluation
