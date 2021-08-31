## Data Collection
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
### With MNIST Digits Dataset
Load Dataset
```
# load data digit
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# check data train and testing dimension
print(f'Shape of x train : {x_train.shape}')
print(f'Shape of y train : {y_train.shape}')
print(f'Shape of x test : {x_test.shape}')
print(f'Shape of y test : {y_test.shape}')
```

Load Dataset using Tensorflow Dataset Library if using Pipeline 
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

Display a single Digit
```
from IPython.display import display
import pandas as pd

# check dataset shape and labels
print(f'shape of datasets : {x_train.shape}')
print(f'Labels : {y_train}')

# print single digit of MNIST
single = x_train[0]
print(f'Shape of Single Digits{single.shape}\n')

display(pd.DataFrame(single.reshape(28, 28)))
```

Display as Image
```
import matplotlib.pyplot as plt
%matplotlib inline

# change to choose new digit
digit = 105
digit2 = 106
a = x_train[digit]
b = x_train[digit2]
plt.imshow(a, cmap='gray', interpolation='nearest')
print(f'Image (#{digit}): which is digit {y_train[digit]}')
plt.imshow(b, cmap='gray', interpolation='nearest')
print(f'Image (#{digit2}): which is digit {y_train[digit2]}')
```

Display the Digits
```
# Display as Text
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 5)
print(f'Shape of Dataset : {x_train.shape}')
print(f'Labels : {y_train}\n')

# Single MNIST digit
single = x_train[0]
print(f'Shape for Single : {single.shape}\n')

pd.DataFrame(single.reshape(28, 28))
```
Display the digits with random sampling
```
import random

rows = 6
random_indices = random.sample(range(x_train.shape[0]), rows*rows)

sample_images = x_train[random_indices, :]

plt.clf()

fig, axes = plt.subplots(rows, rows, figsize=(rows, rows),
                         sharex=True, sharey=True)

for i in range(rows*rows):
    subplot_row = i//rows
    subplot_col = i%rows
    ax = axes[subplot_row, subplot_col]

    plottable_image = np.reshape(sample_images[i,:], (28,28))
    ax.imshow(plottable_image, cmap='gray_r')

    ax.set_xbound([0,28])

plt.tight_layout()
plt.show()
```
### With Captchas Dataset
Download dataset from dataset's link
```
# !curl -LO https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip
# !unzip -qq captcha_images_v2.zip
```

to set the data directory, to be load into notebook and save the dataset in images, labels and character as variables
```
# path to the data directory untuk load data ke dalam google colab
data_dir = Path("./captcha_images_v2")

# ambil list dari dataset, terus di simpan di images, labels dan character
# disini gambar akan diurutin, kemudian menggunakan glob untuk load gambar png 
images = sorted(list(map(str, list(data_dir.glob("*.png")))))
# untuk split label dari informasi pada gambar, label ditampung berwujud list
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
# jumlah character individual yang unik dalam bentuk set
characters = set(char for label in labels for char in label)

print(f'Number of Images found : {len(images)}')
print(f'Number of Labels found : {len(labels)}')
print(f'Number of Unique characters : {len(characters)}')
print(f'Characters Present : {characters}')
```

## Data Transformation
```
# reshape data menjadi [samples][width][heights][channels]
# sesuai dimensi yang diminta layer Conv2D
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
```

## Data Normalization
normalize to scale the X_train
```
X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)
```
normalisasi nilai pixel yang asalnya ada di rentang 0 - 255 diubah menjadi 0 - 1
```
# normalisasi input dari 0-255 menjadi 0-1
X_train = X_train / 255
X_test = X_test / 255
```

## Data Preprocessing
### Label Encoding
encoding label dataset menggunakan onehot encoding
```
# onehot encode pada masing-masing label outputnya
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
```
### Downsampling
Menentukan **Faktor Downsample** sebelum dibawa ke **blok convolutional**
- Faktor dimana gambar akan di downsampling oleh blok convolutional
- Dengan terdapat dua blok konvolutional yang setiap bloknya terdapat pooling layer untuk downsample fiturnya dengan faktor 2
- Maka faktor yang di downsampling total akan menjadi 4
```
# ukuran batch untuk training and validation
batch_size = 16

# menentukan dimensi gambar
img_width = 200
img_height = 50

# Faktor yang di downsampling total menjadi 4.
downsample_factor = 4

# length maximum dari tiap captcha pada dataset
max_length = max([len(label) for label in labels])
```
melakukan proses encoding dan decoding untuk mapping menggunakan `StringLookup`
- diawali dengan proses mapping character ke integers karena integernya mau dipake buat tiap karakter 
- menentukan character yang mau diperoleh dari kosakata nya
- kemudian **split data** jadi **set training** dan **set validasi**
```
# mapping characters ke integers
char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary = list(characters), mask_token = None
)

# integers di mapping kembali ke original characters
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary = char_to_num.get_vocabulary(), mask_token = None, invert = True
)

# function untuk split data
def split_data(images, labels, train_size=0.9, shuffle=True):
    # Get the total size of the dataset
    size = len(images)
    # Buat array indices sebagia index, terus diacak pake shuffle if diperlukan
    indices = np.arange(size)
    if shuffle:
        # shuffle tiap datanya di split sesuai index nya, dari np.random
        np.random.shuffle(indices)
    # Get the size of training samples
    train_samples = int(size * train_size)
    # Split data jadi training 90 % dan validasi untuk 10 %
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid

# jalanin function buat split data jadi train dan validasi
x_train, x_valid, y_train, y_valid = split_data(images = np.array(images), 
                                                labels = np.array(labels))

# function buat encode tiap sampel gambar
def encode_single_sample(img_path, label):
    # read file gambarnya
    img = tf.io.read_file(img_path)
    # decode terus di konversi ke grayscale
    img = tf.io.decode_png(img, channels=1)
    # konversi tipe data gambar ke float 32 di numpy
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize ukuran image sesuai kebutuhan
    img = tf.image.resize(img, [img_height, img_width])
    # transpose gambar, supaya waktu dimensinya sesuai width image nya
    img = tf.transpose(img, perm=[1, 0, 2])
    # map characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # return modelnya dalam bentuk dict, sebagai espect inputnya
    return {"image": img, "label": label}
```
## Data Visualization
menampilkan tiap gambar captchas dari seluruh data gambar melalui visualisasi matplotlib dengan masing-masing labelnya
```
_, ax = plt.subplots(4, 4, figsize=(10, 5))
# batch sebagai index untuk tiap gambarnya
for batch in train_dataset.take(1):
    # gambar ama label
    images = batch['image']
    labels = batch['label']
    # index tiap array hingga 16 gambar
    for i in range(16):
        img = (images[i] * 255).numpy().astype('uint8')
        label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode('utf-8')
        ax[i//4, i%4].imshow(img[:, :, 0].T, cmap='gray')
        ax[i//4, i%4].set_title(label)
        ax[i//4, i%4].axis('off')
plt.show()
```
## Model Building
### Build Pipeline
Pipeline menggunakan function untuk Normalisasi Gambar dan Labelnya
- tipe data pada gambar dari `uint8` diubah menjadi `float32`
- normalisasi dengan pembagian 255, karena data digit berukuran dari 0-255
- menggunakan `tf.cast` sebagai normalizernya
```
# function untuk normalisasi image
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label
```
Training Pipeline untuk Transformasi gambar
- `ds.map`, menjalankan function normalisasi pada gambar yang tipe data asalnya `tf.uint8` dari TFDS
  - sementara model tensorflow, tipe datanya berbentuk `tf.float32`
- `ds.cache`, cache data sebelum `shuffling` untuk mencocokan dataset agar performa model lebih baik
  - None, transformasi random (acak) harus diterapkan setelah `caching`
- `ds.shuffle`, untuk mengacak (random) data dengan mengatur buffer shuffle pada dataset ukuran penuh   
  - None, karena nilai standarnya 1000 sesuai sistem, maka tidak cukup digunakan pada yang lebih besar.
- `ds.batch`, `batch` setelah `shuffle` untuk memperoleh `batch` unique dari tiap `epoch`
- `ds.prefetch`, sebagai penutup pipeline
```
# mengubah tipe data menjadi tf.float32
ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
# mencocokan dataset 
ds_train = ds_train.cache()
# proses shuffling untuk random data
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
# mencari batch terbaik dari tiap epoch
ds_train = ds_train.batch(128)
# menutup pipeline
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
```
Build Testing Pipeline
- tidak menggunakan call `ds.shuffle()`
- melakukan caching setelah batch (karena tidak bisa sama antar epoch)
```
# mengubah tipe data menjadi tf.float32
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
# mencari batch terbaik dari tiap epoch
ds_test = ds_test.batch(128)
# mencocokan dataset 
ds_test = ds_test.cache()
# menutup pipeline
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
```

### Create the Dataset Object
buat object dataset masing-masing buat train dan validasi
- agar masing-masing object bisa dilolosin ke fit function, 

cara kerja tensorflow & keras, masing2 training dan validasi harus dibuat object terlebih dahulu
```
# training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(
        encode_single_sample, num_parallel_calls = tf.data.AUTOTUNE
    ).batch(batch_size).prefetch(buffer_size = tf.data.AUTOTUNE)
)

# validation dataset
validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = (
    validation_dataset.map(
        encode_single_sample, num_parallel_calls = tf.data.AUTOTUNE
    ).batch(batch_size).prefetch(buffer_size = tf.data.AUTOTUNE)
)
```
### CTC Loss Class
- valuable operation untuk menangani urutan masalah yang timingnya berupa variable
- ideal untuk digunakan pada data speech dan tulisan tangan
- maka tidak diperlukan untuk aligned dataset, karena tiap karakter tidak diperlukan pada lokasi yang tepat

contoh kasus
- misalnya ada duplicate atau tidak mau ada data yang perfectly aligned
- kemudian duplicate nya ingin dihapus atau ga disimpan ke tempat tertentu

buat class Layer CTC
- terdapat `keras.backend.ctc_batch_cost` untuk menjalankan algoritma ctc loss pada tiap batch element
- kemudian function `call` untuk menghitung nilai loss pada waktu pelatihan dan hasil perhitungan ditambahkan pada layer loss
```
# CTC Layer Class
class CTCLayer(layers.Layer):
    # function __init__ untuk setiap deklarasi class
    def __init__(self, name=None):
        super().__init__(name=name)
        # untuk menjalankan algoritma ctc loss pada tiap batch elemen
        self.loss_fn = keras.backend.ctc_batch_cost

    # function call untuk menghitung nilai loss pada waktu pelatihan 
    def call(self, y_true, y_pred):
        # menghitung nilai loss pada waktu pelatihan 
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        # kemudian ditambahin ke layer menggunakan 'self.add_loss()`
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # balikin nilai prediksi yang udah dihitung
        return y_pred
```
### Create the Model
Create simple model with data pipeline

pelatihan model dijalankan dengan function `.fit()`
- dengan memasukan input pipeline `ds_train` sebagai data training
- dan memasukan input pipeline `ds_test` sebagai data validasi
- serta epochnya
```
# define the model object
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy()]
)

# fit the data train and data test as validation
model.fit(ds_train, epochs=6, validation_data=ds_test)
```
Simple CNN Model Building
```
# build the model function with logarithmic loss and ADAM gradient descent 
def baseline_model():
    # create model
    model = Sequential()
    # Layer Conv2D dengan 32 featue map
    model.add(Conv2D(32, # filter
                     (5, 5), # kernel
                     input_shape=(1, 28, 28), # ukuran input 
                     activation='relu')) # relu activation
    # layer maxpooling dengan ukuran pool size 2x2
    model.add(MaxPooling2D())
    # layer dropout dengan probability 0.2
    model.add(Dropout(0.2))
    # layer flatten untuk konversi matrix 2d jadi vektor 
    model.add(Flatten())
    # layer dense dengan 128 neuron dan activation relu
    model.add(Dense(128, activation='relu'))
    # layer dense dengan jumlah neuron sebagai class dan activation softmax
    model.add(Dense(num_classes, activation='softmax'))

    # compile the model menggunakan optimisasi adam
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
```
Large CNN Model Building
```
# define the larger 
def larger_model():
    # create cnn model with Sequential Object
    model = Sequential()
    # layer convolutional 1 dengan max pooling nya
    model.add(Conv2D(30, 
                     (5, 5), 
                     input_shape=(1, 28, 28), 
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # layer convolutional 2 dengan max pooling nya
    model.add(Conv2D(15, 
                     (3, 3), 
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # layer droppout dengan probability 0.2 
    model.add(Dropout(0.2))
    # layer flatten 
    model.add(Flatten())
    # layer dense
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # compile cnn model
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model
```
CNN Model Building with CTC Loss 
```
# Function Model Building
def build_model():
    # menentukan layer input untuk image ke model
    input_img = layers.Input(
          # menentukan shape sehingga terdapat size gambar dan width nya
          # channel 1 karena dijalanin pake grayscale dengan nama image
          shape=(img_width, img_height, 1), name="image", dtype="float32"
          # float32 sebagai tipe data default pada tensorflow keras
    )
    # menentukan layer input untuk label
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # blok layer convolutional dengan masing-masing maxpoolingnya dan kernel 3x3
    # layer maxpooling untuk memperoleh max value
    
    # create cnn model with Sequential Object
    model = Sequential()
    
    # blok layer convolutional 1 dengan 32 filter 
    x = layers.Conv2D(32, 
                      (3, 3), 
                      activation='relu',
                      padding='same', 
                      name='Conv1')(input_img)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)

    # blok layer convolutional 2
    x = layers.Conv2D(64, 
                      (3, 3), 
                      activation='relu', 
                      padding='same', 
                      name='Conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)

    # menggunakan 2 max pool dengan ukuran pool dan strides 2 
    # sehingga map fitur yang didownsampled 4x lebih kecil ukurannya                  
    # jumlah filter dari layer convolutional terkhir di layer ke 2 adalah 64  
    
    # reshape kembali sesuai bentuk asalnya sebelum nerusin output ke bagian RNN
    # // sebagai floor division untuk membagi masing2 lebar dan tinggi gambar
    # floor division menghasilkan nilai berupa hasil dari pembagian bersisa
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.2)(x)

    # Algoritma RNN Disini
    x = layers.Bidirectional(layers.LSTM(128, return_sequences = True, 
                                         dropout = 0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences = True, 
                                         dropout = 0.25))(x)

    # Layer Output
    x = layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, 
        activation = 'softmax', 
        name = 'dense2'
    )

    # nambahin Layer CTC buat ngitung CTC Loss di each step
    output = CTCLayer(name='ctc_loss')(labels, x)
    
    # Define the model
    model = keras.models.Model(
        inputs = [input_img, labels], outputs = output, name = 'ocr_model_v1'
    )

    # Optimizer
    opt = keras.optimizers.Adam()
    # compile model, terus di return
    model.compile(loss='categorical_crossentropy', 
                  optimizer = opt,
                  metrics=['accuracy'])
    return model
```
## Model Training
train model dengan memanggil function yang berisi model, kemudian menggunakan function `.fit` dengan `epochs` dan `batch size` yang ditentukan
```
# run the simple cnn model function
# model = baseline_model()

# run the larger cnn model function
# model = larger_model()

# run model with ctc loss function
# model = build_model()

# train the model by fit function
model.fit(X_train, y_train, 
          validation_data = (X_test, y_test),
          epochs=10, 
          batch_size= 200, 
          verbose=2)
```
train the model with early stopping
```
epochs = 100
early_stopping_patience = 10

# early stopping, merupakan method dari callbacks pada keras
early_stopping = keras.callbacks.EarlyStopping(
    # monitor validation lossnya agar tidak meningkat lagi
    monitor = 'val_loss', 
    # kemudian terminate proses trainingny
    patience = early_stopping_patience, 
    restore_best_weights= True
)

# Train the model
history = model.fit( # terdiri dari data train dan validasi
                    train_dataset, validation_data = validation_dataset,
                    # kemudian epochs dan callback pake earlystopping
                    epochs = epochs, callbacks = [early_stopping])
```

Menampilkan hasil Loss dari Model Training
```
# Plot the loss of the model during training
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xticks(range(0, epochs+1, 10))
plt.title('Loss of OCR Model')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(loc='upper right')
plt.show()
```

## Model Evaluation
kemudian evaluasi performa model dengan `evaluate` menggunakan data testing `X_test` dan `y_test`
```
# evaluation using test dataset
scores = model.evaluate(X_test, y_test, verbose=0)
print(f'CNN Error: {100-scores[1]*100:.2f}')
```
evaluate the model by viewing each Accuracy and Loss
```
score = model.evaluate(x_test, y_test, verbose=0)
print(f'Tess Loss : {score[0]}')
print(f'Test Accuracy : {score[1]}')
```

## Model Prediction
buat object prediction
```
# get predicton model dengan ekstrak layer ampe ke outputnya
prediction_model = keras.models.Model(
    # layer image sebagai input
    model.get_layer(name = 'image').input, 
    # layer density sebagai output
    model.get_layer(name = 'dense2').output
)

# jalankan prediksi modelnya
prediction_model.fit()
```

decode output dari CNN
```
# function buat decode output dari network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # pake greedy search. buat tugas yang complex pake beam search
    results = keras.backend.ctc_decode(pred, input_length = input_len, 
                                       greedy=True)[0][0][:, :max_length]
    # iterasi result dari decode nya 
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    terus di return lagi text nya
    return output_text
```

periksa results dari sample validation
```
# jumlah batch diperoleh dari object dataset validasinya
for batch in validation_dataset.take(1)
    # batch image dan labelnya
    batch_images = batch['image']
    batch_labels = batch['label']

    # prediksi jalankan pada batch gambar, terus di decode ke text
    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    # original text untuk memastikan kalau prediksi modelnya correct
    orig_texts = []
    for label in batch_labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode('utf-8')
        orig_texts.append(label)

    # visualisasi data pada actual result
    _, ax = plt.subplots(4, 4, figsize=(15, 5))
    for i in range(len(pred_texts)):
        img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        img = img.T
        title = f'Prediction : {pred_texts[i]}'
        ax[i // 4, i % 4].imshow(img, cmap = 'gray')
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis('off')
plt.show()
```
