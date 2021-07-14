#Header code used to connect the google colab jupyter notebook to google drive
#for read and write access to the training dataset.

"""
Comment out the following code if planning to train the model on a cloud platform
other than google colab.

Uptil line ~ 25
"""
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}

!mkdir -p drive
!google-drive-ocamlfuse drive

#Imports all the necessary packages for the model

import keras
import tensorflow as tf
import os, sys
import numpy as np
import random
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave

#Access the training dataset available on the google drive and store it as a
#numpy array
Xtrain = []
i = 0
for filename in os.listdir('drive/colorizer/train'):
  print(i, filename)
  i += 1
  Xtrain.append(img_to_array(load_img('drive/colorizer/train'+filename)))
Xtrain = np.array(Xtrain, dtype=np.float64)
Xtrain = 1.0/255*Xtrain


cnnModel = Sequential()
cnnModel.add(InputLayer(input_shape=(256, 256, 1)))
cnnModel.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
cnnModel.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
cnnModel.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
cnnModel.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
cnnModel.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
cnnModel.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
cnnModel.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
cnnModel.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
cnnModel.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
cnnModel.add(UpSampling2D((2, 2)))
cnnModel.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
cnnModel.add(UpSampling2D((2, 2)))
cnnModel.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
cnnModel.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
cnnModel.add(UpSampling2D((2, 2)))
cnnModel.summary()

#Compiling the defined CNN network architecture
cnnModel.compile(optimizer='rmsprop', loss='mse')

#Load the trained model weights from the HDF5 file
model.load_weights('model_weights.h5')

#Load the grayscle test images stored in google drive
testImages = []
for filename in os.listdir('drive/colorizer/test/'):
        testImages.append(img_to_array(load_img('drive/colorizer/test/'+filename)))
testImages = np.array(testImages, dtype=float)
testImages = rgb2lab(1.0/255*testImages)[:,:,:,0]
testImages = testImages.reshape(color_me.shape+(1,))

#Test the trained model for the grayscale test images
output = cnnModel.predict(testImages)
output = output * 128

#Convert the output from an array to a colored image from LAB color space
#RGB color space.
for i in range(len(output)):
        res = np.zeros((256, 256, 3))
        res[:,:,0] = testImages[i][:,:,0]
        res[:,:,1:] = output[i]
        imsave("drive/colorizer/result/img_"+str(i)+".png", lab2rgb(res))
