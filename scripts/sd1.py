#os path,loss.
#validation error important
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random
import os
import ntpath
import pandas as pd
import Nvidia_Model
import augmentation
import Data
import plot
DISPLAY_PLOTS = True
def disp(plt):
    global DISPLAY_PLOTS
    if DISPLAY_PLOTS == True:
        plt.show()

#part1
datadir='track-master'#opening folder
columns=['center','left','right','steering','throttle','reverse','speed']
data=pd.read_csv(os.path.join(datadir,'driving_log.csv'),names=columns)
Data.check(data)

#plotting steering angles
num_bins=25#divide entire range of values in intervals
samples_per_bin=400# every single bar of histogram has max of 400 values
hist,bins=np.histogram(data['steering'],num_bins)#dividing entire range in to 25 intervals
center=(bins[:-1]+bins[1:])*0.5
print(bins)
plt.bar(center,hist,width=0.05)
disp(plt)

#removing extra straight road driving history
print('total data:', len(data))
remove_list = []
for j in range(num_bins):
    list_ = []
    for i in range(len(data['steering'])):
        if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j + 1]:
            list_.append(i)
    list_ = shuffle(list_)
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)

print('removed:', len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print('remaining:', len(data))

hist, _ = np.histogram(data['steering'], (num_bins))
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
disp(plt)
# print(data.iloc[1])


print("divding data in to training and validation")
#taking steering data from csv file and dividing into ttraining and validation data
def load_img_steering(datadir, df):
    image_path = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]# each iloc contains one row of csv files
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        image_path.append(os.path.join(datadir, center.strip()))#.strip to elminate any spaces in string
        steering.append(float(indexed_data[3]))
        # left image append
        image_path.append(os.path.join(datadir, left.strip()))
        steering.append(float(indexed_data[3]) + 0.15)
        # right image append
        image_path.append(os.path.join(datadir, right.strip()))
        steering.append(float(indexed_data[3]) - 0.15)
    image_paths = np.asarray(image_path)
    steerings = np.asarray(steering)
    return image_paths, steerings

image_paths, steerings = load_img_steering(datadir + '/IMG', data)
print("image paths :",len(image_paths))
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)
print('Training Samples: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))


# #augmentation
# image=image_paths[random.randint(0,1000)]
# #zooming
# augmentation.plot_zoom(image)
# #pan
# augmentation.plot_pan(image)
# #brightness
# augmentation.plot_brightness(image)
#image flipping
random_index = random.randint(0, 1000)
image = image_paths[random_index]
steering_angle = steerings[random_index]
augmentation.plot_flip(image,steering_angle)


# Pre-Processing
def img_preprocess(img):
    img=img[60:135:,:,]#removing hood of car from  and background
    img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)#this color is better for NVIDEA model
    img=cv2.GaussianBlur(img,(3,3),0)#to reduce noise
    img=cv2.resize(img,(200,66))#decreasing size for faster computation
    img=img/255 #normalize the values to range from 0 to 1. instead of 0-256
    return img

image = image_paths[100]#100th index ,number is random
original_image = mpimg.imread(image)
#original_image = image
preprocessed_image = img_preprocess(original_image)# function used to remove hood of car
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[1].imshow(preprocessed_image)
axs[1].set_title('Preprocessed Image')
disp(plt) # showing new image after removing hood


#batch generator/image data generator- it will create augmented images along wih labels using dataset.
#it creates augmented images on the fly rather than augmenting all images at once and storing
#in valuable memory space, generator allow to create small batches only when generator is called
#this is memory effecient
def batch_generator(image_paths, steering_ang, batch_size, istraining):
    while True:
        batch_img = []
        batch_steering = []

        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)

            if istraining:
                im, steering = augmentation.random_augment(image_paths[random_index], steering_ang[random_index])

            else:
                im = mpimg.imread(image_paths[random_index])
                steering = steering_ang[random_index]

            im = img_preprocess(im)
            batch_img.append(im)
            batch_steering.append(steering)
        yield (np.asarray(batch_img), np.asarray(batch_steering))


x_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
x_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(x_train_gen[0])
axs[0].set_title('Training Image')
axs[1].imshow(x_valid_gen[0])
axs[1].set_title('Validation Image')
plt.show()


#Part3
#neural model
model = Nvidia_Model.nvidia_model()
# print(model.summary())
history = model.fit_generator(batch_generator(X_train, y_train, 100, 1),
                                  steps_per_epoch=300,
                                  epochs=10,
                                  validation_data=batch_generator(X_valid, y_valid, 100, 0),
                                  validation_steps=200,
                                  verbose=1,
                                  shuffle = 1)#batc generator creats 100 per step

#The training loss indicates how well the model is fitting the training data
# while the validation loss indicates how well the model fits new data.


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
# model.save('model.h5')
# from google.colab import files
#
# files.download('model.h5')










