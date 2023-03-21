import os
import pathlib
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2



model = tf.keras.models.load_model('my_model.h5')

batch_size =  32
img_size = 180
test_dir = pathlib.Path('chest_xray/test')

test_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  image_size = (img_size, img_size),
  batch_size = batch_size,
  shuffle=False)


loss, acc, prec, rec = model.evaluate(test_ds)



# Load the image as grayscale
#img = cv2.imread('chest_xray/test/PNEUMONIA/person3_virus_17.jpeg', cv2.IMREAD_GRAYSCALE)
#WRONG RESULTS 
#img = cv2.imread('chest_xray/test/NORMAL/NORMAL2-IM-0237-0001.jpeg')
#img = cv2.imread('chest_xray/test/NORMAL/NORMAL2-IM-0238-0001.jpeg')
#img = cv2.imread('chest_xray/test/NORMAL/NORMAL2-IM-0241-0001.jpeg')
#img = cv2.imread('chest_xray/test/NORMAL/NORMAL2-IM-0381-0001.jpeg')

img = cv2.imread('chest_xray/test/PNEUMONIA/person1_virus_9.jpeg')

# Convert the image to RGB with 3 color channels
#img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# Resize the image to 180x180
img_resized = cv2.resize(img, (180, 180))

# Convert the image to a NumPy array with dtype float32
img_array = np.array(img_resized, dtype=np.float32)

# Scale the pixel values to the range [0, 1]
#img_array /= 255.0


# Add a batch dimension to the image
img_batch = np.expand_dims(img_array, axis=0)



# Make a prediction using the model
print("oo", model)
prediction = model(img_batch)

print(prediction)
'''
image = Image.open('chest_xray/test/NORMAL/IM-0022-0001.jpeg')
#image.show()
image = image.resize((180,180))

image = image.convert('L')
image.show()
image_arr = np.array(image) / 255

image2 = Image.fromarray(image_arr)
image2.show()
'''