#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import random
import os


# In[8]:


data_dir = "path/to/your/data"
output_folder = "path/to/your/data"


img_height, img_width = 224, 224
batch_size = 32


# In[9]:


# Load the datasets
def load_datasets(batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels="inferred",  
    label_mode="int"    
)
    
    # Normalize the images to the range [0, 1]
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Configure the dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return val_ds


# In[10]:


# Grad-CAM functions
def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    gate_f = tf.cast(output > 0, "float32")
    gate_r = tf.cast(grads > 0, "float32")
    guided_grads = gate_f * gate_r * grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = np.zeros(output.shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam.numpy(), (img_array.shape[2], img_array.shape[1]))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    return heatmap


# In[11]:


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # Resize heatmap to match image size
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    superimposed_img = heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()


# In[ ]:


best_model = load_model('final_model.h5')

# Load the validation dataset
val_ds = load_datasets(batch_size=32)

# Get random images from the validation dataset
val_images = []
val_labels = []

for images, labels in val_ds.take(1):
    val_images = images.numpy()
    val_labels = labels.numpy()

random_indices = random.sample(range(len(val_images)), 10)
selected_images = [val_images[i] for i in random_indices]

# Perform Grad-CAM visualization
for i, img in enumerate(selected_images):
    img_path = f'val_image_{i}.jpg'
    tf.keras.preprocessing.image.save_img(img_path, img)

    img_array = get_img_array(img_path, size=(img_height, img_width))
    heatmap = make_gradcam_heatmap(img_array, best_model, last_conv_layer_name="conv5_block3_out", classifier_layer_names=["global_average_pooling2d", "dense"])
    save_and_display_gradcam(img_path, heatmap, cam_path=f'{output_folder}/cam_{i}.jpg')


# In[ ]:




