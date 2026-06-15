#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from PIL import Image
import numpy as np

data_dir = "path/to/your/data"

# function to check if a file is an image
def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))

# function to process a single image file
def process_image_file(filepath):
    try:
        with Image.open(filepath) as img:
            img = img.convert("RGB")  
            return np.array(img)
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return None

class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])


dataset = {class_name: [] for class_name in class_names}

# process files within each directory
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    for root, _, files in os.walk(class_dir):
        for file in files:
            if is_image_file(file):
                file_path = os.path.join(root, file)
                image = process_image_file(file_path)
                if image is not None:
                    dataset[class_name].append(image)


for class_name, images in dataset.items():
    print(f"Class '{class_name}' has {len(images)} images.")


# In[ ]:




