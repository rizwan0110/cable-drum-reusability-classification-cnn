import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# applying random cropping
def random_crop(image, crop_size):
    cropped_image = tf.image.random_crop(image, size=crop_size)
    return cropped_image

# applying  color jitter
def color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
    image = tf.image.random_brightness(image, max_delta=brightness)
    image = tf.image.random_contrast(image, lower=1-contrast, upper=1+contrast)
    image = tf.image.random_saturation(image, lower=1-saturation, upper=1+saturation)
    image = tf.image.random_hue(image, max_delta=hue)
    return image

# function that performs data augmentation
def augment_images(image_dir, output_dir, num_augmented_images_per_image=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    images = []

    for root, _, files in os.walk(image_dir):
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(root, filename)
                try:
                    img = load_img(img_path)
                    x = img_to_array(img)
                    x = x / 255.0  
                    images.append((x, filename))
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    if len(images) == 0:
        print("No images found for augmentation.")
        return

    # generate augmented images for each input image
    for idx, (image, filename) in enumerate(images):
        for j in range(num_augmented_images_per_image):
            
            crop_size = [int(image.shape[0] * 0.8), int(image.shape[1] * 0.8), image.shape[2]]  
            cropped_image = random_crop(image, crop_size)
            resized_image = tf.image.resize(cropped_image, [image.shape[0], image.shape[1]])

            jittered_image = color_jitter(resized_image)

            augmented_image = np.clip(jittered_image * 255, 0, 255).astype(np.uint8)  

            # save the augmented image
            base_filename = os.path.splitext(filename)[0]
            save_path = os.path.join(output_dir, f'{base_filename}_aug_{j}.jpeg')
            tf.keras.preprocessing.image.save_img(save_path, augmented_image)

            print(f'Saved augmented image: {save_path}')

image_dir = "path/to/your/data"
output_dir = "path/to/your/data"
augment_images(image_dir, output_dir, num_augmented_images_per_image=10)