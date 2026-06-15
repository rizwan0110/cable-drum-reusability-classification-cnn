import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# function to augment images and save them to the output directory
def augment_images(image_dir, output_dir, num_augmented_images=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    # access through directories
    for root, _, files in os.walk(image_dir):
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(root, filename)
                img = load_img(img_path)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)

                # Generate new images
                i = 0
                for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix='aug', save_format='jpeg'):
                    i += 1
                    if i >= num_augmented_images:
                        break
                    
image_dir = "path/to/your/data"
output_dir = "path/to/your/data"
augment_images(image_dir, output_dir, num_augmented_images=10)