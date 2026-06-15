

#Import necesary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report

weights_path = "path/to/your/data"
data_dir = "path/to/your/data"

# image size and batch size
img_height, img_width = 224, 224
batch_size = 32





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
    # Extract class names
    class_names = val_ds.class_names
    print("Class names:", class_names)
    
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Configure the dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds





def model_builder():
    base_model = ResNet50(weights=weights_path, include_top=False, input_shape=(img_height, img_width, 3))

    # Adding custom layers on top of ResNet50
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)  

    
    model = Model(inputs=base_model.input, outputs=x)

    for layer in base_model.layers:
        layer.trainable = False
      
    # compile the model
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model





def save_initial_model():
    model = model_builder()
    train_ds, val_ds = load_datasets(batch_size=32)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('final_model.h5', save_best_only=True)

    model.fit(train_ds, epochs=25, validation_data=val_ds, callbacks=[early_stopping, model_checkpoint])
    model.save('final_model.h5')




save_initial_model()

best_model = load_model('final_model.h5')


_, val_ds = load_datasets(batch_size=32) 

val_loss, val_accuracy = best_model.evaluate(val_ds)

val_predictions = best_model.predict(val_ds)
val_predictions = np.where(val_predictions > 0.5, 1, 0)  

true_labels = np.concatenate([y for x, y in val_ds], axis=0)

# calculate precision, recall, and F1 score
report = classification_report(true_labels, val_predictions, target_names=['nonreusable', 'reusable'], output_dict=True)

overall_precision = report['weighted avg']['precision']
overall_recall = report['weighted avg']['recall']
overall_f1 = report['weighted avg']['f1-score']

print("Overall Classification Report:")
print(f"Precision: {overall_precision}")
print(f"Recall: {overall_recall}")
print(f"F1 Score: {overall_f1}")

