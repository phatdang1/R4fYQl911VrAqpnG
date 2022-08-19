import tensorflow as tf
from keras.layers import Dense, Conv2D, MaxPooling2D, Rescaling, Flatten
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
from img_setting import img_height, img_width, batch_size

def cache_dataset(train_dataset, val_dataset):
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return train_dataset, val_dataset

def cnn_model(num_classes, train_dataset):
    # standardize the data to rgb value
    normalization_layer = layers.Rescaling(1./255)
    normalized_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    image_batch, label_batch  = next(iter(normalized_dataset))
    first_img = image_batch[0]
    # the pixel values are now in range [0,1].
    print(np.min(first_img), np.max(first_img))
    #keras sequential model with convolution blocks
    model = Sequential([Rescaling(1./255, input_shape=(img_height, img_width, 3)),
                    Conv2D(16, 3, padding='same', activation='relu'),
                    MaxPooling2D(),
                    Conv2D(32, 3, padding='same', activation='relu'),
                    MaxPooling2D(),
                    Conv2D(64, 3, padding='same', activation='relu'),
                    MaxPooling2D(),
                    layers.Dropout(0.2),
                    Flatten(),
                    Dense(128, activation='relu'),
                    Dense(num_classes)])
    # compile the model
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    # print out model summary
    model.summary()

# train the model:
def train_model(model, train_dataset, val_dataset):
    epochs=10
    history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
    )
    return epochs, history

def visualize_training_result(epochs, history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def predict(model, class_names, not_flip_test, not_flip_size):
    for i in range(10):
        img = tf.keras.utils.img_to_array(not_flip_test[i])
        img = tf.expand_dims(img,0)
        predictions = model.predict(img)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )