import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import pathlib
from img_setting import img_height, img_width, batch_size

# this function take a glance at training data and print out 
# useful information about the data
def get_data_info():
    # set images path
    train_data_dir = pathlib.Path('images/training/')
    test_data_dir = pathlib.Path('images/testing')
    # pull training and testing data
    train_image_count = len(list(train_data_dir.glob('*/*.jpg')))
    test_image_count = len(list(test_data_dir.glob('*/*.jpg')))
    print('Number of images in the training set: ',train_image_count)
    print('Number of images in the testing set:',test_image_count)

    flip_train = list(train_data_dir.glob('flip/*'))
    not_flip_train = list(train_data_dir.glob('notflip/*'))
    flip_test = list(test_data_dir.glob('flip/*'))
    not_flip_test = list(test_data_dir.glob('notflip/*'))
    img = Image.open(str(flip_train[0]))

    # load data using keras utility
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_data_dir,
        validation_split=0.2,
        subset='training',
        seed = 123,
        image_size = (img_height, img_width),
        batch_size = batch_size
    )
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        train_data_dir,
        validation_split=0.2,
        subset='validation',
        seed = 123,
        image_size = (img_height, img_width),
        batch_size = batch_size
    )
    # print class name/ directory name
    class_names = train_dataset.class_names
    print(class_names)

    # print out first 9 images with label
    plt.figure(figsize=(10,10))
    for images, labels in train_dataset.take(1):
        for i in range (9):
            ax = plt.subplot(3,3,i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    # print out image batch shape and label batch shape
    for image_batch, label_batch in train_dataset:
        print(image_batch.shape)
        print(label_batch.shape)
        break
    return train_dataset, val_dataset, class_names, not_flip_test, flip_test