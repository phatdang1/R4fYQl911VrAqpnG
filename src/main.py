from eda import get_data_info
from model import cache_dataset, cnn_model, train_model, visualize_training_result, predict

# data eda, and get train dataset and validation dataset
train_dataset, val_dataset, class_names, not_flip_test, flip_test = get_data_info()
# cache data for fast operation
cache_dataset(train_dataset, val_dataset)
# get number of classes in the dataset
num_classes = len(class_names)
# construct the model
model = cnn_model(num_classes)
# train the model
epochs, history = train_model(model, train_dataset, val_dataset)
# visualizing the training result
visualize_training_result(epochs, history)
# range of test image use for testing
not_flip_size = len(not_flip_test)
flip_test_size = len(flip_test)
# predict
predict(model, class_names, not_flip_test, not_flip_size)

