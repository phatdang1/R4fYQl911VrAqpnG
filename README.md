# R4fYQl911VrAqpnG
Run the model:
    run main.py

Background:

    MonReader is a new mobile document digitalization experience for the blind, for researchers and for everyone else in need for fully automatic, highly fast and high-quality document scanning in bulk. It is composed of a mobile app and all the user needs to do is flip pages and everything is handled by MonReader: it detects page flips from low-resolution camera preview and takes a high-resolution picture of the document

Data Description:

    collected page flipping video from smart phones and labelled them as flipping and not flipping.

    clipped the videos as short videos and labelled them as flipping or not flipping. The extracted frames are then saved to disk in a sequential order with the following naming structure: VideoID_FrameNumber

Goal: Predict if the page is being flipped using a single image

Algorithm: 
    -Put training images in different directory for correct labeling.
    -Divide training data into train datadet and validation dataset
    -Cache dataset for better performance
    -Rescaling the pixel from range[0,255] to [0,1]
    -Build a Convolutional Neural Network to be the main model (adjust number of layers, activation functions, and which type of layer to get the best result)
    -Complie the model with optizier 'adam' for faster training time
    -Train the model for about 10 epoches and make a prediction

Model output:

Number of images in the training set:  2989
Number of images in the testing set: 597
Found 2989 files belonging to 2 classes.
Using 2392 files for training.
Found 2989 files belonging to 2 classes.
Using 597 files for validation.
['flip', 'notflip']
(32, 180, 180, 3)
(32,)

0.0 0.9993466

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 rescaling_1 (Rescaling)     (None, 180, 180, 3)       0         
                                                                 
 conv2d (Conv2D)             (None, 180, 180, 16)      448       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 90, 90, 16)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 90, 90, 32)        4640      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 45, 45, 32)       0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 45, 45, 64)        18496     
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 22, 22, 64)       0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 30976)             0         
                                                                 
 dense (Dense)               (None, 128)               3965056   
                                                                 
 dense_1 (Dense)             (None, 2)                 258       
                                                                 
=================================================================
Total params: 3,988,898
Trainable params: 3,988,898
Non-trainable params: 0
_________________________________________________________________
Epoch 1/10
75/75 [==============================] - 127s 2s/step - loss: 0.5654 - accuracy: 0.7007 - val_loss: 0.2964 - val_accuracy: 0.8693
Epoch 2/10
75/75 [==============================] - 109s 1s/step - loss: 0.2046 - accuracy: 0.9214 - val_loss: 0.1159 - val_accuracy: 0.9531
Epoch 3/10
75/75 [==============================] - 109s 1s/step - loss: 0.1500 - accuracy: 0.9394 - val_loss: 0.1201 - val_accuracy: 0.9514
Epoch 4/10
75/75 [==============================] - 109s 1s/step - loss: 0.0458 - accuracy: 0.9866 - val_loss: 0.0217 - val_accuracy: 0.9966
Epoch 5/10
75/75 [==============================] - 108s 1s/step - loss: 0.0242 - accuracy: 0.9929 - val_loss: 0.0247 - val_accuracy: 0.9933
Epoch 6/10
75/75 [==============================] - 108s 1s/step - loss: 0.0487 - accuracy: 0.9833 - val_loss: 0.1270 - val_accuracy: 0.9481
Epoch 7/10
75/75 [==============================] - 108s 1s/step - loss: 0.0379 - accuracy: 0.9862 - val_loss: 0.0294 - val_accuracy: 0.9933
Epoch 8/10
75/75 [==============================] - 111s 1s/step - loss: 0.0225 - accuracy: 0.9921 - val_loss: 0.0322 - val_accuracy: 0.9899
Epoch 9/10
75/75 [==============================] - 124s 2s/step - loss: 0.0041 - accuracy: 0.9992 - val_loss: 0.0049 - val_accuracy: 1.0000
Epoch 10/10
75/75 [==============================] - 113s 2s/step - loss: 8.7535e-04 - accuracy: 1.0000 - val_loss: 0.0021 - val_accuracy: 1.0000