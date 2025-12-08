Unit: 2
CO: CO2
Q1. Which of the following statements correctly distinguishes between a grayscale image and an RGB image?
A. Grayscale images have 3 color channels, while RGB images have 1.
B. Grayscale images use 8 bits per pixel, while RGB images typically use 24 bits per pixel.
C. RGB images represent intensity only, while grayscale images represent color.
D. Grayscale images require more storage space than RGB images of the same resolution.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q2. In a digital image, what does the resolution typically refer to?
A. The total number of pixels along the height and width.
B. The number of color channels.
C. The bit-depth of the image.
D. The compression format used.
Correct Answer: A
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q3. What is the primary purpose of a Convolutional Layer in a CNN?
A. To reduce the spatial dimensions of the input.
B. To flatten the input into a 1D vector.
C. To extract local features such as edges and textures using learnable filters.
D. To classify the image into final categories.
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q4. In the context of CNNs, what is a "kernel" or "filter"?
A. A small matrix of weights that slides over the input data.
B. The final output class label.
C. A function that removes noise from the dataset.
D. The bias term added to the output.
Correct Answer: A
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q5. If an input image has dimensions $32 \times 32 \times 3$, what does the number '3' represent?
A. The width of the image.
B. The number of color channels (Red, Green, Blue).
C. The number of filters used.
D. The batch size.
Correct Answer: B
Type: Numerical
Difficulty: Easy

Unit: 2
CO: CO2
Q6. Which operation effectively increases the receptive field of a neuron in later layers without increasing the number of parameters?
A. Padding
B. Pooling
C. Flattening
D. Batch Normalization
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q7. What is the output height of a convolutional layer given input height $H$, filter size $F$, stride $S$, and zero padding $P$?
A. $(H - F + 2P) / S + 1$
B. $(H - F - 2P) / S + 1$
C. $(H + F + 2P) / S - 1$
D. $(H \times F) / S$
Correct Answer: A
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q8. Consider an input volume of size $32 \times 32 \times 1$. You apply a convolution with 10 filters of size $5 \times 5$, stride $S=1$, and no padding (valid). What is the spatial dimension of the output volume?
A. $32 \times 32$
B. $28 \times 28$
C. $27 \times 27$
D. $5 \times 5$
Correct Answer: B
Type: Numerical
Difficulty: Medium

Unit: 2
CO: CO2
Q9. In TensorFlow/Keras, what argument in `Conv2D` controls the number of output filters?
A. `kernel_size`
B. `filters`
C. `strides`
D. `input_shape`
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 2
CO: CO2
Q10. What is the result of using "Same" padding in a convolutional layer with a stride of 1?
A. The output spatial dimensions are smaller than the input.
B. The output spatial dimensions are the same as the input.
C. The output spatial dimensions are larger than the input.
D. The padding is automatically set to 0.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q11. Which activation function is most commonly used in the hidden layers of modern CNNs to introduce non-linearity and avoid the vanishing gradient problem?
A. Sigmoid
B. Tanh
C. ReLU (Rectified Linear Unit)
D. Softmax
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q12. Analyze the following code snippet:
```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
])
```
How many parameters (weights + biases) does this layer have?
A. 320
B. 288
C. 32
D. 9
Correct Answer: A
Type: Numerical
Difficulty: Medium

Unit: 2
CO: CO2
Q13. In a Max Pooling layer with a $2 \times 2$ pool size and stride 2, what happens to the input dimensions?
A. They remain the same.
B. The height and width are both halved.
C. The depth (number of channels) is halved.
D. The height and width are reduced by 2 pixels.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q14. Which component of a CNN architecture is primarily responsible for combining local features into global predictions (classification)?
A. Convolutional Layer
B. Pooling Layer
C. Fully Connected (Dense) Layer
D. Dropout Layer
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q15. Given an input volume of $14 \times 14 \times 64$, what is the output shape after applying a Global Average Pooling layer?
A. $1 \times 1 \times 64$ (or a vector of size 64)
B. $7 \times 7 \times 64$
C. $14 \times 14 \times 1$
D. $196$
Correct Answer: A
Type: Numerical
Difficulty: Medium

Unit: 2
CO: CO2
Q16. What is the primary innovation introduced by the LeNet-5 architecture?
A. Use of ReLU activation functions.
B. Use of convolution and pooling layers for handwritten digit recognition.
C. Use of residual skip connections.
D. Use of Inception modules.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q17. Which CNN architecture won the ILSVRC 2012 challenge and popularized the use of deep CNNs, GPUs, and Dropout?
A. LeNet-5
B. AlexNet
C. VGGNet
D. ResNet
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q18. Look at the code below. What is the purpose of the `Flatten` layer?
```python
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation='softmax'))
```
A. To reduce the number of channels to 1.
B. To convert the 2D feature maps into a 1D vector for the Dense layer.
C. To normalize the pixel values.
D. To apply dropout regularization.
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 2
CO: CO2
Q19. VGGNet is characterized by its use of:
A. Very large filters (e.g., $11 \times 11$).
B. A sequence of many $3 \times 3$ convolutional filters.
C. Inception modules with parallel convolutions.
D. Skip connections to train very deep networks.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q20. What problem does the ResNet (Residual Network) architecture solve using skip connections?
A. Overfitting on small datasets.
B. High computational cost of convolutions.
C. Vanishing gradient problem in very deep networks.
D. Inability to capture color information.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q21. Calculate the number of parameters in a Convolutional layer with 64 filters of size $3 \times 3 \times 3$ (assuming bias is included).
A. $3 \times 3 \times 3 \times 64 = 1728$
B. $(3 \times 3 \times 3 + 1) \times 64 = 1792$
C. $(3 \times 3 + 1) \times 64 = 640$
D. $3 \times 3 \times 64 = 576$
Correct Answer: B
Type: Numerical
Difficulty: Medium

Unit: 2
CO: CO2
Q22. In the Inception architecture (GoogLeNet), what is the main purpose of the $1 \times 1$ convolution?
A. To increase the spatial dimensions.
B. To serve as a bottleneck for dimensionality reduction (reducing depth).
C. To perform average pooling.
D. To remove noise from the image.
Correct Answer: B
Type: Theory
Difficulty: Hard

Unit: 2
CO: CO2
Q23. Which regularization technique randomly sets a fraction of input units to 0 at each update during training time?
A. Batch Normalization
B. Data Augmentation
C. Dropout
D. L2 Regularization
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q24. Identify the Keras layer used to implement Batch Normalization.
A. `tf.keras.layers.Normalization()`
B. `tf.keras.layers.BatchNormalization()`
C. `tf.keras.layers.StandardScaler()`
D. `tf.keras.layers.Lambda(tf.nn.batch_norm)`
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 2
CO: CO2
Q25. Given an image input of shape $(224, 224, 3)$, what is the output shape after applying `MaxPooling2D(pool_size=(2, 2), strides=2)`?
A. $(112, 112, 3)$
B. $(112, 112, 1)$
C. $(222, 222, 3)$
D. $(224, 224, 1)$
Correct Answer: A
Type: Numerical
Difficulty: Medium

Unit: 2
CO: CO2
Q26. What does the `padding='valid'` argument imply in a Keras Conv2D layer?
A. Zero padding is applied to maintain the output size same as input.
B. No padding is applied; the output size shrinks.
C. Invalid pixels are discarded.
D. Padding is applied only if the stride is greater than 1.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q27. Why is Data Augmentation useful in training CNNs?
A. It reduces the training time.
B. It increases the resolution of input images.
C. It generates new training samples from existing ones to reduce overfitting.
D. It automatically tunes the hyperparameters.
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q28. In a CNN for classification, if the training accuracy is high but validation accuracy is low, the model is likely:
A. Underfitting
B. Overfitting
C. Learning perfectly
D. Converging too slowly
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q29. Consider the following code snippet using `ImageDataGenerator`. What does `rescale=1./255` do?
```python
train_datagen = ImageDataGenerator(rescale=1./255)
```
A. It resizes the image to 255x255 pixels.
B. It normalizes pixel values to the range.
C. It augments the data by scaling the image size.
D. It converts the image to grayscale.
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 2
CO: CO2
Q30. Which loss function is appropriate for a multi-class classification problem (e.g., CIFAR-10) where labels are integers?
A. `binary_crossentropy`
B. `categorical_crossentropy`
C. `sparse_categorical_crossentropy`
D. `mean_squared_error`
Correct Answer: C
Type: Code
Difficulty: Medium

Unit: 2
CO: CO2
Q31. In transfer learning, what does "freezing" the base model mean?
A. Making the weights non-trainable so they don't get updated during training.
B. Saving the model to a file.
C. Stopping the training process early.
D. Reducing the learning rate to zero.
Correct Answer: A
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q32. How many total parameters are in a dense layer with 10 units connected to a flattened input of size 100?
A. 1000
B. 1010
C. 110
D. 100
Correct Answer: B
Type: Numerical
Difficulty: Medium

Unit: 2
CO: CO2
Q33. Which of the following is NOT a typical application of CNNs?
A. Image Classification
B. Object Detection
C. Stock Price Prediction (Time Series)
D. Semantic Segmentation
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q34. What is the main advantage of using small filters (like $3 \times 3$) stacked on top of each other compared to a single large filter (like $7 \times 7$)?
A. They cover less spatial area.
B. They increase the number of parameters.
C. They introduce more non-linearities and use fewer parameters.
D. They are faster to compute in all hardware.
Correct Answer: C
Type: Theory
Difficulty: Hard

Unit: 2
CO: CO2
Q35. In Keras, what does `model.summary()` display?
A. The training accuracy history.
B. The architecture of the model, output shapes, and parameter counts.
C. The visualization of feature maps.
D. The values of the weights.
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 2
CO: CO2
Q36. Which object detection algorithm processes the image once to predict bounding boxes and class probabilities simultaneously?
A. R-CNN
B. Faster R-CNN
C. YOLO (You Only Look Once)
D. Sliding Window
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q37. An image with resolution $100 \times 100$ passes through a Conv2D layer with 32 filters, kernel size $3 \times 3$, stride 1, and `padding='same'`. What is the shape of the output feature map?
A. $(100, 100, 32)$
B. $(98, 98, 32)$
C. $(50, 50, 32)$
D. $(100, 100, 3)$
Correct Answer: A
Type: Numerical
Difficulty: Medium

Unit: 2
CO: CO2
Q38. What is the receptive field?
A. The total number of neurons in a layer.
B. The region of the input image that a particular CNN feature is looking at.
C. The size of the padding added to the image.
D. The number of classes the model can predict.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q39. Which Keras layer is used to randomly flip images horizontally during training?
A. `tf.keras.layers.RandomFlip("horizontal")`
B. `tf.keras.layers.FlipImage()`
C. `tf.keras.layers.DataAugmentation()`
D. `tf.keras.layers.HorizontalFlip()`
Correct Answer: A
Type: Code
Difficulty: Easy

Unit: 2
CO: CO2
Q40. The ReLU activation function is defined as:
A. $f(x) = 1 / (1 + e^{-x})$
B. $f(x) = \tanh(x)$
C. $f(x) = \max(0, x)$
D. $f(x) = x$
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q41. A tensor with shape `(None, 28, 28, 1)` is input to a model. What does `None` represent?
A. The height of the image.
B. The number of channels.
C. The variable batch size.
D. An error in the definition.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q42. In Semantic Segmentation, the goal is to:
A. Draw a bounding box around objects.
B. Classify the entire image into one category.
C. Classify each pixel of the image into a category.
D. Generate a caption for the image.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q43. Which of the following is true about $1 \times 1$ convolutions?
A. They do not change the spatial dimensions but can change the number of channels (depth).
B. They are used to increase the spatial dimensions.
C. They effectively do nothing.
D. They are only used in Fully Connected layers.
Correct Answer: A
Type: Theory
Difficulty: Hard

Unit: 2
CO: CO2
Q44. Calculate the output size: Input $227 \times 227$, Filter $11 \times 11$, Stride 4, Padding 0.
A. 55
B. 54
C. 56
D. 28
Correct Answer: A
Type: Numerical
Difficulty: Medium

Unit: 2
CO: CO2
Q45. Which code snippet correctly instantiates a VGG16 model with pre-trained ImageNet weights in Keras?
A. `tf.keras.applications.VGG16(weights='imagenet')`
B. `tf.keras.models.VGG16(pretrained=True)`
C. `tf.keras.layers.VGG16(weights='imagenet')`
D. `tf.keras.applications.VGG16(weights=None)`
Correct Answer: A
Type: Code
Difficulty: Medium

Unit: 2
CO: CO2
Q46. Batch Normalization is typically applied:
A. Before the input layer.
B. Before or after the activation function of a layer.
C. Only at the output layer.
D. After the loss calculation.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q47. If you have a small dataset but a pre-trained model on a large dataset (like ImageNet) is available, the best strategy is:
A. Train a large model from scratch.
B. Use Transfer Learning.
C. Use Unsupervised Learning.
D. Discard the data.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q48. What is the stride in a convolutional layer?
A. The number of filters used.
B. The step size the filter takes when sliding over the input.
C. The amount of zero padding added.
D. The size of the filter.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q49. Given a 1D tensor ``, applying a Max Pooling with pool size 2 and stride 2 results in:
A. ``
B. ``
C. ``
D. ``
Correct Answer: A
Type: Numerical
Difficulty: Easy

Unit: 2
CO: CO2
Q50. Which function converts the logits (raw outputs) of the final layer into class probabilities?
A. ReLU
B. Tanh
C. Softmax
D. Dropout
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q51. What is the primary benefit of using "Global Average Pooling" over "Flattening" before the dense layers?
A. It increases the number of parameters.
B. It preserves spatial information better.
C. It drastically reduces the number of parameters and prevents overfitting.
D. It is computationally more expensive.
Correct Answer: C
Type: Theory
Difficulty: Hard

Unit: 2
CO: CO2
Q52. The MNIST dataset consists of:
A. Color images of animals.
B. Grayscale images of handwritten digits (0-9).
C. Color images of clothing items.
D. Satellite images.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q53. A CNN model summary shows a layer output shape of `(None, 10, 10, 32)`. If this is followed by a `Flatten()` layer, what is the output shape?
A. `(None, 100)`
B. `(None, 320)`
C. `(None, 3200)`
D. `(None, 10, 10)`
Correct Answer: C
Type: Numerical
Difficulty: Easy

Unit: 2
CO: CO2
Q54. Which Keras callback is useful for stopping training when the validation loss stops improving?
A. `ModelCheckpoint`
B. `EarlyStopping`
C. `TensorBoard`
D. `ReduceLROnPlateau`
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 2
CO: CO2
Q55. In the context of `tf.keras.layers.Conv2D`, what does `activation=None` imply?
A. A linear activation (identity) is used.
B. ReLU is used by default.
C. The layer outputs zeros.
D. The layer is not trainable.
Correct Answer: A
Type: Code
Difficulty: Medium

Unit: 2
CO: CO2
Q56. Which of the following is a symptom of a high learning rate?
A. The loss decreases very slowly.
B. The loss oscillates or diverges.
C. The model overfits immediately.
D. The training accuracy is 100%.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q57. A 3-channel RGB image is processed by a Conv layer with 16 filters of size $3 \times 3$. The number of depth channels in the output feature map will be:
A. 3
B. 9
C. 16
D. 27
Correct Answer: C
Type: Numerical
Difficulty: Easy

Unit: 2
CO: CO2
Q58. What is the effect of L2 regularization on the weights of a CNN?
A. It forces weights to be exactly zero.
B. It penalizes large weights, encouraging simpler models.
C. It randomly sets weights to zero during training.
D. It normalizes the batch statistics.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q59. To perform object detection using a pre-trained model in TensorFlow Hub, which module would you likely search for?
A. ResNet-50 Classification
B. SSD MobileNet
C. BERT
D. StyleGAN
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q60. If you have a training set of 1000 images and you use `ImageDataGenerator` to generate batches, how much data can the model potentially see?
A. Exactly 1000 images.
B. Indefinitely many variations of the 1000 images.
C. 2000 images exactly.
D. Only the validation set.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q61. What is the output of `tf.keras.layers.GlobalMaxPooling2D()` on an input of shape `(None, 7, 7, 512)`?
A. `(None, 512)`
B. `(None, 49, 512)`
C. `(None, 7, 7, 1)`
D. `(None, 1)`
Correct Answer: A
Type: Numerical
Difficulty: Medium

Unit: 2
CO: CO2
Q62. Which architecture introduced the concept of "Depthwise Separable Convolutions" (Xception/MobileNet)?
A. AlexNet
B. VGG16
C. Xception
D. LeNet
Correct Answer: C
Type: Theory
Difficulty: Hard

Unit: 2
CO: CO2
Q63. The Fashion MNIST dataset contains images of size:
A. $28 \times 28$ grayscale
B. $32 \times 32$ color
C. $224 \times 224$ color
D. $64 \times 64$ grayscale
Correct Answer: A
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q64. Code Check: Which line correctly adds a Dropout layer with a 50% drop rate?
A. `model.add(tf.keras.layers.Dropout(0.5))`
B. `model.add(tf.keras.layers.Dropout(50))`
C. `model.add(tf.keras.layers.Drop(0.5))`
D. `model.add(tf.keras.layers.Regularization(0.5))`
Correct Answer: A
Type: Code
Difficulty: Easy

Unit: 2
CO: CO2
Q65. If a CNN is underfitting, which action is most appropriate?
A. Add Dropout.
B. Increase the complexity of the model (add more layers/filters).
C. Add L2 regularization.
D. Reduce the training time.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q66. What is the typical range of pixel values in a standard 8-bit digital image?
A. 0 to 1
B. -1 to 1
C. 0 to 255
D. 0 to 100
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q67. Calculate the output width: Input width 32, Filter width 3, Stride 1, Padding 'valid'.
A. 32
B. 30
C. 29
D. 10
Correct Answer: B
Type: Numerical
Difficulty: Easy

Unit: 2
CO: CO2
Q68. In Keras, how do you specify the input shape for the very first layer of a Sequential model?
A. Using `input_dim` argument.
B. Using `input_shape` argument.
C. It is inferred automatically.
D. Using `shape` argument.
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 2
CO: CO2
Q69. Which of the following best describes "feature maps"?
A. The weights of the convolutional filters.
B. The output of applying filters to the input image/layer.
C. The labels of the training data.
D. The learning rate schedule.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q70. Which architecture uses "Inception modules" to compute convolutions with different kernel sizes in parallel?
A. ResNet
B. VGGNet
C. GoogLeNet
D. AlexNet
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q71. What does `pool_size=(2,2)` mean in `MaxPooling2D`?
A. The layer outputs 2 feature maps.
B. The pooling window is 2 pixels high and 2 pixels wide.
C. The stride is 2.
D. The input image is resized to $2 \times 2$.
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 2
CO: CO2
Q72. If a model has high training accuracy but low validation accuracy, it is suffering from:
A. High Bias
B. Underfitting
C. Overfitting
D. Vanishing Gradient
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q73. What is the total number of parameters in a `Conv2D` layer with 32 filters, kernel size $3 \times 3$, and input depth 1 (including bias)?
A. $32 \times 3 \times 3 = 288$
B. $(3 \times 3 \times 1 + 1) \times 32 = 320$
C. $(3 \times 3 + 1) \times 32 = 320$
D. $3 \times 3 \times 32 = 288$
Correct Answer: B
Type: Numerical
Difficulty: Medium

Unit: 2
CO: CO2
Q74. Which layer is typically used as the final layer for a binary classification CNN?
A. `Dense(1, activation='sigmoid')`
B. `Dense(2, activation='softmax')`
C. `Dense(1, activation='relu')`
D. `Conv2D(1, kernel_size=1)`
Correct Answer: A
Type: Code
Difficulty: Medium

Unit: 2
CO: CO2
Q75. The "vanishing gradient" problem is most severe in:
A. Shallow networks with ReLU.
B. Deep networks with Sigmoid activations.
C. Networks with Batch Normalization.
D. Networks with Residual connections.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q76. A stride of 2 in a convolutional layer has an effect similar to:
A. Padding
B. Max Pooling with stride 2
C. Dropout
D. Flattening
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q77. What is the purpose of the `rescale` parameter in `ImageDataGenerator`?
A. To resize the image dimensions.
B. To multiply pixel values by a factor (e.g., 1/255) for normalization.
C. To randomly zoom the image.
D. To change the aspect ratio.
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 2
CO: CO2
Q78. In a VGG16 architecture, what is the size of the filters used throughout the convolutional layers?
A. $1 \times 1$
B. $3 \times 3$
C. $5 \times 5$
D. $7 \times 7$
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q79. Which function calculates the cross-entropy loss between true labels and predicted probabilities in TensorFlow?
A. `tf.keras.losses.SparseCategoricalCrossentropy()`
B. `tf.keras.losses.MeanSquaredError()`
C. `tf.keras.losses.Hinge()`
D. `tf.keras.metrics.Accuracy()`
Correct Answer: A
Type: Code
Difficulty: Medium

Unit: 2
CO: CO2
Q80. Calculate the output height: Input height 64, Filter 3, Stride 2, Padding 1 (Same-like behavior logic approximation for manual calc: $(H+2P-F)/S + 1$).
Let's use explicit values: $(64 + 2\times1 - 3)/2 + 1$.
A. 32
B. 31
C. 33
D. 64
Correct Answer: A
Type: Numerical
Difficulty: Hard

Unit: 2
CO: CO2
Q81. The "bottleneck" layer in Inception modules uses which kernel size?
A. $3 \times 3$
B. $5 \times 5$
C. $1 \times 1$
D. $7 \times 7$
Correct Answer: C
Type: Theory
Difficulty: Hard

Unit: 2
CO: CO2
Q82. Which of the following is a technique to visualize what a CNN filter has learned?
A. Gradient Descent
B. Backpropagation
C. Feature Map Visualization / Activation Maximization
D. Dropout
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q83. In the code `model.fit(train_images, train_labels, epochs=10)`, what is an epoch?
A. One forward pass of a single batch.
B. One update of the weights.
C. One complete pass through the entire training dataset.
D. The time it takes to train the model.
Correct Answer: C
Type: Code
Difficulty: Easy

Unit: 2
CO: CO2
Q84. Which metric is most commonly used for evaluating Object Detection models?
A. Accuracy
B. Mean Squared Error (MSE)
C. Intersection over Union (IoU) / Mean Average Precision (mAP)
D. Log Loss
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q85. Input: $6 \times 6$. Max Pooling: $2 \times 2$, Stride 2. Output size?
A. $3 \times 3$
B. $2 \times 2$
C. $4 \times 4$
D. $6 \times 6$
Correct Answer: A
Type: Numerical
Difficulty: Easy

Unit: 2
CO: CO2
Q86. What is the primary role of the Flatten layer before the Dense layers?
A. To normalize the data.
B. To reduce overfitting.
C. To convert multidimensional tensor data into a 1D array.
D. To apply activation functions.
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q87. Which Keras function is used to convert integer class vectors to binary class matrix (one-hot encoding)?
A. `tf.keras.utils.to_categorical`
B. `tf.one_hot`
C. `tf.keras.layers.OneHot`
D. `tf.keras.preprocessing.encode`
Correct Answer: A
Type: Code
Difficulty: Medium

Unit: 2
CO: CO2
Q88. Which CNN architecture relies heavily on depthwise separable convolutions to reduce parameter count for mobile devices?
A. VGG16
B. ResNet-50
C. MobileNet / Xception
D. AlexNet
Correct Answer: C
Type: Theory
Difficulty: Hard

Unit: 2
CO: CO2
Q89. In a grayscale image, a pixel value of 0 usually represents:
A. White
B. Black
C. Grey
D. Red
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q90. Calculate parameters for a Dense layer with 100 inputs and 10 outputs.
A. $100 \times 10 = 1000$
B. $(100 + 1) \times 10 = 1010$
C. $100 + 10 = 110$
D. $100 \times 10 + 100 = 1100$
Correct Answer: B
Type: Numerical
Difficulty: Medium

Unit: 2
CO: CO2
Q91. Which component of the CNN allows it to be translation invariant?
A. The activation function.
B. The pooling layer.
C. The dense layer.
D. The bias term.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q92. What does `model.evaluate(test_images, test_labels)` return in Keras?
A. The predictions for the test images.
B. The loss value and metrics values for the model in test mode.
C. The gradients of the loss.
D. The summary of the model.
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 2
CO: CO2
Q93. A color image of $100 \times 100$ pixels has how many values in the input tensor?
A. 10,000
B. 20,000
C. 30,000
D. 3
Correct Answer: C
Type: Numerical
Difficulty: Easy

Unit: 2
CO: CO2
Q94. The "Same" padding formula $P = (F - 1) / 2$ assumes:
A. Stride = 1
B. Stride = 2
C. Filter size is even
D. Filter size is 1
Correct Answer: A
Type: Numerical
Difficulty: Hard

Unit: 2
CO: CO2
Q95. Which statement about the ReLU function is false?
A. It outputs 0 for negative inputs.
B. It is computationally efficient.
C. It suffers from the saturation problem for large positive values.
D. It helps mitigate the vanishing gradient problem.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q96. In transfer learning, if your new dataset is small and similar to the original dataset, you should:
A. Fine-tune all layers.
B. Freeze the convolutional base and train only the top dense layers.
C. Train the entire model from scratch.
D. Freeze the dense layers and train the convolutional base.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 2
CO: CO2
Q97. Which code snippet sets the optimizer to Adam with a learning rate of 0.001?
A. `model.compile(optimizer='adam', ...)`
B. `model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), ...)`
C. `model.compile(optimizer=tf.keras.optimizers.SGD(0.001), ...)`
D. `model.compile(optimizer='sgd', ...)`
Correct Answer: B
Type: Code
Difficulty: Medium

Unit: 2
CO: CO2
Q98. What is the spatial output size of a $1 \times 1$ convolution applied to a $28 \times 28$ input?
A. $1 \times 1$
B. $14 \times 14$
C. $28 \times 28$
D. $26 \times 26$
Correct Answer: C
Type: Numerical
Difficulty: Easy

Unit: 2
CO: CO2
Q99. Semantic segmentation differs from object detection because:
A. It puts a box around the object.
B. It classifies every pixel rather than drawing boxes.
C. It only works on grayscale images.
D. It is faster.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 2
CO: CO2
Q100. If you have an input of $6 \times 6$ and apply a $3 \times 3$ filter with stride 2 and valid padding, what is the output size?
A. $2 \times 2$
B. $3 \times 3$
C. $4 \times 4$
D. $2.5 \times 2.5$
Correct Answer: A
Type: Numerical
Difficulty: Hard