Here are 100 MCQs covering Unit 5 – Autoencoders & Generative Models for Unsupervised Deep Learning, aligned with Course Outcome 5 (CO5).

Unit: 5
CO: CO5
Q1. Which of the following best describes the primary goal of a basic autoencoder?
A. To classify input images into distinct categories.
B. To learn a compressed representation of the input data and reconstruct it.
C. To generate completely new data samples from random noise without input.
D. To predict the next word in a sequence using attention mechanisms.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q2. In the context of machine learning, what is the fundamental difference between discriminative and generative models?
A. Discriminative models model $P(X|Y)$, while generative models model $P(Y|X)$.
B. Discriminative models learn the decision boundary between classes, while generative models model the distribution of the data itself.
C. Generative models are always supervised, while discriminative models are always unsupervised.
D. Discriminative models require less data than generative models.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q3. Which component of an autoencoder is responsible for mapping the input data to a lower-dimensional latent space?
A. The Decoder
B. The Discriminator
C. The Encoder
D. The Generator
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q4. In a standard autoencoder, the layer between the encoder and decoder that holds the compressed representation is commonly called the:
A. Output layer
B. Bottleneck or latent vector
C. Softmax layer
D. Attention head
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q5. If an autoencoder has a latent dimension larger than the input dimension, it is referred to as:
A. Undercomplete
B. Overcomplete
C. Sparse
D. Variational
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q6. Which loss function is most commonly used for a standard autoencoder trained on continuous image pixel data?
A. Categorical Crossentropy
B. Hinge Loss
C. Mean Squared Error (MSE)
D. Kullback-Leibler Divergence
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q7. Consider the following Keras code snippet for an encoder. What is the number of parameters in the `Dense` layer?
```python
input_img = tf.keras.Input(shape=(784,))
encoded = tf.keras.layers.Dense(32, activation='relu')(input_img)
```
A. 25,088
B. 25,120
C. 784
D. 32
Correct Answer: B
Type: Numerical
Difficulty: Easy

Unit: 5
CO: CO5
Q8. What is the output shape of the `encoded` tensor in the code below?
```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation='relu')
])
```
A. (None, 28, 28)
B. (None, 784)
C. (None, 64)
D. (None, 128)
Correct Answer: C
Type: Code
Difficulty: Easy

Unit: 5
CO: CO5
Q9. In a Denoising Autoencoder (DAE), what is the input to the network during training?
A. The original clean image.
B. A corrupted version of the original image (e.g., with added noise).
C. Random Gaussian noise unrelated to the image.
D. The target class label of the image.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q10. What is the target output (ground truth) used to calculate the loss for a Denoising Autoencoder?
A. The corrupted input image.
B. The original clean image.
C. A vector of zeros.
D. The class label.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q11. Which layer is typically used in the decoder of a Convolutional Autoencoder to increase the spatial dimensions of the feature maps?
A. `Conv2D` with `strides=2`
B. `MaxPooling2D`
C. `Conv2DTranspose` (or UpSampling2D)
D. `Flatten`
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q12. Look at the following decoder snippet. What is the shape of `x` after the reshape layer?
```python
decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(7 * 7 * 32, activation='relu', input_shape=(64,)),
    tf.keras.layers.Reshape((7, 7, 32))
])
```
A. (None, 1568)
B. (None, 7, 7, 32)
C. (None, 32, 7, 7)
D. (None, 49, 32)
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 5
CO: CO5
Q13. In a Variational Autoencoder (VAE), the encoder outputs:
A. A single fixed vector $z$.
B. Parameters of a probability distribution (mean $\mu$ and log-variance $\log\sigma^2$).
C. A reconstruction of the input.
D. A binary classification score.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 5
CO: CO5
Q14. The loss function of a VAE consists of two terms: the reconstruction loss and the:
A. Adversarial loss.
B. Perceptual loss.
C. Kullback-Leibler (KL) Divergence.
D. Mean Absolute Error.
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q15. Why is the "reparameterization trick" ($z = \mu + \sigma \odot \epsilon$) used in VAEs?
A. To increase the capacity of the model.
B. To allow backpropagation through the stochastic sampling process.
C. To ensure the latent space is perfectly Gaussian.
D. To reduce the number of parameters in the decoder.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 5
CO: CO5
Q16. Calculate the total number of parameters in this simple autoencoder:
Input size: 10
Encoder Dense Layer: 5 units (with bias)
Decoder Dense Layer: 10 units (with bias)
A. 50
B. 105
C. 115
D. 65
Correct Answer: C
Type: Numerical
Difficulty: Medium

Unit: 5
CO: CO5
Q17. Which Keras layer is usually used to implement the reparameterization trick in a VAE?
A. `tf.keras.layers.Sampling`
B. `tf.keras.layers.Lambda`
C. `tf.keras.layers.GaussianNoise`
D. `tf.keras.layers.Dense`
Correct Answer: B
Type: Code
Difficulty: Medium

Unit: 5
CO: CO5
Q18. In a Generative Adversarial Network (GAN), what is the goal of the Generator?
A. To classify images as real or fake.
B. To compress the input data.
C. To create synthetic data that is indistinguishable from real data.
D. To minimize the KL divergence.
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q19. In a GAN, what is the goal of the Discriminator?
A. To generate realistic images.
B. To distinguish between real data samples and fake samples produced by the generator.
C. To encode data into a latent space.
D. To maximize the generator's loss.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q20. If a GAN is trained successfully to equilibrium, what should the discriminator's output probability be for both real and fake inputs?
A. 1.0 (100% confidence)
B. 0.0 (0% confidence)
C. 0.5 (Random guessing)
D. 0.9 for real, 0.1 for fake
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 5
CO: CO5
Q21. What is the "min-max" game in GANs?
A. The generator minimizes the error, while the discriminator minimizes the accuracy.
B. The generator tries to minimize the probability that the discriminator is correct, while the discriminator tries to maximize it.
C. Both networks try to maximize the likelihood of the data.
D. The generator maximizes the reconstruction loss.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 5
CO: CO5
Q22. Consider the following Generator code. What is the shape of the output image?
```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(7*7*128, input_dim=100))
model.add(tf.keras.layers.Reshape((7, 7, 128)))
model.add(tf.keras.layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same'))
model.add(tf.keras.layers.Conv2DTranspose(1, (4,4), strides=(2,2), padding='same'))
```
A. (7, 7, 1)
B. (14, 14, 64)
C. (28, 28, 1)
D. (28, 28, 64)
Correct Answer: C
Type: Numerical
Difficulty: Medium

Unit: 5
CO: CO5
Q23. In the code above (Q22), if the input `latent_dim` is 100, how many parameters does the first `Dense` layer have (including bias)?
A. $100 \times 128$
B. $100 \times (7 \times 7 \times 128)$
C. $(100 + 1) \times (7 \times 7 \times 128)$
D. $7 \times 7 \times 128$
Correct Answer: C
Type: Numerical
Difficulty: Medium

Unit: 5
CO: CO5
Q24. A common problem in GAN training where the generator produces only a limited variety of samples (e.g., only one type of digit) is called:
A. Overfitting
B. Mode Collapse
C. Vanishing Gradient
D. Exploding Gradient
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 5
CO: CO5
Q25. Which activation function is typically used in the final layer of a Generator producing images normalized to the range [-1, 1]?
A. Sigmoid
B. ReLU
C. Tanh
D. Softmax
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 5
CO: CO5
Q26. Which activation function is typically used in the final layer of a standard Discriminator (binary classifier)?
A. Sigmoid
B. ReLU
C. Tanh
D. Linear
Correct Answer: A
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q27. Identify the error in this Discriminator model definition for a standard GAN.
```python
discriminator = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='softmax') # Error is here
])
```
A. Flatten layer cannot take input shape.
B. Dense layer units should be higher.
C. Final activation should be 'sigmoid' for binary classification, not 'softmax'.
D. There is no error.
Correct Answer: C
Type: Code
Difficulty: Easy

Unit: 5
CO: CO5
Q28. What is the purpose of `LeakyReLU` in a GAN Discriminator?
A. To ensure outputs are always positive.
B. To allow a small gradient when the unit is not active, preventing "dead" neurons.
C. To normalize the batch.
D. To output probabilities.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 5
CO: CO5
Q29. In a Convolutional Autoencoder, if an input of shape (28, 28, 1) is passed through `Conv2D(16, (3,3), strides=2, padding='same')`, what is the output shape?
A. (14, 14, 16)
B. (28, 28, 16)
C. (13, 13, 16)
D. (12, 12, 16)
Correct Answer: A
Type: Numerical
Difficulty: Medium

Unit: 5
CO: CO5
Q30. If the bottleneck layer of an autoencoder has 2 neurons and the input has 784 neurons, the compression ratio is:
A. 2 : 1
B. 392 : 1
C. 784 : 1
D. 1 : 392
Correct Answer: B
Type: Numerical
Difficulty: Easy

Unit: 5
CO: CO5
Q31. Which of the following is a generative model architecture based on attention mechanisms?
A. ResNet
B. Transformer (e.g., GPT)
C. Random Forest
D. Support Vector Machine
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q32. In the context of VAEs, the KL divergence term forces the learned latent distribution to be close to:
A. A Uniform distribution.
B. A Standard Normal distribution ($\mathcal{N}(0, I)$).
C. The distribution of the input data.
D. A Bernoulli distribution.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 5
CO: CO5
Q33. What does the following TensorFlow/Keras code snippet calculate?
```python
# z_mean and z_log_var are outputs from encoder
kl_loss = -0.5 * tf.reduce_mean(
    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
)
```
A. Binary Crossentropy Loss
B. Mean Squared Error
C. KL Divergence Loss for VAE
D. GAN Discriminator Loss
Correct Answer: C
Type: Code
Difficulty: Medium

Unit: 5
CO: CO5
Q34. In a VAE, if `z_mean` has shape (32, 2) and `z_log_var` has shape (32, 2), what is the shape of the sampled latent vector `z`?
A. (32, 4)
B. (32, 2)
C. (64, 1)
D. (1, 2)
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 5
CO: CO5
Q35. Which of the following is NOT an application of Generative AI?
A. Image Super-resolution
B. Text-to-Image Synthesis
C. Linear Regression for house price prediction
D. Music Composition
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q36. What is the role of `UpSampling2D` in an autoencoder decoder?
A. To reduce the spatial dimensions.
B. To increase the spatial dimensions by repeating rows and columns.
C. To add noise to the image.
D. To apply a convolution operation.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q37. Calculate the number of parameters in a `Conv2D` layer with:
Filters: 10
Kernel size: 3x3
Input channels: 1
Biases: True
A. 90
B. 91
C. 100
D. 10
Correct Answer: C
Type: Numerical
Difficulty: Medium

Unit: 5
CO: CO5
Q38. Why is `Conv2DTranspose` often preferred over `UpSampling2D` followed by `Conv2D` in GAN generators?
A. It is computationally cheaper.
B. It has learnable parameters that perform upsampling and convolution simultaneously.
C. It strictly repeats pixels without learning.
D. It reduces the number of channels.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 5
CO: CO5
Q39. What is the primary disadvantage of a basic autoencoder compared to a VAE for generating new data?
A. Basic autoencoders are harder to train.
B. The latent space of a basic AE is often discontinuous, making interpolation poor.
C. Basic autoencoders cannot reconstruct images.
D. Basic autoencoders require labelled data.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 5
CO: CO5
Q40. In a GAN, if the discriminator becomes too strong too quickly (perfect accuracy), what happens to the generator?
A. It learns very fast.
B. It stops learning because the gradients vanish (become near zero).
C. It overfits the data.
D. It collapses to a single mode.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 5
CO: CO5
Q41. Which code snippet correctly instantiates a Keras optimizer commonly used for GANs?
A. `optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)`
B. `optimizer = tf.keras.losses.BinaryCrossentropy()`
C. `optimizer = tf.keras.layers.Dense(1)`
D. `optimizer = tf.keras.optimizers.SGD(momentum=2.0)`
Correct Answer: A
Type: Code
Difficulty: Easy

Unit: 5
CO: CO5
Q42. What represents the "noise" input to a GAN generator?
A. Real images from the dataset.
B. A vector of random numbers (e.g., from a normal distribution).
C. The labels of the dataset.
D. The gradients from the discriminator.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q43. A Conditional GAN (cGAN) differs from a standard GAN by:
A. Removing the discriminator.
B. Feeding label information to both generator and discriminator.
C. Using only fully connected layers.
D. Training the generator only.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 5
CO: CO5
Q44. Given a batch of 64 images of size 28x28 flattened to 784 vectors, what is the shape of the input tensor to a dense autoencoder?
A. (28, 28, 64)
B. (64, 784)
C. (784, 64)
D. (64, 28, 28)
Correct Answer: B
Type: Numerical
Difficulty: Easy

Unit: 5
CO: CO5
Q45. In the context of Autoencoders, "reconstruction error" refers to:
A. The difference between the encoder input and the latent vector.
B. The difference between the original input and the decoder output.
C. The difference between real and fake images.
D. The accuracy of the classifier.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q46. Which function is used to calculate the binary crossentropy loss in a GAN using TensorFlow?
A. `tf.keras.losses.MeanSquaredError()`
B. `tf.keras.losses.BinaryCrossentropy(from_logits=True)`
C. `tf.keras.losses.CategoricalCrossentropy()`
D. `tf.keras.losses.KLDivergence()`
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 5
CO: CO5
Q47. If a VAE encoder outputs `z_mean` and `z_log_var` of dimension 10, how many values does the network output in total for these parameters?
A. 10
B. 20
C. 100
D. 2
Correct Answer: B
Type: Numerical
Difficulty: Easy

Unit: 5
CO: CO5
Q48. Which of the following is a "Generative" task?
A. Predicting if an email is spam.
B. Creating a new portrait in the style of Van Gogh.
C. Identifying objects in a video feed.
D. Forecasting stock prices.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q49. When training a GAN, the labels for the "real" images passed to the discriminator are usually set to:
A. 0
B. 1
C. 0.5
D. -1
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q50. When training the Generator in a GAN, what label is passed to the Discriminator to calculate the generator's loss?
A. 0 (Fake)
B. 1 (Real) - to trick the discriminator
C. 0.5 (Unsure)
D. Random labels
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 5
CO: CO5
Q51. Consider this autoencoder code. What is the compression factor of the latent space relative to the input?
```python
input_dim = 1000
latent_dim = 10
encoder = tf.keras.layers.Dense(latent_dim)(input)
```
A. 10x
B. 100x
C. 50x
D. 1000x
Correct Answer: B
Type: Numerical
Difficulty: Easy

Unit: 5
CO: CO5
Q52. In an autoencoder used for anomaly detection, anomalies are identified by:
A. High classification confidence.
B. High reconstruction error.
C. Low reconstruction error.
D. The encoder output being zero.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q53. Which TensorFlow operation is used to sample random noise from a normal distribution?
A. `tf.random.uniform`
B. `tf.random.normal`
C. `tf.zeros`
D. `tf.ones`
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 5
CO: CO5
Q54. What is the correct way to define a custom training step in Keras for a GAN?
A. Override the `train_step` method in a subclass of `tf.keras.Model`.
B. Use `model.fit()` with default settings.
C. Use `model.evaluate()`.
D. Override the `call` method only.
Correct Answer: A
Type: Code
Difficulty: Medium

Unit: 5
CO: CO5
Q55. In a VAE, if the KL divergence weight is too high during training, what happens?
A. The reconstruction quality becomes perfect.
B. The latent distribution collapses to the prior (Posterior collapse), ignoring the input data.
C. The model generates extremely diverse images.
D. The training diverges immediately.
Correct Answer: B
Type: Theory
Difficulty: Hard

Unit: 5
CO: CO5
Q56. Which of the following architectures is explicitly designed to handle sequential data generation (like text)?
A. Convolutional Autoencoder
B. Transformer (Decoder-only or Encoder-Decoder)
C. Standard GAN
D. Multilayer Perceptron
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q57. In a WGAN (Wasserstein GAN), the discriminator is often referred to as a:
A. Classifier
B. Critic
C. Encoder
D. Decoder
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 5
CO: CO5
Q58. Code Calculation: A decoder has an input size of (4, 4, 128) and applies `Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')`. What is the output height/width?
A. 4x4
B. 6x6
C. 8x8
D. 16x16
Correct Answer: C
Type: Numerical
Difficulty: Medium

Unit: 5
CO: CO5
Q59. How many parameters does a standard BatchNormalization layer have if the input feature map has 64 channels? (Consider Gamma, Beta, Moving Mean, Moving Variance).
A. 64
B. 128
C. 256
D. 4
Correct Answer: C
Type: Numerical
Difficulty: Hard

Unit: 5
CO: CO5
Q60. Which technique helps stabilize GAN training by smoothing the labels (e.g., using 0.9 instead of 1.0 for real data)?
A. Gradient Clipping
B. Label Smoothing
C. Batch Normalization
D. Dropout
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 5
CO: CO5
Q61. In the context of Deepfakes, which technology is primarily used?
A. Linear Regression
B. GANs (Generative Adversarial Networks)
C. K-Means Clustering
D. PCA
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q62. What is the output of `tf.keras.layers.Flatten()(x)` if `x` has shape `(None, 10, 10, 3)`?
A. (None, 30)
B. (None, 100)
C. (None, 300)
D. (None, 1000)
Correct Answer: C
Type: Code
Difficulty: Easy

Unit: 5
CO: CO5
Q63. What is the main benefit of using a VAE over a GAN?
A. VAEs generate sharper images.
B. VAEs provide an explicit density estimation and a structured latent space.
C. VAEs are faster to train.
D. VAEs do not use neural networks.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 5
CO: CO5
Q64. In TensorFlow, how do you prevent the discriminator weights from updating while training the generator?
A. Set `discriminator.trainable = False` before compiling the combined GAN model.
B. Set `generator.trainable = False`.
C. Use `tf.stop_gradient`.
D. It happens automatically.
Correct Answer: A
Type: Code
Difficulty: Medium

Unit: 5
CO: CO5
Q65. Which loss function is used in WGAN to replace binary crossentropy?
A. Wasserstein Loss (difference between average scores).
B. Mean Squared Error.
C. Categorical Crossentropy.
D. Hinge Loss.
Correct Answer: A
Type: Theory
Difficulty: Hard

Unit: 5
CO: CO5
Q66. Which architecture is typically used for "seq2seq" tasks like translation in Generative AI?
A. Encoder-Decoder Transformer
B. CNN
C. Dense Autoencoder
D. Isolation Forest
Correct Answer: A
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q67. A code snippet for a Dense layer is `Dense(100, activation='relu', input_shape=(50,))`. How many trainable parameters (weights + biases)?
A. 500
B. 5000
C. 5100
D. 50100
Correct Answer: C
Type: Numerical
Difficulty: Medium

Unit: 5
CO: CO5
Q68. What is "Mode Collapse" in GANs?
A. The generator produces a single or very few types of outputs regardless of noise input.
B. The discriminator accuracy drops to 0.
C. The training takes too long.
D. The model parameters become NaN.
Correct Answer: A
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q69. Which layer is typically used to flatten the 2D output of a Convolutional Discriminator before the final classification?
A. `GlobalAveragePooling2D` or `Flatten`
B. `UpSampling2D`
C. `Conv2DTranspose`
D. `Reshape`
Correct Answer: A
Type: Code
Difficulty: Easy

Unit: 5
CO: CO5
Q70. In a VAE, the "Encoder" is often probabilistically referred to as the:
A. Prior network
B. Inference network (approximating posterior $q_\phi(z|x)$)
C. Generative network
D. Discriminator
Correct Answer: B
Type: Theory
Difficulty: Hard

Unit: 5
CO: CO5
Q71. Calculate the size of the latent vector if the encoder output is a feature map of $4 \times 4 \times 512$ and is flattened.
A. 2048
B. 8192
C. 512
D. 16
Correct Answer: B
Type: Numerical
Difficulty: Easy

Unit: 5
CO: CO5
Q72. What is the purpose of the `padding='same'` argument in `Conv2D`?
A. To discard the borders of the image.
B. To ensure the output spatial dimensions match the input (if stride=1).
C. To reduce the output size.
D. To add random noise to the borders.
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 5
CO: CO5
Q73. Which of the following is an "autoregressive" generative model?
A. GPT (Generative Pre-trained Transformer)
B. GAN
C. VAE
D. K-Nearest Neighbors
Correct Answer: A
Type: Theory
Difficulty: Medium

Unit: 5
CO: CO5
Q74. In Keras, what does `model.summary()` display?
A. The training graphs.
B. The layer architecture, output shapes, and parameter counts.
C. The predictions on test data.
D. The loss values.
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 5
CO: CO5
Q75. Why is a "bottleneck" necessary in an autoencoder?
A. To speed up training.
B. To force the model to learn meaningful features/compression rather than copying the input.
C. To increase the number of parameters.
D. To allow for classification.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q76. If you want to generate images of digits 0-9 specifically on command using a GAN, which variant should you use?
A. Basic GAN
B. Conditional GAN (cGAN)
C. Denoising Autoencoder
D. WGAN
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q77. Code snippet:
```python
x = tf.keras.layers.Conv2D(32, 3, activation='relu')(input)
x = tf.keras.layers.MaxPooling2D()(x)
```
If input is (28, 28, 1), what is the shape after `MaxPooling2D` (default pool size 2x2)?
A. (26, 26, 32)
B. (14, 14, 32)
C. (13, 13, 32)
D. (28, 28, 32)
Correct Answer: C
Type: Numerical
Difficulty: Hard

Unit: 5
CO: CO5
Q78. What is the typical range of pixel values for images fed into a GAN Generator using `tanh` activation?
A.
B.
C. [-1, 1]
D. [-∞, +∞]
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q79. Which metric is commonly used to evaluate the quality of GAN-generated images?
A. Accuracy
B. F1-Score
C. Fréchet Inception Distance (FID)
D. Mean Squared Error
Correct Answer: C
Type: Theory
Difficulty: Hard

Unit: 5
CO: CO5
Q80. In the TensorFlow/Keras Functional API, how do you connect two layers `L1` and `L2`?
A. `L2(L1)`
B. `L2.connect(L1)`
C. `L1.output = L2`
D. `model.add(L1, L2)`
Correct Answer: A
Type: Code
Difficulty: Easy

Unit: 5
CO: CO5
Q81. A Sparse Autoencoder achieves sparsity by:
A. Removing connections randomly.
B. Adding a regularization term (e.g., L1) to the loss function penalizing activations in the hidden layer.
C. Using only 1 hidden neuron.
D. Using dropout.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 5
CO: CO5
Q82. Which statement is TRUE about Discriminative models?
A. They model the joint probability $P(X, Y)$.
B. They focus on finding the decision boundary between classes.
C. They can generate new data samples.
D. Autoencoders are discriminative models.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q83. Given `inputs = Input(shape=(784,))` and `h = Dense(64)(inputs)`, which line creates the decoding layer back to original size?
A. `outputs = Dense(64)(h)`
B. `outputs = Dense(784)(h)`
C. `outputs = Flatten()(h)`
D. `outputs = Conv2D(784)(h)`
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 5
CO: CO5
Q84. What is the gradient penalty used for in WGAN-GP?
A. To enforce the 1-Lipschitz constraint on the critic/discriminator.
B. To penalize the generator for bad images.
C. To reduce the learning rate.
D. To increase sparsity.
Correct Answer: A
Type: Theory
Difficulty: Hard

Unit: 5
CO: CO5
Q85. In a VAE, if the latent dimension is 2, what allows us to visualize the generation capabilities?
A. Plotting the loss curve.
B. Traversing the 2D latent plane (grid walk) and decoding points.
C. Checking the accuracy.
D. Using a confusion matrix.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 5
CO: CO5
Q86. Transformers revolutionized Generative AI primarily because of:
A. The use of Convolutional layers.
B. The Self-Attention mechanism allowing parallel processing of sequences.
C. The use of Sigmoid activations.
D. Small model sizes.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q87. Code: `tf.keras.layers.GaussianNoise(0.1)(x)` is commonly used in:
A. Basic Autoencoders
B. Denoising Autoencoders (at input)
C. The output layer of a Classifier
D. The loss function
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 5
CO: CO5
Q88. Calculate the number of output neurons needed for a decoder generating an RGB image of size 64x64.
A. 4096
B. 12288
C. 64
D. 3
Correct Answer: B
Type: Numerical
Difficulty: Medium

Unit: 5
CO: CO5
Q89. Which component in a GAN is analogous to an "Art Forger"?
A. Discriminator
B. Generator
C. Loss function
D. Optimizer
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q90. Which component in a GAN is analogous to an "Art Critic"?
A. Discriminator
B. Generator
C. Latent Vector
D. Activation Function
Correct Answer: A
Type: Theory
Difficulty: Easy

Unit: 5
CO: CO5
Q91. What is "Posterior Collapse" in VAEs?
A. The decoder ignores the latent code and generates generic data.
B. The encoder outputs NaNs.
C. The discriminator overpowers the generator.
D. The training loss becomes negative.
Correct Answer: A
Type: Theory
Difficulty: Hard

Unit: 5
CO: CO5
Q92. For a latent vector `z` of shape (Batch, 100), which layer is best suited to reshape it for the start of a Conv2DTranspose generator?
A. `Flatten`
B. `Dense` followed by `Reshape`
C. `MaxPooling2D`
D. `Dropout`
Correct Answer: B
Type: Code
Difficulty: Medium

Unit: 5
CO: CO5
Q93. If a Dense layer has 10 inputs and 5 outputs, the weight matrix shape is:
A. (10, 5)
B. (5, 10)
C. (10, 1)
D. (5, 5)
Correct Answer: A
Type: Numerical
Difficulty: Easy

Unit: 5
CO: CO5
Q94. Which of these is an example of an "Implicit Density" model?
A. VAE (approximates density)
B. GAN (does not explicitly define density function)
C. PixelRNN
D. Gaussian Mixture Model
Correct Answer: B
Type: Theory
Difficulty: Hard

Unit: 5
CO: CO5
Q95. In text generation with Transformers, what is "temperature"?
A. A parameter that controls the randomness of predictions (higher = more random).
B. The heat generated by the GPU.
C. The learning rate schedule.
D. The number of attention heads.
Correct Answer: A
Type: Theory
Difficulty: Medium

Unit: 5
CO: CO5
Q96. Code Debug: The following VAE loss returns an error. Why?
```python
def vae_loss(y_true, y_pred):
    recon = mse(y_true, y_pred)
    # kl_loss is calculated globally
    return recon + kl_loss
```
A. Keras losses must accept exactly two arguments (y_true, y_pred) and return a tensor per sample.
B. You cannot add losses.
C. MSE is not supported.
D. No error.
Correct Answer: A
Type: Code
Difficulty: Hard

Unit: 5
CO: CO5
Q97. What is the output shape of `Conv2DTranspose(32, (3,3), strides=(2,2), padding='same')` applied to input `(16, 16, 64)`?
A. (8, 8, 32)
B. (32, 32, 32)
C. (16, 16, 32)
D. (32, 32, 64)
Correct Answer: B
Type: Numerical
Difficulty: Medium

Unit: 5
CO: CO5
Q98. A "U-Net" architecture, often used in diffusion models and segmentation, is essentially:
A. A GAN.
B. An Autoencoder with skip connections between encoder and decoder levels.
C. A standard CNN classifier.
D. A Recurrent Neural Network.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 5
CO: CO5
Q99. What is the purpose of `noise = tf.random.normal([batch_size, 100])` in a GAN training loop?
A. To add regularization to weights.
B. To serve as the input seed for the Generator to create fake images.
C. To corrupt the real images for the Discriminator.
D. To initialize the biases.
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 5
CO: CO5
Q100. Which type of Autoencoder learns to map inputs to a binary code (0s and 1s)?
A. Variational Autoencoder
B. Semantic Hashing / Binary Autoencoder
C. Denoising Autoencoder
D. Linear Autoencoder
Correct Answer: B
Type: Theory
Difficulty: Hard