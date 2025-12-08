Unit: 1
CO: CO1
Q1. Deep Learning is fundamentally a specialized subfield of Machine Learning distinguished by which key architectural feature?
A. Reliance solely on statistical regression methods.
B. Use of decision trees for hierarchical data processing.
C. Utilization of artificial neural networks with multiple hidden layers.
D. Requirement for exclusively small, structured datasets.
Correct Answer: C
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q2. What is the primary role of the optimizer algorithm in training a deep neural network?
A. To define the complexity of the network architecture.
B. To manually select the input features for generalization.
C. To iteratively refine model parameters (weights/biases) to minimize the loss function.
D. To convert text inputs into numerical token representations.
Correct Answer: C
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q3. In an Artificial Neural Network, what are weights and biases collectively referred to, as defined by their function in the learning process?
A. Activation states.
B. Hyperparameters.
C. Learnable parameters.
D. Gradient anchors.
Correct Answer: C
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q4. What is the function of the loss function (or cost function) in a deep learning model?
A. To calculate the learning rate dynamically during training.
B. To calculate the error, defined as the discrepancy between the predicted and actual values.
C. To initialize the weights randomly before the first training epoch.
D. To define the computational complexity of the hidden layers.
Correct Answer: B
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q5. Which characteristic typically necessitates the selection of a Deep Learning approach over classical Machine Learning algorithms?
A. Availability of highly structured CSV data.
B. Preference for easier model interpretability.
C. Handling unstructured data like images, audio, or raw text.
D. Need for faster training times on a CPU.
Correct Answer: C
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q6. When describing the training process of a deep learning model, what does one "epoch" represent?
A. A single forward and backward pass through a minibatch of data.
B. The total number of iterations required for the model to converge.
C. The number of times the optimization algorithm runs across the entire training dataset.
D. The process of tuning learning rate decay.
Correct Answer: C
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q7. What is the fundamental purpose of the Backpropagation algorithm?
A. To select the optimal initial weights for the network.
B. To efficiently compute the gradient of the loss function with respect to every weight in the network.
C. To define the non-linear transformation applied at each neuron.
D. To adjust the batch size dynamically during training.
Correct Answer: B
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q8. A neural network is classified as "Deep" primarily because it employs:
A. The Sigmoid activation function.
B. Convolutional layers exclusively.
C. Multiple successive transformations (layers) of the input data.
D. Stochastic Gradient Descent for training.
Correct Answer: C
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q9. In classification tasks, how is the model's performance commonly measured using "accuracy"?
A. The variance of the loss across all epochs.
B. The proportion of examples for which the model produces the correct output.
C. The mean squared difference between predicted logits and true labels.
D. The total cumulative training time in seconds.
Correct Answer: B
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q10. Which loss function is conventionally used as the objective for multi-class classification problems, particularly when modeling conditional probabilities via Softmax?
A. Mean Squared Error (MSE).
B. L1 Norm Loss.
C. Cross-Entropy Loss (Log Loss).
D. Huber Loss.
Correct Answer: C
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q11. What is the primary functional range of the Sigmoid activation function?
A. $(-\infty, \infty)$
B. $[-1, 1]$
C. $(0, 1)$
D. $[0, \infty)$
Correct Answer: C
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q12. The Tanh (Hyperbolic Tangent) activation function squashes its input values into which range?
A. $(-\infty, \infty)$
B. $(0, 1)$
C. $[0, \infty)$
D. $(-1, 1)$
Correct Answer: D
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q13. Which activation function has largely become the default recommendation for hidden layers in modern deep neural networks due to its simplicity and robust gradient properties?
A. Tanh.
B. Sigmoid.
C. Softmax.
D. ReLU (Rectified Linear Unit).
Correct Answer: D
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q14. In training a large model, when resources permit, Mini-Batch Gradient Descent is generally preferred over full Batch Gradient Descent because:
A. It guarantees faster convergence to the global minimum.
B. It reduces computational cost and allows for efficient vectorization.
C. It eliminates the need for activation functions.
D. It avoids numerical instability inherently.
Correct Answer: B
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q15. In deep learning, techniques like L1 and L2 penalties are applied primarily to mitigate which issue?
A. Vanishing gradients.
B. Local minima.
C. Overfitting.
D. Non-differentiability of the loss function.
Correct Answer: C
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q16. A B.Tech student attempts to classify Fashion-MNIST using a simple Keras model. What is the output shape (excluding batch dimension) after this snippet, assuming the input shape is (784,)?
A. (784,)
B. (64,)
C. (10,)
D. (784, 64)
Correct Answer: B
Type: Code
Difficulty: Easy
Unit: 1
CO: CO1
Q17. Calculate the total number of trainable parameters in a tf.keras.layers.Dense layer having 512 input features and 256 output units, assuming both weights ($W$) and biases ($b$) are included.
A. 131,072
B. 131,328
C. 131,584
D. 196,864
Correct Answer: B
Type: Numerical
Difficulty: Easy
Unit: 1
CO: CO1
Q18. In a CNN applied to a 28x28 grayscale image, the feature map size after the last convolutional block is (1, 5, 5, 32) (Batch, Height, Width, Channels). What is the output shape after applying tf.keras.layers.Flatten()?
A. (5, 5, 32)
B. (32,)
C. (1, 800)
D. (80,)
Correct Answer: D
Type: Code
Difficulty: Easy
Unit: 1
CO: CO1
Q19. A simple feedforward network has an input size of 10. The first hidden layer (H1) has 32 units, and the output layer (O1) has 1 unit. Calculate the total number of weights and biases in the network.
A. 320
B. 353
C. 32
D. 352
Correct Answer: B
Type: Numerical
Difficulty: Easy
Unit: 1
CO: CO1
Q20. If a model predicts $\hat{Y} = 4.5$ and the true value is $Y = 5.5$, what is the Mean Squared Error (MSE) for this single observation, assuming the standard definition $L=(Y-\hat{Y})^2$?
A. 1.0
B. 0.5
C. 2.0
D. 0.0
Correct Answer: A
Type: Numerical
Difficulty: Easy
Unit: 1
CO: CO1
Q21. A convolutional layer uses a 3x3 kernel. If the input image is 20x20 and no padding or stride is used, what are the dimensions of the output feature map? (Excluding channel and batch dimensions)
A. 20x20
B. 18x18
C. 22x22
D. 19x19
Correct Answer: B
Type: Numerical
Difficulty: Easy
Unit: 1
CO: CO1
Q22. When evaluating a model, a researcher observes the output of a tf.keras.layers.Dropout(0.5) layer. What is the expected behavior of this layer during this evaluation (prediction) phase?
A. It randomly sets 50% of the inputs to zero.
B. It scales all inputs by multiplying by 0.5.
C. It passes all inputs through without modification or scaling.
D. It applies a ReLU activation only to 50% of the inputs.
Correct Answer: C
Type: Code
Difficulty: Easy
Unit: 1
CO: CO1
Q23. An input feature map of size 32x32 is passed through a Max Pooling layer with a pool size of 2x2 and a stride of 2. What is the size of the output feature map?
A. 31x31
B. 64x64
C. 16x16
D. 8x8
Correct Answer: C
Type: Numerical
Difficulty: Easy
Unit: 1
CO: CO1
Q24. If a training set error is high and the validation set error is similarly high, indicating that the model cannot capture the fundamental relationship in the data, what issue is the model exhibiting?
A. High Variance (Overfitting).
B. Low Variance (Equilibrium).
C. High Bias (Underfitting).
D. Exploding Gradient.
Correct Answer: C
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q25. Which optimizer is frequently recommended as the default starting point for training deep neural networks due to its robust combination of momentum and adaptive learning rates?
A. Stochastic Gradient Descent (SGD).
B. Adagrad.
C. RMSProp.
D. Adam.
Correct Answer: D
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q26. Why is a dedicated validation set required during hyperparameter tuning (HPO)?
A. To calculate the final, unbiased generalization error reported to stakeholders.
B. To prevent the hyperparameters from being chosen based on test set performance, avoiding test set overfitting.
C. To be used as input during the forward propagation training step.
D. To ensure the model architecture is deep enough.
Correct Answer: B
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q27. When training deep recurrent neural networks, which inelegant but effective heuristic is often employed to mitigate the instability caused by excessively large gradients?
A. Early Stopping.
B. Gradient Clipping.
C. Weight Decay.
D. Learning Rate Warmup.
Correct Answer: B
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q28. In contrast to traditional Gradient Descent, Stochastic Gradient Descent (SGD) introduces an element of randomness primarily by:
A. Randomly initializing all weight parameters.
B. Sampling only a single data example (or a small batch) per update iteration.
C. Using a random learning rate in each step.
D. Applying random activation functions.
Correct Answer: B
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q29. For large datasets where computing the gradient over all samples is prohibitively slow, what is the key practical advantage of using SGD or Mini-Batch GD over full Batch GD?
A. Lower memory requirements for the weight parameters.
B. Constant O(1) computational cost per update iteration, regardless of dataset size.
C. Guaranteed convergence to the global minimum.
D. Increased model interpretability.
Correct Answer: B
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q30. What is the fundamental concept underlying optimization algorithms like Momentum that enables them to accelerate convergence, especially on ill-conditioned loss landscapes?
A. Resetting the learning rate periodically.
B. Incorporating an exponentially weighted average of past gradients (velocity).
C. Dynamically calculating the second derivative (Hessian).
D. Using one-hot encoding for inputs.
Correct Answer: B
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q31. Consider a neural network designed for a sentiment classification task with 10 output classes. If the last tf.keras.layers.Dense layer has 512 input units, calculate the number of weights (excluding biases) in this final layer.
A. 5120
B. 5130
C. 512
D. 522
Correct Answer: A
Type: Numerical
Difficulty: Easy
Unit: 1
CO: CO1
Q32. When defining the Mean Squared Error (MSE) loss for gradient computation in deep learning frameworks, the term is often multiplied by $1/2$. What is the primary purpose of this factor?
A. To guarantee that the loss is always positive.
B. To ensure that the error is equivalent to the $L_2$ norm.
C. To simplify the derivative calculation, canceling the factor of 2 when taking the derivative of $(Y-\hat{Y})^2$.
D. To reduce the computational complexity from $O(N^2)$ to $O(N)$.
Correct Answer: C
Type: Numerical
Difficulty: Easy
Unit: 1
CO: CO1
Q33. Which option correctly sets the output layer of a Keras sequential model designed for binary classification, modeling the probability of the positive class?
A. model.add(Dense(1, activation='softmax'))
B. model.add(Dense(2, activation='sigmoid'))
C. model.add(Dense(1, activation='sigmoid'))
D. model.add(Dense(2, activation='relu'))
Correct Answer: C
Type: Code
Difficulty: Easy
Unit: 1
CO: CO1
Q34. An input tensor has a spatial size of $H \times W$. If a convolutional operation uses a kernel size $K$, and padding $P = (K-1)/2$ (assuming $K$ is odd) and stride $S=1$, what is the size of the output tensor?
A. $(H-K+1) \times (W-K+1)$
B. $(H+P) \times (W+P)$
C. $H \times W$
D. $(H/2) \times (W/2)$
Correct Answer: C
Type: Numerical
Difficulty: Easy
Unit: 1
CO: CO1
Q35. Which of the following snippets correctly defines a sequential model in Keras using the add() method?
A.
B.
C.
D.
Correct Answer: A
Type: Code
Difficulty: Easy
Unit: 1
CO: CO1
Q36. When using Cross-Entropy Loss for a binary classification problem, which output prediction (logit, $o$) for the true label ($y=1$) will result in the highest loss value?
A. $\hat{Y} = 0.9$ (High probability of correct class)
B. $\hat{Y} = 0.5$ (Uncertainty)
C. $\hat{Y} = 0.1$ (Low probability of correct class)
D. $\hat{Y} = 0.0001$ (Near impossible probability of correct class)
Correct Answer: D
Type: Numerical
Difficulty: Easy
Unit: 1
CO: CO1
Q37. Two vectors $\mathbf{v}$ and $\mathbf{w}$ are considered orthogonal if their dot product $\mathbf{v} \cdot \mathbf{w}$ equals what value?
A. 1
B. $-1$
C. 0
D. The product of their norms.
Correct Answer: C
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q38. If the input to a ReLU activation function is $x = 5.0$, what is the value of the derivative $\frac{d(\text{ReLU}(x))}{dx}$ during the backward pass?
A. 0
B. 5.0
C. 1
D. -1
Correct Answer: C
Type: Numerical
Difficulty: Easy
Unit: 1
CO: CO1
Q39. A programmer creates a tf.keras.layers.Dense(64) layer instance but does not specify the input shape and does not call the layer yet. If the programmer inspects layer.weights, what will they find?
A. A list containing initialized weights of shape $(?, 64)$.
B. A list containing only bias weights, as kernel weights require input shape.
C. An empty list, as weights are only created when the input shape is first determined.
D. A list containing weights initialized to zeros.
Correct Answer: C
Type: Code
Difficulty: Easy
Unit: 1
CO: CO1
Q40. What is the defining characteristic that separates data used for Supervised Learning from data used for Unsupervised Learning?
A. Supervised data is exclusively numerical; unsupervised data is categorical.
B. Supervised data contains features and corresponding labels/targets.
C. Unsupervised data must be drawn from an i.i.d. distribution.
D. Unsupervised data must be high-dimensional.
Correct Answer: B
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q41. You are performing Gradient Descent on a convex function $L(w) = w^2$. If the current weight $w_0 = 5.0$ and the learning rate $\eta = 0.1$, calculate the updated weight $w_1$.
A. $4.0$
B. $4.5$
C. $4.9$
D. $5.1$
Correct Answer: A
Type: Numerical
Difficulty: Easy
Unit: 1
CO: CO1
Q42. Deep learning relies heavily on the abundance of data and high computational power (GPUs). In the 1990s, when data and compute were scarce, which type of ML algorithm was often empirically preferred due to its predictable results and strong theoretical guarantees?
A. Recurrent Neural Networks (RNNs).
B. Linear Models and Kernel Methods.
C. Generative Adversarial Networks (GANs).
D. Deep Multi-layer Perceptrons (MLPs).
Correct Answer: B
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q43. Batch Normalization layers operate by normalizing inputs based on minibatch statistics. This layer is beneficial because it helps ensure:
A. The loss function is always convex.
B. The parameters are always initialized to zero.
C. Variables in intermediate layers maintain stable means and variances during training.
D. The gradient is always perfectly smooth.
Correct Answer: C
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q44. What factor is typically required by Deep Learning models but often unnecessary for classical Machine Learning models?
A. The use of a loss function.
B. The use of a backpropagation algorithm.
C. Large volumes of data (in the millions).
D. The ability to perform regression.
Correct Answer: C
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q45. When initializing the weights of a neural network, why must the weights be initialized randomly (e.g., using Gaussian noise) rather than setting all weights to the same constant (like zero or one)?
A. To guarantee the model is a Universal Approximator.
B. To ensure that the training error starts high.
C. To break symmetry among hidden units so they can learn different representations.
D. To prevent the loss function from being non-convex.
Correct Answer: C
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q46. The Mean Squared Error (MSE) of an estimator $\hat{\theta}$ can be decomposed into the sum of the estimator's variance and its squared bias. For an unbiased estimator, what does the MSE simplify to?
A. The square of the bias.
B. The variance of the estimator.
C. Zero.
D. The inverse of the learning rate.
Correct Answer: B
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q47. If a tf.keras.layers.Dropout(0.2) layer is actively used during model training, what is the expected scale of the remaining (non-dropped out) neuron outputs?
A. Outputs are multiplied by 0.2.
B. Outputs are multiplied by 0.8.
C. Outputs are multiplied by 1.25.
D. Outputs remain unscaled.
Correct Answer: C
Type: Numerical
Difficulty: Easy
Unit: 1
CO: CO1
Q48. Given a prediction $\hat{Y} = 7$ and a true label $Y = 3$. If the standard squared loss (excluding the $1/2$ factor) is calculated, what is the loss value?
A. 4
B. 8
C. 16
D. 0
Correct Answer: C
Type: Numerical
Difficulty: Easy
Unit: 1
CO: CO1
Q49. A feature vector $\mathbf{x} \in \mathbb{R}^n$ is processed by a layer. If a tf.keras.layers.LayerNormalization layer is applied to the output, across which dimension does it calculate the mean and variance for normalization?
A. Across the batch dimension.
B. Across the feature dimension.
C. Across time steps.
D. Across the entire dataset population.
Correct Answer: B
Type: Code
Difficulty: Easy
Unit: 1
CO: CO1
Q50. Which set of data must hyperparameters be tuned on to prevent the final model from overfitting to the evaluation data?
A. Training set.
B. Test set.
C. Validation set.
D. Entire population data.
Correct Answer: C
Type: Theory
Difficulty: Easy
Unit: 1
CO: CO1
Q51. A model trained on 10 epochs shows 95% training accuracy and 55% validation accuracy. The instructor suggests stopping training early. Which primary issue is the model exhibiting, and what is the standard solution?
A. High Bias (Underfitting); Solution: Use a simpler model.
B. High Variance (Overfitting); Solution: Apply Early Stopping or Dropout.
C. Vanishing Gradient; Solution: Switch to Sigmoid activation.
D. Low Capacity; Solution: Decrease the learning rate.
Correct Answer: B
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q52. L2 regularization (weight decay) limits the magnitude of weights during training. For an MLP using the Tanh activation function, how does a large L2 penalty help prevent overfitting?
A. It forces the output of Tanh into the non-linear saturation region.
B. It drives weights towards zero, causing $z$ values to be small, approximating Tanh with its central linear region.
C. It increases the dimensionality of the input vectors.
D. It ensures that the learning rate is always large.
Correct Answer: B
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q53. Given a smooth, convex objective function $f(w)$, if the learning rate $\eta$ is chosen too large for Gradient Descent, what pathological behavior is the optimization process likely to exhibit?
A. Vanishing gradient, leading to stalled progress.
B. Convergence to a non-zero minimum due to insufficient steps.
C. Oscillation or divergence, failing to smoothly approach the minimum.
D. Increased numerical instability in the loss calculation.
Correct Answer: C
Type: Numerical
Difficulty: Medium
Unit: 1
CO: CO1
Q54. Backpropagation is described as an efficient application of the chain rule. What core computational advantage does backpropagation exploit to outperform a naive, independent calculation of gradients for every weight?
A. It calculates the gradient exclusively through matrix multiplication (forward mode differentiation).
B. It computes gradients one layer at a time, iterating backward, avoiding redundant recalculations of intermediate terms.
C. It bypasses the need for activation function derivatives entirely.
D. It guarantees that the Hessian matrix remains sparse.
Correct Answer: B
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q55. A developer is designing a Keras model that requires a Residual Connection (a non-linear topology where a layer's output skips the next layer and is added to the output of the layer after that). Why is the tf.keras.Sequential model inappropriate for this structure?
A. Sequential models cannot use tf.keras.layers.Dense.
B. Sequential models must define the input shape in the first layer.
C. Sequential models can only handle a plain stack of layers where each layer has exactly one input and one output tensor.
D. Sequential models are restricted to using ReLU activation.
Correct Answer: C
Type: Code
Difficulty: Medium
Unit: 1
CO: CO1
Q56. Mini-Batch Stochastic Gradient Descent (SGD) reduces the variance of the gradient estimate compared to using a single observation ($\text{batch size}=1$). If a batch size of $B$ is used, the resulting minibatch gradient is the average of $B$ independent gradients. By what factor is the statistical standard deviation of this average gradient reduced compared to the standard deviation of a single gradient?
A. $1/B$
B. $\sqrt{B}$
C. $1/\sqrt{B}$
D. $B^2$
Correct Answer: C
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q57. In minimizing ill-conditioned quadratic objectives (like a narrow, deep valley), standard Gradient Descent struggles due to oscillations in the steep directions and slow movement in the flat directions. How does the Momentum method fundamentally solve this issue?
A. It zeros out the gradient component in the steepest direction.
B. It accumulates gradients, canceling out oscillations while reinforcing consistent movement in shallow directions.
C. It dynamically sets the learning rate to be proportional to the loss magnitude.
D. It transforms the objective function into a perfectly spherical shape.
Correct Answer: B
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q58. What is the fundamental distinction between the RMSProp optimizer and the Adam optimizer?
A. RMSProp calculates adaptive learning rates based on the first moment (mean), while Adam only uses the learning rate decay schedule.
B. Adam uses exponential moving averages of both the first moment (mean) and the second moment (uncentered variance) of the gradient, while RMSProp only uses the second moment.
C. RMSProp requires manual tuning of the learning rate, while Adam sets the learning rate adaptively to zero.
D. Adam employs stochasticity, while RMSProp uses full-batch gradients.
Correct Answer: B
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q59. During training, the total norm of the aggregated gradient vector $\mathbf{g}$ is calculated as $|\mathbf{g}| = 4.0$. If the gradient clipping threshold $\theta$ is set to $1.0$, calculate the factor by which the gradient $\mathbf{g}$ is scaled down before updating the weights.
A. 1.0
B. 0.25
C. 4.0
D. 0.5
Correct Answer: B
Type: Numerical
Difficulty: Medium
Unit: 1
CO: CO1
Q60. A feature map of shape (Batch, 32, 32, 16) is input to a prediction module. This module uses a tf.keras.layers.Conv2D layer with 5 anchor boxes ($a=5$) and 3 object classes ($q=3$) for prediction. If the class prediction layer must output $a(q+1)$ channels, calculate the total number of channels required for the output feature map (excluding the batch dimension).
A. 15
B. 16
C. 20
D. 64
Correct Answer: C
Type: Numerical
Difficulty: Medium
Unit: 1
CO: CO1
Q61. A student writes a Keras model but forgets to specify the input shape in the first Dense layer. Which action must be taken to ensure model.summary() executes successfully without requiring the model to be called on actual input data first?
A. Use a large batch size in the training loop.
B. Add tf.keras.layers.Flatten() before the Dense layer.
C. Explicitly add a tf.keras.Input(shape=(...)) object or pass the input_shape argument to the first layer.
D. Switch the optimizer to Adam.
Correct Answer: C
Type: Code
Difficulty: Medium
Unit: 1
CO: CO1
Q62. The RMSProp state variable $\mathbf{s}_t$ uses a leaky average $\mathbf{s}t \leftarrow \gamma \mathbf{s}{t-1} + (1 - \gamma) \mathbf{g}_t^2$. If the hyperparameter $\gamma$ is set to $0.9$, the half-life time of an observation contributing to the average is roughly 10 steps. What is the effective sample size or number of past gradient squares contributing significantly to $\mathbf{s}_t$?
A. 1.0
B. $1 / (1 - \gamma)$
C. $100$
D. $1 / \sqrt{1 - \gamma}$
Correct Answer: B
Type: Numerical
Difficulty: Medium
Unit: 1
CO: CO1
Q63. For convex optimization problems using SGD, what theoretical constraint must the learning rate $\eta_t$ satisfy as the number of time steps $t$ increases to guarantee convergence to the optimal solution?
A. $\eta_t$ must remain constant.
B. $\eta_t$ must decay at least as quickly as $O(t^{-1/2})$.
C. $\eta_t$ must grow linearly with $t$.
D. $\eta_t$ must be proportional to the second derivative (Hessian).
Correct Answer: B
Type: Numerical
Difficulty: Medium
Unit: 1
CO: CO1
Q64. A single-channel input image of size $128 \times 128$ is processed by a single layer of tf.keras.layers.MaxPooling2D(pool_size=2, strides=2). What is the resulting output feature map shape?
A. $(128, 128)$
B. $(64, 64)$
C. $(127, 127)$
D. $(32, 32)$
Correct Answer: B
Type: Numerical
Difficulty: Medium
Unit: 1
CO: CO1
Q65. A single-channel image of size $20 \times 20$ is processed by a 3x3 kernel with no padding and a stride of 1. What is the output size?
A. $20 \times 20$
B. $17 \times 17$
C. $18 \times 18$
D. $19 \times 19$
Correct Answer: C
Type: Numerical
Difficulty: Medium
Unit: 1
CO: CO1
Q66. An attention mechanism computes an output tensor of shape $(B, N_Q, N_V)$. If the keys and values are the same size as the queries, and the number of hidden units $D_{hidden}=100$, what is the shape of the concatenation of the $H=4$ heads before the final projection?
A. $(B \cdot 4, N_Q, D_{hidden})$
B. $(B, N_Q, D_{hidden})$
C. $(B, N_Q, 4 \cdot D_{hidden})$
D. $(B, N_Q, D_{hidden} / 4)$
Correct Answer: B
Type: Numerical
Difficulty: Medium
Unit: 1
CO: CO1
Q67. Calculate the number of weights (kernel parameters) in a tf.keras.layers.Dense layer with 128 input features and 64 output units.
A. $128 \times 64$
B. $128 + 64$
C. $128 \times 64 + 64$
D. $128 \times 64 + 128$
Correct Answer: A
Type: Numerical
Difficulty: Medium
Unit: 1
CO: CO1
Q68. A model uses Softmax output $P(\text{Dog}) = 0.8$ and $P(\text{Cat}) = 0.2$. The true label is Cat. The cross-entropy loss is given by $L = -\log(P(\text{Cat}))$. Calculate the loss value (round to 3 decimal places).
A. 0.223
B. 1.609
C. 0.800
D. 0.999
Correct Answer: B
Type: Numerical
Difficulty: Medium
Unit: 1
CO: CO1
Q69. Batch Normalization (BN) layers behave differently during training and prediction. How does the parameter normalization statistic used in BN typically differ between these two modes?
A. Training uses fixed, initial statistics; prediction uses dynamically calculated minibatch statistics.
B. Training uses aggregated statistics from the entire dataset; prediction uses zero mean/unit variance.
C. Training uses minibatch statistics; prediction uses stable estimates calculated over the entire training dataset.
D. BN is only applied during training, not prediction.
Correct Answer: C
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q70. Which constraint, highly relevant to high-performance computing, is the primary reason for choosing minibatches (batch size $>1$) over single observation SGD (batch size $=1$)?
A. The constraint that GPUs are single-threaded devices.
B. The constraint that computation cannot be parallelized.
C. The inefficiency of reading and writing data to/from CPU/GPU caches for small, non-vectorized operations.
D. The necessity to use complex activation functions.
Correct Answer: C
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q71. What does the term "generalization gap" quantify in the context of model performance?
A. The difference between the highest and lowest validation error across epochs.
B. The difference between the model's performance on the training set and its expected performance on unseen data (test error).
C. The ratio of bias to variance.
D. The length of the computational graph.
Correct Answer: B
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q72. Dropout regularization achieves its goal by randomly setting a fraction of neurons' outputs to zero during training. What underlying problem does this mechanism aim to break among neurons?
A. Vanishing gradients across layers.
B. The complexity of the loss function.
C. Co-adaptation of hidden units in layers.
D. Exploding gradients in deeper networks.
Correct Answer: C
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q73. Why is L2 regularization applied during Mini-Batch SGD often referred to as "weight decay"?
A. Because the penalty term forces the weights to be positive.
B. Because the L2 penalty introduces an extra term that linearly shrinks the magnitude of the current weight vector towards zero in each update step.
C. Because it causes the learning rate to decrease exponentially.
D. Because it encourages sparse weights (setting many to zero).
Correct Answer: B
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q74. In adaptive optimizers like Adagrad and RMSProp, a small constant $\epsilon$ is often added in the denominator $\frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}}$. What is the primary purpose of this additive constant?
A. To increase the effective learning rate.
B. To ensure the optimizer achieves quadratic convergence.
C. To prevent division by zero or overly large step sizes if $\mathbf{s}_t$ is very small.
D. To implement bias correction for initial steps.
Correct Answer: C
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q75. Vectorization and parallelization are key to computational efficiency in deep learning. Which operation benefits the most from parallelization when performing Gradient Descent on a minibatch?
A. The scalar summation of the loss function results.
B. The element-wise computation of the activation function.
C. The large matrix-matrix multiplication ($\mathbf{X} \mathbf{W}$) and associated matrix-vector operations involved in the forward/backward pass.
D. The reading of hyperparameters from CPU memory.
Correct Answer: C
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q76. The Sigmoid activation function was popular initially but was largely superseded by ReLU. Why was the Sigmoid function problematic for training very deep networks?
A. Its computation was too slow for modern GPUs.
B. Its range was too narrow, restricting activation dynamics.
C. Its gradient vanishes (approaches zero) when inputs are large positive or large negative, hindering deep gradient flow.
D. It introduced bias by having a non-zero mean output.
Correct Answer: C
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q77. The core idea behind Gradient Descent is that the direction of steepest descent is given by the negative gradient. This idea is valid due to which mathematical concept?
A. The Law of Large Numbers.
B. The first-order Taylor expansion (linear approximation).
C. The Chain Rule of differentiation.
D. The Frobenius Norm calculation.
Correct Answer: B
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q78. When measuring model accuracy (proportion of correct predictions) during training, why is accuracy itself rarely used as the primary loss function to be minimized directly by the optimizer?
A. Accuracy is a continuous metric, unlike MSE.
B. Accuracy is typically non-differentiable or piecewise constant, preventing the smooth flow of gradients needed for backpropagation.
C. Accuracy requires too much memory to calculate on large minibatches.
D. Accuracy only works for regression problems.
Correct Answer: B
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q79. For optimization problems, what critical property of convex functions guarantees that the numerical solution obtained by Gradient Descent will be the absolute best solution available?
A. The function is always symmetric around the origin.
B. All local minima are also global minima.
C. The gradient is always one.
D. The domain is always a unit ball.
Correct Answer: B
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q80. Training RNNs on long sequences suffers from numerical instability. The problem is characterized by gradients that can become unstable due to the recurrent computation. What is the fundamental mathematical reason for this instability across long sequences?
A. The $T$ layers of matrix products along the time steps causing eigenvalues to vanish or diverge.
B. The use of the Softmax function in the output layer.
C. The non-linear conversion of data types (e.g., float to integer).
D. The L1 regularization of weights.
Correct Answer: A
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q81. If the learning rate $\eta$ is chosen to be excessively high in Gradient Descent, what is the most likely immediate consequence?
A. Slow convergence due to minimal updates.
B. The optimization process diverges or oscillates wildly.
C. Convergence to a high-bias solution.
D. Numerical underflow, stalling training.
Correct Answer: B
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q82. Which statement accurately captures the distinction between the goal of optimization in training and the goal of deep learning?
A. Optimization focuses on minimizing generalization error; Deep Learning focuses on minimizing training error.
B. Optimization focuses on finding the best model parameters given infinite data; Deep Learning focuses on model portability.
C. Optimization focuses on minimizing training error; Deep Learning focuses on minimizing generalization error.
D. Optimization uses only convex objectives; Deep Learning uses only non-convex objectives.
Correct Answer: C
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q83. Machine Learning theory often relies on the "i.i.d. assumption." What does this assumption state regarding the samples in the training and test sets?
A. They must be collected independently from the same underlying probability distribution.
B. They must be continuous and Gaussian distributed.
C. They must contain zero noise and be high dimensional.
D. They must be collected sequentially over time.
Correct Answer: A
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q84. In high-dimensional, non-convex loss landscapes, why are saddle points often considered more problematic than local minima?
A. Saddle points always lead to catastrophic divergence.
B. The large number of negative eigenvalues in the Hessian matrix makes saddle points far more common than local minima.
C. Saddle points inherently lead to vanishing gradients in all directions.
D. Saddle points only exist when L1 regularization is used.
Correct Answer: B
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q85. Adam computes normalized state variables $\hat{\mathbf{v}}_t$ and $\hat{\mathbf{s}}_t$ by dividing the raw moments $\mathbf{v}_t$ and $\mathbf{s}_t$ by factors like $(1 - \beta_1^t)$ and $(1 - \beta_2^t)$. What is the function of this operation?
A. To force the convergence of $\mathbf{v}_t$ and $\mathbf{s}_t$ to zero.
B. To reduce the computational cost of the square root operation.
C. To correct for the initial bias towards zero introduced by initializing the moments at $\mathbf{v}_0 = \mathbf{s}_0 = 0$.
D. To implement weight decay inherently within the optimizer.
Correct Answer: C
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q86. Why is the choice of batch size critical when applying Batch Normalization layers?
A. Batch Normalization only works if the batch size is a power of 2.
B. If the batch size is too small (e.g., 1), the resulting mean and variance estimates are high variance and useless, destroying the model's signal.
C. Batch Normalization introduces O($N^3$) complexity, which increases rapidly with batch size.
D. Small batch sizes automatically enforce L2 regularization.
Correct Answer: B
Type: Theory
Difficulty: Medium
Unit: 1
CO: CO1
Q87. A student attempts to build a small Keras model and runs the following code, resulting in an error during model construction because the input shape of the first Dense layer is unknown.
Which line must be added or modified to specify that the model expects inputs of shape (784,)?
A. model.add(tf.keras.Input(shape=(784,)))
B. model.add(tf.keras.layers.BatchNormalization())
C. model.add(tf.keras.layers.Dense(32, activation='relu', input_dim=784))
D. model.compile(..., input_shape=(784))
Correct Answer: A
Type: Code
Difficulty: Medium
Unit: 1
CO: CO1
Q88. An instructor decides to switch from using a batch size of 32 to a batch size of 128 (4x increase). Assuming the statistical properties hold, by what approximate factor will the statistical noise (standard deviation) of the gradient estimate be reduced?
A. Reduced by a factor of 4.
B. Reduced by a factor of 16.
C. Reduced by a factor of 2.
D. Increased by a factor of 4.
Correct Answer: C
Type: Numerical
Difficulty: Medium
Unit: 1
CO: CO1
Q89. A training sample $\mathbf{X}$ of shape (2, 512) is fed through a Keras layer where weights have already been inferred. What will be the shape of the output tensor $\mathbf{Y}$ if the layer is defined as:
A. (512, 64)
B. (2, 64)
C. (2, 512)
D. (64, 64)
Correct Answer: B
Type: Code
Difficulty: Medium
Unit: 1
CO: CO1
Q90. An MLP processes 784 inputs. It has a hidden layer of 256 units and an output layer of 10 units. Calculate the total number of biases in the entire network.
A. 784
B. 256
C. 266
D. 1040
Correct Answer: C
Type: Numerical
Difficulty: Medium
Unit: 1
CO: CO1
Q91. In classical learning theory (e.g., VC dimension), high model capacity (like millions of parameters) suggests a high risk of poor generalization. Why does the practice of training deep neural networks often contradict this, showing good generalization despite massive parameter counts?
A. Deep networks primarily rely on convex optimization techniques.
B. Effective regularization heuristics (Dropout, Early Stopping) and the optimization algorithm itself encode implicit inductive biases that limit the effective search space to "simpler" solutions that generalize well.
C. They exclusively use the Softmax activation function, which enforces stability.
D. The universal approximation theorem guarantees generalization regardless of dataset size.
Correct Answer: B
Type: Theory
Difficulty: Hard
Unit: 1
CO: CO1
Q92. For tasks like matrix factorization where sparsity (having many parameters set to zero) is desirable for interpretability and storage, which regularization technique is theoretically preferred?
A. L2 Regularization (Ridge Regression), as it gradually shrinks all weights.
B. L1 Regularization (Lasso Regression), as it tends to concentrate weights on a small set of features by pushing others exactly to zero.
C. Dropout, as it randomly sets weights to zero during inference.
D. Batch Normalization, as it normalizes parameter scales.
Correct Answer: B
Type: Theory
Difficulty: Hard
Unit: 1
CO: CO1
Q93. Consider a long chain of matrix multiplications $f(\mathbf{x}) = \mathbf{W}_3(\mathbf{W}_2(\mathbf{W}_1 \mathbf{x}))$. Why would a naive attempt to compute $\partial f / \partial \mathbf{W}_1$ by algebraically expanding the full expression and applying forward derivatives be catastrophically inefficient compared to backpropagation?
A. Forward derivation requires complex number arithmetic.
B. Backpropagation relies on the central limit theorem, which is faster.
C. Naive expansion leads to redundant computations of intermediate terms and matrix-matrix products, whereas backpropagation efficiently reuses derivatives from later stages to compute gradients for earlier stages.
D. Backpropagation converts the process to an equivalent $O(1)$ scalar operation.
Correct Answer: C
Type: Numerical
Difficulty: Hard
Unit: 1
CO: CO1
Q94. Which of the following equations correctly represents one step of the weight update rule for standard Mini-Batch Stochastic Gradient Descent (SGD) with learning rate $\eta$, where $\mathbf{g}$ is the calculated gradient and $\mathbf{w}t$ is the current weight vector?
A. $\mathbf{w}{t+1} \leftarrow \mathbf{w}t + \mathbf{g}$
B. $\mathbf{w}{t+1} \leftarrow \mathbf{w}t - \eta \cdot \mathbf{g}$
C. $\mathbf{w}{t+1} \leftarrow \eta \cdot \mathbf{w}t - \mathbf{g}$
D. $\mathbf{w}{t+1} \leftarrow \mathbf{w}_t - \mathbf{g} / \sqrt{\mathbf{w}_t}$
Correct Answer: B
Type: Code
Difficulty: Hard
Unit: 1
CO: CO1
Q95. Finding the optimal learning rate and regularization strength is computationally expensive and requires iterative training. Explain why it is impossible to correctly tune these hyperparameters based solely on the model's performance on the training dataset.
A. The computational cost of computing gradients becomes too high if training loss is used.
B. Tuning on training data always leads to maximizing the model's capacity, setting regularization to zero and forcing the learning rate to its maximum, resulting in severe overfitting to the training set.
C. The backpropagation algorithm is only valid if the training set is not used for HPO.
D. The model's performance metric (accuracy) is non-differentiable on the training set.
Correct Answer: B
Type: Theory
Difficulty: Hard
Unit: 1
CO: CO1
Q96. What is the fundamental requirement for a Multilayer Perceptron (MLP) with one or more hidden layers to be able to model complex, non-linear functions (as guaranteed by the universal approximation theorem), contrasting it with a simple affine model?
A. The output layer must use a Softmax activation.
B. The network must use a non-linear, differentiable activation function (like ReLU or Tanh) in the hidden layers.
C. The input must be one-hot encoded categorical data.
D. The hidden layer size must be greater than the input size.
Correct Answer: B
Type: Theory
Difficulty: Hard
Unit: 1
CO: CO1
Q97. An optimization problem is ill-conditioned (highly skewed loss landscape). Why does using the Momentum method (where $\beta \to 1$) over standard SGD help the optimization quickly move past the plateau along the shallow dimensions?
A. The large $\beta$ ensures that stochastic noise is maximized.
B. The accumulated velocity vector (v) aligns strongly with the consistent shallow gradient direction, accelerating progress, while oscillating gradients in steep directions cancel out.
C. Momentum forces the problem to become fully convex.
D. Momentum automatically calculates the inverse Hessian for preconditioning.
Correct Answer: B
Type: Numerical
Difficulty: Hard
Unit: 1
CO: CO1
Q98. Compare standard SGD ($\eta \mathbf{g}_t$) and SGD with Momentum ($\mathbf{v}t \leftarrow \beta \mathbf{v}{t-1} + \mathbf{g}t, \mathbf{w}{t+1} \leftarrow \mathbf{w}_t - \eta \mathbf{v}_t$). If both are applied to a non-convex, noisy problem, what is the primary statistical advantage of the momentum update?
A. Momentum guarantees convergence to the global optimum faster than SGD.
B. Momentum effectively integrates gradients over a larger history, reducing noise variance and stabilizing the direction of movement towards the optimum.
C. Momentum requires less memory because $\mathbf{v}_t$ is sparse.
D. Momentum prevents the learning rate from decaying.
Correct Answer: B
Type: Theory
Difficulty: Hard
Unit: 1
CO: CO1
Q99. The tf.linalg.matmul operation is called using two tensors, $\mathbf{A} \in \mathbb{R}^{2 \times 3}$ and $\mathbf{B} \in \mathbb{R}^{3 \times 4}$. The result $\mathbf{C}$ is a $2 \times 4$ matrix. If $\mathbf{A}$ and $\mathbf{B}$ are now 3D tensors representing a batch of 16 matrices, $\mathbf{A}' \in \mathbb{R}^{16 \times 2 \times 3}$ and $\mathbf{B}' \in \mathbb{R}^{16 \times 3 \times 4}$, what is the shape of the batch matrix multiplication output $\mathbf{C}' = \text{BMM}(\mathbf{A}', \mathbf{B}')$?
A. $\mathbb{R}^{16 \times 2 \times 4}$
B. $\mathbb{R}^{16 \times 3 \times 3}$
C. $\mathbb{R}^{16 \times 4 \times 3}$
D. $\mathbb{R}^{16 \times 3 \times 4}$
Correct Answer: A
Type: Numerical
Difficulty: Hard
Unit: 1
CO: CO1
Q100. The derivative of the Sigmoid activation function $\sigma(z)$ used during backpropagation is given by $\sigma'(z) = \sigma(z)(1-\sigma(z))$. If the layer output $o_j = \sigma(\text{net}_j)$, what is the analytic expression for the derivative $\partial o_j / \partial \text{net}_j$ in terms of the output $o_j$?
A. $\text{net}_j (1 - o_j)$
B. $o_j (1 - o_j)$
C. $1 - o_j^2$
D. $o_j^2 - 1$
Correct Answer: B
Type: Code
Difficulty: Hard
