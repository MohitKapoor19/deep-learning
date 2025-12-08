Unit: 4
CO: CO4
Q1. In a standard Recurrent Neural Network (RNN), the hidden state $h_t$ at time step $t$ is computed using the previous hidden state $h_{t-1}$ and the current input $x_t$. Which of the following equations represents this update (ignoring bias for simplicity)?
A. $h_t = \tanh(W_{hh}h_{t-1} + W_{hx}x_t)$
B. $h_t = \sigma(W_{hh}h_{t-1} \times W_{hx}x_t)$
C. $h_t = \tanh(W_{hh}x_t + W_{hx}h_{t-1})$
D. $h_t = \text{softmax}(W_{hh}h_{t-1} + W_{hx}x_t)$
Correct Answer: A
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q2. What is the primary reason simple RNNs struggle with learning long-term dependencies in long sequences?
A. The number of parameters grows exponentially with sequence length.
B. The Vanishing Gradient problem caused by repeated multiplication of weight matrices < 1 during backpropagation.
C. They cannot process inputs of variable lengths.
D. They require future information to compute the current state.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q3. Which technique is commonly used to mitigate the "Exploding Gradient" problem in RNNs?
A. Using ReLUs instead of Tanh.
B. Gradient Clipping.
C. Increasing the learning rate.
D. Using a deeper network.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q4. In the context of Long Short-Term Memory (LSTM) networks, what is the role of the **Forget Gate**?
A. To determine which parts of the new input to store in the cell state.
B. To output the hidden state based on the cell state.
C. To decide what information from the previous cell state should be discarded or kept.
D. To reset the hidden state to zero.
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q5. Which of the following gates is found in a Gated Recurrent Unit (GRU) but NOT in a standard LSTM?
A. Input Gate
B. Forget Gate
C. Update Gate
D. Output Gate
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 4
CO: CO4
Q6. Consider the following Keras code snippet using TensorFlow 2.x:
```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=1000, output_dim=64))
model.add(tf.keras.layers.SimpleRNN(128))
```
If the input batch has shape `(32, 10)`, what is the shape of the output from the `SimpleRNN` layer?
A. `(32, 10, 128)`
B. `(32, 128)`
C. `(32, 10, 64)`
D. `(128, 10)`
Correct Answer: B
Type: Code
Difficulty: Medium

Unit: 4
CO: CO4
Q7. What does the argument `return_sequences=True` do in a Keras Recurrent layer?
A. It returns the internal state (cell state) along with the output.
B. It returns the output at every time step in the input sequence, resulting in a 3D output tensor.
C. It reverses the input sequence before processing.
D. It returns only the last output of the sequence but duplicates it for the batch.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q8. In a GRU architecture, what happens if the **Reset Gate** ($r_t$) is close to 0?
A. The unit acts like a standard LSTM.
B. The previous hidden state is effectively ignored when computing the candidate hidden state.
C. The current input is ignored.
D. The previous hidden state is copied directly to the next state.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 4
CO: CO4
Q9. Calculate the number of trainable parameters in a `SimpleRNN` layer with `units=10` and an `input_dim=5` (input feature size). Assume use of bias.
Formula: $h \times h + h \times x + h$ (weights for hidden, weights for input, bias).
A. 150
B. 160
C. 65
D. 110
Correct Answer: B
Type: Numerical
Difficulty: Medium

Unit: 4
CO: CO4
Q10. Which activation function is primarily used for the **gates** (Input, Forget, Output) in an LSTM to output values between 0 and 1?
A. Tanh
B. ReLU
C. Sigmoid
D. Softmax
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q11. You are designing a model to forecast the temperature 24 hours into the future based on the past 24 hours of data. The data is sampled hourly. Which input shape would be appropriate for a Keras RNN layer processing this window?
A. `(batch_size, 24)`
B. `(batch_size, 1, 24)`
C. `(batch_size, 24, num_features)`
D. `(24, batch_size, num_features)`
Correct Answer: C
Type: Code
Difficulty: Easy

Unit: 4
CO: CO4
Q12. Refer to the Keras snippet below. What is the output shape of the `bidirectional` layer?
```python
layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))
# Input shape is (Batch, 10, 5)
```
A. `(Batch, 10, 32)`
B. `(Batch, 10, 64)`
C. `(Batch, 32)`
D. `(Batch, 64)`
Correct Answer: B
Type: Code
Difficulty: Medium

Unit: 4
CO: CO4
Q13. Why might one choose a GRU over an LSTM?
A. GRUs are strictly more powerful than LSTMs for all tasks.
B. GRUs have fewer parameters and are computationally faster to train while often achieving comparable performance.
C. GRUs do not suffer from the vanishing gradient problem, whereas LSTMs do.
D. GRUs can handle longer sequences than LSTMs because they have more gates.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q14. In the LSTM cell state equation $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$, what does the symbol $\odot$ represent?
A. Matrix multiplication
B. Element-wise addition
C. Element-wise multiplication (Hadamard product)
D. Dot product
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q15. Consider a Keras model for text classification.
```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
If the input is a batch of sentences padded to length 50, how many parameters does the Embedding layer have?
A. 320,000
B. 10,000
C. 1,600
D. 32
Correct Answer: A
Type: Numerical
Difficulty: Easy

Unit: 4
CO: CO4
Q16. What is "Backpropagation Through Time" (BPTT)?
A. A technique to train RNNs by unrolling the network over the time steps and applying standard backpropagation.
B. A method to speed up RNN training by skipping time steps.
C. A forward-only algorithm for making predictions in time series.
D. An initialization strategy for recurrent weights.
Correct Answer: A
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q17. Which component of the LSTM architecture is explicitly designed to handle the **Vanishing Gradient** problem by creating a "gradient superhighway"?
A. The Hidden State $h_t$
B. The Cell State $C_t$
C. The Output Gate
D. The Tanh activation
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 4
CO: CO4
Q18. You are training a character-level RNN for text generation. The vocabulary size is 50. If you use a `SimpleRNN` layer with 100 units, and the input is one-hot encoded, what is the size of the input weight matrix $W_{hx}$?
A. $100 \times 100$
B. $50 \times 100$
C. $100 \times 50$
D. $50 \times 50$
Correct Answer: B
Type: Numerical
Difficulty: Medium

Unit: 4
CO: CO4
Q19. In TensorFlow/Keras, what argument in `LSTM` layer allows you to retrieve the final cell state $C_t$ alongside the output?
A. `return_sequences=True`
B. `return_state=True`
C. `stateful=True`
D. `output_state=True`
Correct Answer: B
Type: Code
Difficulty: Medium

Unit: 4
CO: CO4
Q20. When using `GridSearchCV` with a Keras model wrapper (e.g., `KerasClassifier` or `KerasRegressor`), why is `n_jobs=-1` sometimes problematic on Windows or with GPU?
A. It causes the GPU to overheat.
B. It can cause serialization/pickling errors or conflicts with the backend's internal parallelism.
C. It forces the model to use CPU only.
D. It disables cross-validation.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 4
CO: CO4
Q21. Which statement best describes the "Update Gate" $z_t$ in a GRU?
A. It determines how much of the past information to forget and how much new information to add.
B. It completely resets the memory content.
C. It controls the output visibility of the hidden state.
D. It only filters the input data.
Correct Answer: A
Type: Theory
Difficulty: Medium

Unit: 4
CO: CO4
Q22. Calculate the number of parameters in a standard LSTM layer with `units=10` and `input_features=5`. Use the formula: $4 \times (h \times h + x \times h + h)$.
A. 640
B. 400
C. 600
D. 240
Correct Answer: A
Type: Numerical
Difficulty: Hard

Unit: 4
CO: CO4
Q23. Which of the following problems is an LSTM specifically designed to solve compared to a basic RNN?
A. High computational cost per step.
B. Difficulty in mapping fixed-size inputs to fixed-size outputs.
C. Inability to capture long-term dependencies due to vanishing gradients.
D. Overfitting on small datasets.
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q24. In the context of time series forecasting with Keras, what is the purpose of the `tf.keras.utils.timeseries_dataset_from_array` function?
A. To plot the time series data.
B. To efficiently generate batches of sliding windows of inputs and targets.
C. To normalize the time series data.
D. To convert a time series into a static image.
Correct Answer: B
Type: Code
Difficulty: Medium

Unit: 4
CO: CO4
Q25. Identify the error in the following Keras code for a stacked LSTM:
```python
model = Sequential()
model.add(LSTM(64, input_shape=(10, 5))) # Layer 1
model.add(LSTM(32))                      # Layer 2
```
A. Layer 1 is missing `return_sequences=True`.
B. Layer 2 units must match Layer 1 units.
C. `input_shape` should be `(5, 10)`.
D. LSTM layers cannot be stacked.
Correct Answer: A
Type: Code
Difficulty: Easy

Unit: 4
CO: CO4
Q26. What is the range of the output of the `tanh` activation function used in RNNs and LSTMs?
A. $$
B. $[-1, 1]$
C. $[0, \infty)$
D. $(-\infty, \infty)$
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q27. In a "many-to-one" sequence classification task (e.g., sentiment analysis), which output from the RNN layer is typically used?
A. The output at every time step.
B. The output from the last time step.
C. The average of all outputs.
D. The initial state.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q28. A GRU layer has 10 units and input dimension 5. How does its parameter count compare to an LSTM with the same configuration?
A. GRU has roughly 75% of the parameters of the LSTM.
B. GRU has more parameters than the LSTM.
C. They have exactly the same number of parameters.
D. GRU has half the parameters of the LSTM.
Correct Answer: A
Type: Numerical
Difficulty: Medium

Unit: 4
CO: CO4
Q29. Which technique allows an RNN to process inputs of variable lengths in a batch by ignoring specific padding values?
A. Dropout
B. Batch Normalization
C. Masking
D. Pooling
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q30. Consider a time series prediction model where the model predicts the next value based on the previous 10 values. If the input data has shape `(1000, 1)`, and we create windows of length 10, what is the shape of a single input sample fed into the model?
A. `(1, 10)`
B. `(10, 1)`
C. `(1000, 1)`
D. `(1, 1)`
Correct Answer: B
Type: Numerical
Difficulty: Medium

Unit: 4
CO: CO4
Q31. In the equation $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ for an LSTM, what does $[h_{t-1}, x_t]$ represent?
A. The subtraction of the previous hidden state and current input.
B. The dot product of the previous hidden state and current input.
C. The concatenation of the previous hidden state and the current input.
D. The element-wise multiplication of the previous hidden state and current input.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 4
CO: CO4
Q32. When performing hyperparameter tuning on an RNN using `GridSearchCV`, why might one fix the random seed?
A. To make the grid search run faster.
B. To ensure the results are reproducible despite the stochastic nature of network initialization.
C. To prevent the gradients from exploding.
D. To allow the model to use the GPU.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q33. What is the output shape of the following layer?
```python
tf.keras.layers.GRU(32, return_sequences=True, input_shape=(20, 5))
```
A. `(None, 20, 32)`
B. `(None, 32)`
C. `(None, 20, 5)`
D. `(None, 5, 32)`
Correct Answer: A
Type: Code
Difficulty: Easy

Unit: 4
CO: CO4
Q34. In a "stateful" RNN in Keras (`stateful=True`), what happens to the internal states between batches?
A. They are reset to zero.
B. They are re-initialized randomly.
C. They are preserved and passed as the initial state for the next batch.
D. They are averaged across the batch.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 4
CO: CO4
Q35. Which of the following is an advantage of Transformers over RNNs/LSTMs for sequence processing?
A. Transformers process data sequentially, making them slower.
B. Transformers allow for parallelization of the entire sequence processing.
C. Transformers have fewer parameters than RNNs for the same hidden size.
D. Transformers require no memory for long sequences.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q36. For an LSTM layer with 10 units, what is the dimensionality of the Cell State $C_t$?
A. 1
B. 10
C. 20 (concatenated with hidden state)
D. It depends on the input size.
Correct Answer: B
Type: Numerical
Difficulty: Easy

Unit: 4
CO: CO4
Q37. What does the `tf.keras.layers.Bidirectional` wrapper do?
A. It trains two RNNs: one on the input sequence and one on the reversed input sequence, and merges their outputs.
B. It allows the RNN to output values greater than 1.
C. It connects the output of the RNN back to its input.
D. It doubles the number of time steps in the input.
Correct Answer: A
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q38. In the paper "Comparing LSTM and GRU Models to Predict the Condition of a Pulp Paper Press" (Source), which model generally showed better performance and stability for the given dataset?
A. The standard RNN.
B. The LSTM.
C. The GRU.
D. Both performed exactly the same.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 4
CO: CO4
Q39. A "Peephole" LSTM connection allows the gates to look at:
A. The next input in the sequence.
B. The cell state $C_{t-1}$.
C. The output of the next layer.
D. The gradients of the future steps.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 4
CO: CO4
Q40. Code Snippet:
```python
inputs = tf.keras.Input(shape=(None, 10))
x = tf.keras.layers.LSTM(20)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
```
Is this model architecture suitable for "Many-to-Many" (Sequence output) or "Many-to-One" (Single output) tasks?
A. Many-to-Many
B. Many-to-One
C. One-to-Many
D. One-to-One
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 4
CO: CO4
Q41. Which gradient problem causes the weights of an RNN to update very slowly or not at all, preventing the learning of long-term dependencies?
A. Exploding Gradient
B. Vanishing Gradient
C. Oscillating Gradient
D. Sparse Gradient
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q42. In Keras, how do you specify that an LSTM layer should process input in reverse order?
A. `reverse=True`
B. `go_backwards=True`
C. `direction='backward'`
D. `input_reverse=True`
Correct Answer: B
Type: Code
Difficulty: Medium

Unit: 4
CO: CO4
Q43. Given an input sequence of length $T$ and feature dimension $D$, and a SimpleRNN with $H$ units. What is the time complexity of the forward pass for one sequence?
A. $O(T \times D \times H)$
B. $O(T \times (H^2 + H \times D))$
C. $O(H^2)$
D. $O(T^2 \times H)$
Correct Answer: B
Type: Numerical
Difficulty: Hard

Unit: 4
CO: CO4
Q44. What does the `Dropout` argument in a Keras RNN layer constructor (e.g., `LSTM(64, dropout=0.2)`) apply to?
A. The recurrent connections (state-to-state).
B. The linear transformation of the inputs (input-to-state).
C. The output layer only.
D. The bias terms.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 4
CO: CO4
Q45. If you are tuning hyperparameters for a Keras LSTM model using `GridSearchCV`, which wrapper class from `scikeras.wrappers` (or `tensorflow.keras.wrappers.scikit_learn`) should be used for a regression problem?
A. `KerasClassifier`
B. `KerasRegressor`
C. `KerasGrid`
D. `KerasOptimizer`
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 4
CO: CO4
Q46. The operation $f_t \odot C_{t-1}$ in an LSTM corresponds to which logical operation?
A. Writing new information.
B. Forgetting old information.
C. Outputting the hidden state.
D. Creating the candidate cell state.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q47. When forecasting time series, if we use a "Single-shot" prediction models to predict 24 hours into the future at once, the output layer typically has how many units (assuming 1 feature)?
A. 1
B. 24
C. 12
D. It depends on the batch size.
Correct Answer: B
Type: Numerical
Difficulty: Medium

Unit: 4
CO: CO4
Q48. Which RNN architecture reduces the number of gates compared to LSTM by combining the Forget and Input gates into an Update gate?
A. Bidirectional RNN
B. GRU
C. Peephole LSTM
D. Deep RNN
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q49. Consider the following code for a Conv1D layer used in time series:
```python
tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')
```
If the input sequence length is 10, what is the length of the output sequence (assuming default 'valid' padding)?
A. 10
B. 8
C. 7
D. 12
Correct Answer: B
Type: Numerical
Difficulty: Medium

Unit: 4
CO: CO4
Q50. Why is `n_jobs=-1` in `GridSearchCV` sometimes avoided when training Keras models on a single GPU?
A. It causes the CPU to overheat.
B. Keras/TensorFlow already utilizes the GPU parallelism, and forking processes can cause memory contention or serialization errors.
C. It stops the grid search from running.
D. It reverses the order of the search.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 4
CO: CO4
Q51. What is the value range of the candidate hidden state $\tilde{h}_t$ in a GRU (typically using `tanh`)?
A. $$
B. $[-1, 1]$
C. $[0, \infty)$
D. $(-\infty, \infty)$
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q52. In Keras, what argument is used to specify the initial state of an RNN layer during the `call`?
A. `start_state`
B. `initial_state`
C. `reset_state`
D. `begin_state`
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 4
CO: CO4
Q53. Which loss function is most appropriate for a binary text classification problem using an RNN?
A. `mean_squared_error`
B. `categorical_crossentropy`
C. `binary_crossentropy`
D. `sparse_categorical_crossentropy`
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q54. If an LSTM layer has 100 units, how many "cell states" are maintained internally during processing?
A. 1
B. 100
C. 200 (100 hidden + 100 cell)
D. 400 (one for each gate)
Correct Answer: B
Type: Numerical
Difficulty: Medium

Unit: 4
CO: CO4
Q55. In the context of NLP, what is the purpose of an `Embedding` layer placed before an LSTM?
A. To convert integer token indices into dense vectors of fixed size.
B. To convert text to one-hot encoding.
C. To reduce the sequence length.
D. To normalize the input text.
Correct Answer: A
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q56. Which of the following is NOT a gate in the standard GRU?
A. Update Gate
B. Reset Gate
C. Output Gate
D. All are present in GRU.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 4
CO: CO4
Q57. Calculate the number of parameters for a GRU layer with `units=32` and `input_dim=10` (ignore bias for simplicity).
Formula: $3 \times (h \times h + h \times x)$
A. $3 \times (32 \times 32 + 32 \times 10) = 4032$
B. $3 \times (32 + 10) = 126$
C. $4 \times (32 \times 32 + 32 \times 10) = 5376$
D. $32 \times 32 + 32 \times 10 = 1344$
Correct Answer: A
Type: Numerical
Difficulty: Hard

Unit: 4
CO: CO4
Q58. What is the effect of `recurrent_dropout` in Keras RNN layers?
A. It drops inputs connecting to the layer.
B. It drops connections between the previous hidden state and the current hidden state.
C. It drops the final output of the layer.
D. It drops entire time steps from the sequence.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 4
CO: CO4
Q59. In an LSTM, if the **Input Gate** is 0 and the **Forget Gate** is 1, what happens to the cell state?
A. The cell state is reset to 0.
B. The cell state remains exactly the same as the previous time step ($C_t = C_{t-1}$).
C. The cell state is updated with new information only.
D. The cell state becomes 1.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 4
CO: CO4
Q60. Which Keras callback is commonly used to save the best model during training based on validation loss?
A. `ModelCheckpoint`
B. `EarlyStopping`
C. `ReduceLROnPlateau`
D. `TensorBoard`
Correct Answer: A
Type: Code
Difficulty: Easy

Unit: 4
CO: CO4
Q61. What is "Teacher Forcing" in the context of training sequence-to-sequence RNNs?
A. Using the model's own output from the previous step as input for the current step during training.
B. Using the actual ground truth output from the previous step as input for the current step during training.
C. Freezing the weights of the encoder.
D. Increasing the learning rate over time.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 4
CO: CO4
Q62. A stacked RNN consists of 3 LSTM layers. Which layers MUST have `return_sequences=True`?
A. Only the last layer.
B. The first and second layers.
C. Only the first layer.
D. None of the layers.
Correct Answer: B
Type: Code
Difficulty: Medium

Unit: 4
CO: CO4
Q63. What is the shape of the kernel weights matrix ($W_{hx}$) in a SimpleRNN layer with 50 units and input dimension 20?
A. `(50, 50)`
B. `(20, 50)`
C. `(50, 20)`
D. `(20, 20)`
Correct Answer: B
Type: Numerical
Difficulty: Medium

Unit: 4
CO: CO4
Q64. When using `tf.data.Dataset` for time series, what does the `window()` method do?
A. It normalizes the data.
B. It groups elements into windows (sequences) of a specified size.
C. It shuffles the dataset.
D. It splits the data into train and test sets.
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 4
CO: CO4
Q65. Which limitation of RNNs is addressed by the bidirectional RNN?
A. Vanishing gradients.
B. The inability to use future context in the sequence to predict the current output.
C. High memory usage.
D. Slow training speed.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q66. If `x` has shape `(32, 10, 8)` (Batch, Time, Feat), what is the output shape of:
`tf.keras.layers.GlobalMaxPooling1D()(x)`?
A. `(32, 10)`
B. `(32, 8)`
C. `(32, 1)`
D. `(32, 10, 1)`
Correct Answer: B
Type: Code
Difficulty: Medium

Unit: 4
CO: CO4
Q67. In the context of LSTM, what is $\tilde{C}_t$ (Candidate Cell State)?
A. The final output of the cell.
B. A vector of new candidate values that could be added to the state, created by a tanh layer.
C. The value of the forget gate.
D. The value of the output gate.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q68. In time series forecasting, what does a "Naive Forecast" or "Persistence Model" typically predict?
A. The average of all past values.
B. The value at the last observed time step ($y_{t+1} = y_t$).
C. A random value.
D. Zero.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q69. Which of the following is true about the parameter count of LSTM vs SimpleRNN?
A. LSTM has roughly 4 times the parameters of a SimpleRNN with the same number of units.
B. LSTM has fewer parameters than SimpleRNN.
C. They have the same number of parameters.
D. LSTM has 2 times the parameters.
Correct Answer: A
Type: Numerical
Difficulty: Easy

Unit: 4
CO: CO4
Q70. Code Snippet:
```python
rnn = tf.keras.layers.SimpleRNN(10, return_state=True)
output, state = rnn(inputs)
```
If `inputs` has shape `(Batch, Time, Feat)`, what is the relationship between `output` and `state`?
A. `output` is the full sequence, `state` is the last step.
B. `output` and `state` are identical.
C. `output` is the last step hidden state, `state` is the full sequence.
D. `state` is the cell state, which SimpleRNN does not have.
Correct Answer: B
Type: Code
Difficulty: Hard

Unit: 4
CO: CO4
Q71. Why is the sigmoid function used for gates in LSTM/GRU?
A. It outputs values in $(-1, 1)$, ideal for data scaling.
B. It outputs values in $(0, 1)$, ideal for acting as a "switch" or percentage to let information through.
C. It avoids the vanishing gradient problem better than ReLU.
D. It is computationally cheaper than tanh.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q72. When dealing with variable-length sequences in Keras (e.g., text), what is the standard value used for padding?
A. -1
B. 0
C. NaN
D. 1
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q73. What is the formula for the Reset Gate $r_t$ in a GRU?
A. $r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$
B. $r_t = \tanh(W_r \cdot [h_{t-1}, x_t])$
C. $r_t = \sigma(W_r \cdot h_{t-1} + U_r \cdot x_t)$
D. $r_t = \text{ReLU}(W_r \cdot x_t)$
Correct Answer: A
Type: Theory
Difficulty: Medium

Unit: 4
CO: CO4
Q74. For a "Many-to-Many" task where the input and output sequences have the same length (e.g., POS tagging), which `return_sequences` setting is required for the output RNN layer?
A. `return_sequences=False`
B. `return_sequences=True`
C. `return_state=True`
D. It doesn't matter.
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 4
CO: CO4
Q75. How many bias terms are typically associated with a single LSTM unit (considering all 4 gates/components)?
A. 1
B. 2
C. 3
D. 4
Correct Answer: D
Type: Numerical
Difficulty: Medium

Unit: 4
CO: CO4
Q76. Which hyperparameter is critical for the stability of training RNNs to prevent exploding gradients?
A. Batch size
B. Clipnorm or Clipvalue
C. Dropout
D. L2 Regularization
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q77. In the TensorFlow "Time series forecasting" tutorial, what does the `WindowGenerator.split_window` method do?
A. Splits the data into training and testing sets.
B. Splits a window of consecutive samples into inputs and labels.
C. Splits the time series into hourly chunks.
D. Normalizes the window data.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 4
CO: CO4
Q78. In an autoregressive RNN model for text generation, the prediction at time $t$ is used as:
A. The label for time $t$.
B. The input for time $t+1$.
C. The hidden state for time $t-1$.
D. The reset gate value.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q79. Which statement is TRUE regarding the difference between `tf.keras.layers.RNN` and `tf.keras.layers.LSTM`?
A. `RNN` is a specific layer, `LSTM` is a base class.
B. `LSTM` is a built-in layer optimized for CuDNN, while `RNN` is a base class that takes a cell instance (e.g., `LSTMCell`).
C. There is no difference.
D. `RNN` can only implement SimpleRNN logic.
Correct Answer: B
Type: Code
Difficulty: Hard

Unit: 4
CO: CO4
Q80. Calculate the output shape of:
```python
x = tf.keras.Input((100, 16))
y = tf.keras.layers.Conv1D(filters=32, kernel_size=5, padding='same')(x)
```
A. `(None, 100, 32)`
B. `(None, 96, 32)`
C. `(None, 100, 16)`
D. `(None, 32, 100)`
Correct Answer: A
Type: Code
Difficulty: Medium

Unit: 4
CO: CO4
Q81. What does the "Cell State" in an LSTM represent intuitively?
A. Short-term working memory.
B. The "conveyor belt" or long-term memory that runs through the entire chain with minor linear interactions.
C. The output of the current time step.
D. The forget gate activation.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q82. In a GRU, the equation $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ represents:
A. The calculation of the reset gate.
B. The calculation of the candidate hidden state.
C. The final update of the hidden state, combining old state and new candidate.
D. The output gate.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 4
CO: CO4
Q83. Which optimizer is generally recommended as a first choice for training RNNs/LSTMs due to its adaptive learning rate capabilities?
A. SGD
B. Adam
C. Adadelta
D. Momentum
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q84. What is the purpose of the `Masking` layer in Keras?
A. To hide neurons during training (Dropout).
B. To tell downstream layers to skip processing of timesteps containing a specific padding value.
C. To normalize the input data.
D. To obscure the labels.
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 4
CO: CO4
Q85. If you have a sequence classification problem with 5 classes, what should be the activation function of the final Dense layer?
A. Sigmoid
B. Tanh
C. ReLU
D. Softmax
Correct Answer: D
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q86. In the context of the `WindowGenerator` class (Source), if `input_width=24`, `label_width=1`, and `shift=24`, does the label overlap with the input?
A. Yes, they overlap completely.
B. No, the label immediately follows the input window (non-overlapping, immediately adjacent).
C. Yes, by 1 step.
D. No, there is a gap between them.
Correct Answer: B
Type: Code
Difficulty: Medium

Unit: 4
CO: CO4
Q87. What happens if you try to use `GridSearchCV` with a Keras model that has multiple inputs?
A. It works automatically.
B. It fails because scikit-learn wrappers typically expect a single `X` array; you may need to wrap inputs or use manual loops.
C. It requires the `n_jobs` parameter to be set to 0.
D. It only works with `RandomizedSearchCV`.
Correct Answer: B
Type: Theory
Difficulty: Hard

Unit: 4
CO: CO4
Q88. Which line of code correctly compiles a Keras RNN model for a regression task?
A. `model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])`
B. `model.compile(optimizer='adam', loss='categorical_crossentropy')`
C. `model.compile(optimizer='adam', loss='mse', metrics=['mae'])`
D. `model.compile(optimizer='sgd', loss='binary_crossentropy')`
Correct Answer: C
Type: Code
Difficulty: Medium

Unit: 4
CO: CO4
Q89. In a text classification model using embeddings, what does `input_length` in the Embedding layer specify?
A. The size of the vocabulary.
B. The size of the embedding vector.
C. The length of the input sequences (number of tokens).
D. The batch size.
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q90. Which of the following is a key advantage of using `tf.data.Dataset` pipelines for RNN training?
A. It automatically tunes hyperparameters.
B. It allows for efficient prefetching, batching, and shuffling of large sequence datasets that don't fit in memory.
C. It guarantees convergence.
D. It removes the need for padding.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q91. Given `SimpleRNN(units=64)`, what is the dimension of the hidden state vector?
A. 32
B. 64
C. 128
D. Variable
Correct Answer: B
Type: Numerical
Difficulty: Easy

Unit: 4
CO: CO4
Q92. What is the primary function of the `tanh` activation in the candidate cell state $\tilde{C}_t$ calculation of an LSTM?
A. To gate the information (0 to 1).
B. To regulate the network by outputting values in $[-1, 1]$, preventing values from growing indefinitely.
C. To force values to be positive.
D. To output probabilities.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 4
CO: CO4
Q93. Code Snippet:
```python
encoder_input = Input(shape=(None, 10))
encoder = LSTM(64, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_input)
```
What are `state_h` and `state_c`?
A. The output sequence and the last output.
B. The hidden state and the cell state of the last time step.
C. The hidden state of the first and last time step.
D. The forward and backward states.
Correct Answer: B
Type: Code
Difficulty: Medium

Unit: 4
CO: CO4
Q94. If `return_sequences=True` is used in an LSTM layer connected to a `Dense` layer for per-step classification, what wrapper is often used on the Dense layer in older Keras versions (though not strictly necessary in TF2)?
A. `TimeDistributed`
B. `Bidirectional`
C. `Lambda`
D. `Flatten`
Correct Answer: A
Type: Theory
Difficulty: Medium

Unit: 4
CO: CO4
Q95. Why are RNNs generally slower to train than CNNs or Transformers?
A. They have more parameters.
B. Their sequential nature prevents parallelization across time steps.
C. They require more memory.
D. They use sigmoid functions.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q96. For a time series forecasting task using an LSTM, if you want to predict the next 5 steps based on the previous 10 steps using a "Single-shot" approach, what is the output size of the final Dense layer (assuming 1 feature)?
A. 1
B. 5
C. 10
D. 50
Correct Answer: B
Type: Numerical
Difficulty: Medium

Unit: 4
CO: CO4
Q97. In the context of the paper "Comparing LSTM and GRU Models", what did the authors find regarding the **training time** of GRUs vs LSTMs?
A. GRUs were slower to train.
B. GRUs were faster to train due to fewer parameters.
C. There was no difference.
D. LSTM was faster because of CuDNN optimization.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 4
CO: CO4
Q98. What is the shape of the bias vector in a `SimpleRNN` layer with `units=U`?
A. `(U,)`
B. `(2*U,)`
C. `(input_dim,)`
D. `(U, U)`
Correct Answer: A
Type: Numerical
Difficulty: Easy

Unit: 4
CO: CO4
Q99. Which gate in the LSTM is responsible for solving the **Vanishing Gradient** problem by allowing the gradient to pass through unchanged (if the gate is open)?
A. The input gate via the additive update.
B. The forget gate (when set to 1) preserving the cell state.
C. The output gate.
D. The candidate gate.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 4
CO: CO4
Q100. Consider the following code for a character-level text generation model:
```python
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))
```
If `len(chars)` (vocab size) is 50, what does the model output?
A. A single character index.
B. A probability distribution over the 50 characters for the next time step.
C. A sequence of 50 characters.
D. A scalar value representing the likelihood of the sequence.
Correct Answer: B
Type: Code
Difficulty: Easy