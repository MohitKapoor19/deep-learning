Unit: 3
CO: CO3
Q1. Which statement best describes the primary advantage of using a Long Short-Term Memory (LSTM) network over an elementary Recurrent Neural Network (RNN) for multi-class text classification tasks?
A. LSTMs are simpler and faster to train on short texts.
B. LSTMs exclusively use convolutional filters to capture word relationships.
C. LSTMs utilize specialized "gates" to mitigate the vanishing gradient problem, allowing them to retain context over very long sequences.
D. LSTMs process sequential data (text) by treating the order of words as irrelevant.
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q2. What is the fundamental step in the Text Preprocessing phase where raw text is broken down into smaller manageable units like words or punctuation symbols?
A. Stemming.
B. Lemmatization.
C. Tokenization.
D. Vectorization.
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q3. In the typical NLP pipeline for building a sentiment classification model, what immediate output is produced after the "Model" phase?
A. The raw corpus text (string).
B. A preprocessed list of stopwords.
C. A set of weights representing word embeddings.
D. A prediction (e.g., positive or negative class label).
Correct Answer: D
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q4. Consider a simple scenario for Term Frequency (TF) calculation where Document A contains: "The cat sat on the cat." Using the raw count method for $f_{t,d}$, what is the TF value for the term "cat" in Document A?
A. 1.
B. 2.
C. 5.
D. 0.4.
Correct Answer: B
Type: Numerical
Difficulty: Easy

Unit: 3
CO: CO3
Q5. Which issue, common in basic RNNs, is primarily mitigated by the enhanced memory capabilities (gates) of Long Short-Term Memory (LSTM) networks, particularly when processing long texts?
A. Overfitting to training data.
B. Exploding activation functions.
C. Vanishing gradients, leading to "forgetting" earlier words.
D. Arbitrary integer encoding.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q6. Which text preprocessing step involves converting characters to a uniform case (e.g., lowercase) and removing special characters or numbers from the tokens?
A. Lemmatization.
B. Standardization and Cleansing.
C. N-gram generation.
D. Inverse Document Frequency calculation.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q7. A common real-world application of NLP in the insurance industry involves analyzing claims for patterns. What is the primary benefit this provides?
A. Detecting financial market risks.
B. Automating information extraction and detecting fraud indicators in claims management.
C. Virtual therapy support.
D. Generating marketing copy.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q8. In the formula for Term Frequency-Inverse Document Frequency (TF-IDF), $tfidf(t, d, D) = tf(t, d) \cdot idf(t, D)$, what is the primary purpose of the $idf(t, D)$ component?
A. To measure the raw count of term $t$ in document $d$.
B. To ensure words are assigned unique integer encodings.
C. To measure how common or rare term $t$ is across the entire corpus $D$.
D. To convert the term $t$ to its root form.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q9. The process of converting words like "running," "ran," and "runs" to the base form "run" is called:
A. Tokenization.
B. Stop word removal.
C. Stemming or Lemmatization.
D. Feature Extraction.
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q10. A model is defined using the following Keras layers. If the `vocab_size` is 10,000, `sequence_length` is 100, and `embedding_dim` is 16, what is the exact output shape of the `Embedding` layer?

```python
model = Sequential([
    Embedding(vocab_size, embedding_dim),
    GlobalAveragePooling1D(),
    Dense(1)
])
```
A. (None, 16).
B. (None, 100).
C. (None, 100, 16).
D. (None, 100, 10000).
Correct Answer: C
Type: Code
Difficulty: Medium

Unit: 3
CO: CO3
Q11. What conceptual issue does the TF-IDF measure attempt to solve regarding term importance in a document?
A. High-frequency terms (like "the") should be filtered out entirely (like stop words).
B. The importance of a word should be proportional to its frequency only within a single document.
C. Terms that appear frequently in specific documents but are rare across the entire corpus should have a high weight.
D. Term importance must always be represented by a dense, floating-point vector.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q12. In the KerasNLP architecture for a BERT Classifier, what primary role does the `BertPreprocessor` serve?
A. It performs simple lowercasing and punctuation removal.
B. It loads the ready-to-use pre-trained classification weights.
C. It maps input strings to a dictionary of tensors by performing tokenization and additional preprocessing like padding.
D. It calculates the classification loss and accuracy.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q13. The text preprocessing technique that reduces words to a valid dictionary form, often incorporating the word's part-of-speech, is called:
A. Stemming.
B. Lemmatization.
C. Tokenization.
D. Stop word removal.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q14. The Keras `TextVectorization` layer is utilized in the NLP pipeline for classification models. What essential function does this layer perform on the input text?
A. It calculates the TF-IDF weight for every word.
B. It transforms raw input strings into numerical integer indices corresponding to the vocabulary.
C. It applies backpropagation to update the embedding weights.
D. It ensures all input sequences have variable lengths.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q15. Why is the "one-hot encoding" approach inefficient for text representation, especially when dealing with large vocabularies?
A. It fails to capture semantic relationships between words.
B. The resulting vectors are dense, consuming excessive memory.
C. The integer-encoding is arbitrary and hard to interpret.
D. The resulting vectors are extremely sparse (mostly zeros) and high-dimensional.
Correct Answer: D
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q16. If a corpus $D$ consists of two documents, $D_1$ and $D_2$, and the term "this" appears in both documents ($n_t=2$). Using the standard $idf(t, D) = \log(N/n_t)$ formula (where $N=2$), what is the $idf$ value for the term "this"?
A. $\log(2)$.
B. $\log(1/2)$.
C. $0$.
D. $1$.
Correct Answer: C
Type: Numerical
Difficulty: Medium

Unit: 3
CO: CO3
Q17. A Word Embedding is fundamentally a representation of a word, typically a real-valued vector, designed so that:
A. Words are mapped to vectors based purely on their alphabetical order.
B. Words closer in the vector space are expected to be similar in meaning.
C. The vector space dimensionality must equal the vocabulary size.
D. The vectors are manually specified before model training.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q18. Which statement accurately captures the distinction between stemming and lemmatization?
A. Stemming uses part-of-speech tagging; lemmatization does not.
B. Lemmatization focuses on structural reduction; stemming focuses on dictionary roots.
C. Stemming is faster but can produce non-dictionary root forms; Lemmatization is slower but ensures valid word roots by considering context.
D. Both methods produce identical results, but lemmatization uses a larger stop list.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q19. In morphology and syntax, what term is used to describe a category (like Noun, Verb, Adjective) that acquires new lexical items frequently?
A. Closed class.
B. Open class.
C. Fixed class.
D. Valency class.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q20. When defining a Keras sequential model for binary text classification (e.g., positive/negative sentiment), which combination of the final Dense layer output shape and activation function is appropriate?
A. Output shape (None, 2) with activation 'softmax'.
B. Output shape (None, 1) with activation 'relu'.
C. Output shape (None, 1) with activation 'sigmoid' (or no activation, using `from_logits=True` loss).
D. Output shape (None, vocab_size) with activation 'tanh'.
Correct Answer: C
Type: Code
Difficulty: Easy

Unit: 3
CO: CO3
Q21. The "SocialGuard" study on Bangla gender identification found that traditional Machine Learning (ML) models often outperformed Deep Learning (DL) models when using TF-IDF features. What reason was attributed to DL models' relative underperformance in this specific context?
A. DL models over-rely on syntactic features which are absent in Bangla text.
B. DL models require smaller datasets, which introduces bias.
C. DL models rely on weight-based categorization which exhibits underperformance when compared to the efficiency of SGD's linear decision boundaries.
D. Traditional models were able to use much higher dimensional word embeddings.
Correct Answer: C
Type: Theory
Difficulty: Hard

Unit: 3
CO: CO3
Q22. Given a term "salad" which appears in 2 of Shakespeare's 37 plays, what is the Inverse Document Frequency ($idf$) measure using the formula $\log(N/df)$ (where $N=37$)?
A. $idf = 0$.
B. $idf = \log(37/37) = 0$.
C. $idf = \log(37/2) \approx 1.27$.
D. $idf = \log(37/4) \approx 0.966$.
Correct Answer: C
Type: Numerical
Difficulty: Medium

Unit: 3
CO: CO3
Q23. In the context of TF-IDF, what model does it use to represent a document?
A. A directed acyclic graph (DAG) maintaining dependency relations.
B. A multiset of words, disregarding word order (Bag-of-Words model).
C. A dense vector of floating-point values capturing semantic similarity.
D. A sequence-to-sequence model preserving temporal ordering.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q24. In a Keras sequential model, an `Embedding` layer outputs a 3D tensor of shape `(None, 100, 16)` for a batch of input sequences. If this output is fed directly into a `GlobalAveragePooling1D()` layer, what will be the shape of the resulting tensor?
A. (None, 100).
B. (None, 1).
C. (None, 16).
D. (None, 100, 1).
Correct Answer: C
Type: Code
Difficulty: Medium

Unit: 3
CO: CO3
Q25. In the high-level architecture of the KerasNLP BertClassifier, which component is responsible for converting preprocessed tensors into dense features where the core calculation ("magic") happens?
A. BertPreprocessor.
B. BertTokenizer.
C. BertClassifier `from_preset` method.
D. BertBackbone.
Correct Answer: D
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q26. The training phase of a model that automatically extracts, classifies, and labels elements of text/voice and assigns a statistical likelihood to each possible meaning is characteristic of which approach to NLP?
A. Rules-based NLP.
B. Statistical NLP.
C. Generative AI.
D. Dependency Parsing.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q27. The Stanford Large Movie Reviews Dataset, commonly used for sentiment classification tasks with deep learning models, is characterized by having movie reviews labeled with which type of class indicator?
A. Named entities (PERSON, ORG).
B. Dialogue act types (Question, Statement).
C. Binary integers indicating positive or negative sentiment.
D. Part-of-Speech tags.
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q28. A key challenge when using the traditional Bag-of-Words (BoW) representation is that:
A. It fails to capture the frequency of words.
B. It disregards the grammar and sequential ordering of words.
C. It produces sparse vectors of very low dimensionality.
D. It automatically groups synonyms, removing complexity.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q29. Which core component allows a Recurrent Neural Network (RNN) to handle sequential data like text by maintaining an internal state or "memory" from preceding tokens?
A. Convolutional filtering.
B. Looping back, remembering earlier words while reading new ones.
C. Dropout regularization.
D. Softmax activation.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q30. In the Bangla gender identification study, the researchers generated word embedding vectors of a dimension of 300 for each word from user posts. If a sentence is limited to 100 words, what is the output shape from the embedding layer (assuming batch size of 1)?
A. (1, 100).
B. (1, 300).
C. (1, 100, 300).
D. (1, 300, 100).
Correct Answer: C
Type: Numerical
Difficulty: Medium

Unit: 3
CO: CO3
Q31. The primary motivation for applying Inverse Document Frequency (IDF) when weighting terms in a corpus is to reduce the undue influence of:
A. Extremely long documents.
B. Terms that occur very infrequently.
C. Terms that occur across almost all documents (common terms).
D. Arbitrary integer encodings.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q32. A data scientist is setting up an NLP pipeline for recruiting. Which of the following is a key application area in HR that leverages NLP?
A. Analyzing earnings calls for stock prediction.
B. Automatically screening resumes and matching candidates to job requirements.
C. Extracting clinical notes for computational phenotyping.
D. Identifying fraudulent transactions.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q33. What is the key advantage of Word Embeddings over a simple integer encoding approach for representing words?
A. Word Embeddings are sparse, saving memory.
B. Word Embeddings are manually specified, ensuring high precision.
C. Word Embeddings encode semantic meaning such that similar words have similar dense vector representations.
D. Word Embeddings are always 8-dimensional, simplifying model design.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q34. If an input batch is passed to a Keras `Embedding` layer with shape `(32, 10)` (32 samples of length 10), and the embedding dimension is 64, what will be the output shape?
A. (32, 64).
B. (32, 10).
C. (10, 64).
D. (32, 10, 64).
Correct Answer: D
Type: Code
Difficulty: Hard

Unit: 3
CO: CO3
Q35. When pre-processing text for a sequence model in Keras, using `padding='post'` with `tf.keras.preprocessing.sequence.pad_sequences` ensures that:
A. Sequences are truncated from the start if they exceed maximum length.
B. Sequences are padded with zero values at the beginning if they are shorter than maximum length.
C. Sequences are padded with zero values at the end if they are shorter than maximum length.
D. Sequences are automatically normalized to floating-point values.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q36. What is the typical effect of applying the `Dropout` layer (e.g., `Dropout(0.2)`) during the training phase of a Deep Learning model like an LSTM classifier?
A. It speeds up the training process by optimizing the learning rate.
B. It increases the dimensionality of the model's output.
C. It helps prevent the model from overfitting to the training data.
D. It converts the output to a sparse matrix.
Correct Answer: C
Type: Code
Difficulty: Easy

Unit: 3
CO: CO3
Q37. The "curse of dimensionality" is often cited as a challenge when working with Bag-of-Words (BoW) models primarily because:
A. The number of terms (dimensions) often exceeds the number of documents, leading to high-dimensional and sparse representations.
B. The models require semantic labels that are themselves complex vectors.
C. The models cannot handle sequential data.
D. The computational cost of word embeddings is too high.
Correct Answer: A
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q38. Which phase in the typical NLP pipeline involves feeding the processed data into a machine learning model which adjusts its parameters to minimize errors and improve performance on unseen data?
A. Text Preprocessing.
B. Feature Extraction.
C. Model Training.
D. Semantic Analysis.
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q39. What are N-grams?
A. Random vectors used to initialize embedding layers.
B. Arbitrary integer indices assigned to words.
C. Consecutive word sequences (tokens) of length $n$.
D. Specialized gates within an LSTM network.
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q40. A Keras model layer is defined as: `tf.keras.layers.Dense(3, activation='softmax')`. What is the intended output for a multi-class text classification task (e.g., classifying text into "positive," "neutral," or "negative" sentiment)?
A. A binary prediction (0 or 1).
B. A single floating point value.
C. A vector of 3 probabilities summing to 1, indicating class likelihood.
D. The raw text sequence.
Correct Answer: C
Type: Code
Difficulty: Easy

Unit: 3
CO: CO3
Q41. Transformer models, unlike sequence-to-sequence RNNs, rely on what core mechanism to capture dependencies and relationships between different parts of a sequence?
A. Convolutional filtering.
B. Logarithmic frequency scaling.
C. Tokenization and self-attention.
D. Random indexing.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q42. In the Inverse Document Frequency formula $idf(t, D) = \log(N/n_t)$, what does the variable $n_t$ represent?
A. The total number of terms in the document.
B. The frequency of term $t$ in a specific document $d$.
C. The total number of unique terms (vocabulary size) in the corpus.
D. The number of documents where the term $t$ appears.
Correct Answer: D
Type: Numerical
Difficulty: Medium

Unit: 3
CO: CO3
Q43. When a specialized model (e.g., a Bigram Tagger) cannot determine a tag for a given context, NLP systems often resort to a procedure that falls back on a more general model (e.g., a Unigram Tagger). What is this procedure called?
A. Smoothing.
B. Pruning.
C. Backoff.
D. Recursive Descent.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q44. Which Keras layer is typically placed after an `Embedding` layer, especially in models dealing with variable-length input sequences, to produce a fixed-length output vector for further processing by Dense layers?
A. `tf.keras.layers.TextVectorization`.
B. `tf.keras.layers.Dropout`.
C. `tf.keras.layers.GlobalAveragePooling1D`.
D. `tf.keras.layers.LSTM`.
Correct Answer: C
Type: Code
Difficulty: Easy

Unit: 3
CO: CO3
Q45. What is the difference between a generative classifier (like Naive Bayes) and a conditional classifier (like Maximum Entropy) in terms of what they model?
A. Generative classifiers model $P(\text{label}|\text{input})$; conditional classifiers model $P(\text{input}|\text{label})$.
B. Generative classifiers model the joint probability $P(\text{input}, \text{label})$; conditional classifiers model the conditional probability $P(\text{label}|\text{input})$.
C. Generative classifiers use only binary features; conditional classifiers use numeric features.
D. Generative classifiers use deep learning; conditional classifiers use traditional machine learning.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q46. Author Profiling (AP) is a method that uses shared information to infer an author’s demographic and psychological traits, such as gender, age, or personality. What is the primary content source AP relies on?
A. Structured database entries.
B. Image recognition.
C. Text-based online content and the author’s unique writing style.
D. Acoustic sound waves.
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q47. If a Keras `Embedding` layer is defined with `vocab_size=1000` and `embedding_dim=5`, and the layer is created with random weights, what is the total number of trainable parameters in this layer?
A. 1005.
B. 5000.
C. 1000.
D. 5.
Correct Answer: B
Type: Numerical
Difficulty: Medium

Unit: 3
CO: CO3
Q48. The Keras `Embedding` layer requires an `input_dim` parameter. What linguistic metric must this parameter correspond to?
A. The total number of documents in the corpus.
B. The maximum sequence length of the input sentences.
C. The size of the vocabulary (number of unique tokens).
D. The chosen dimensionality of the output vector.
Correct Answer: C
Type: Theory
Difficulty: Hard

Unit: 3
CO: CO3
Q49. In classifier evaluation, what is the term for the set of annotated data that is kept separate and is unused during the training and feature selection process, serving only for the final assessment of the model's generalization ability?
A. Training set.
B. Dev-test set.
C. Development set.
D. Test set.
Correct Answer: D
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q50. When implementing word embeddings using Word2Vec models in Keras/TensorFlow, how can you ensure that pre-trained weights initialized in the `Embedding` layer are NOT updated during the subsequent model training phase?
A. Set the optimizer to 'adam'.
B. Use binary cross entropy loss.
C. Initialize the embedding layer with a `non-trainable` property.
D. Apply `GlobalAveragePooling1D` immediately after the embedding layer.
Correct Answer: C
Type: Code
Difficulty: Medium

Unit: 3
CO: CO3
Q51. What is the fundamental goal of Named Entity Recognition (NER) in the context of information extraction?
A. To convert the text into a sparse matrix representation.
B. To identify and extract key words or phrases as useful entities (e.g., person names, locations).
C. To determine the emotional tone of the text.
D. To split the text into tokens and remove stop words.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q52. In feature engineering, if an analyst uses a "kitchen sink" approach (including all imaginable features) on a relatively small training dataset, what risk does this pose, leading to a system that performs poorly on new, unseen data?
A. Underfitting.
B. High computational overhead.
C. Overfitting, where the model learns idiosyncrasies of the training data.
D. Lack of expressive power.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q53. A central principle in formal semantics for natural language, which posits that the meaning of a complex expression is a function of the meanings of its parts and their mode of combination, is known as the:
A. Principle of Unbounded Dependency.
B. Principle of Compositionality (Frege's Principle).
C. Maximum Entropy Principle.
D. Naive Bayes Assumption.
Correct Answer: B
Type: Theory
Difficulty: Hard

Unit: 3
CO: CO3
Q54. In the Keras Sequential model definition below, if `vocab_size` is 1000, `embedding_dim` is 32, and the input sequence length is 50, what is the output shape *after* the `Embedding` layer?

```python
model = Sequential([
    TextVectorization(...),
    Embedding(1000, 32),
    GlobalAveragePooling1D(),
    Dense(1)
])
```
A. (None, 50).
B. (None, 32).
C. (None, 50, 32).
D. (None, 1000, 32).
Correct Answer: C
Type: Code
Difficulty: Medium

Unit: 3
CO: CO3
Q55. Why are tokens like "is," "the," and "a" often removed during the text preprocessing stage?
A. They are essential for capturing sentiment.
B. They are unique to specific genres.
C. They are frequently used filler words (stop words) that provide little informational content.
D. They are always polysemous.
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q56. Using a Document-Term Matrix (DTM) representation, if a corpus has 50 documents and a vocabulary size of 5,000 unique terms, what are the dimensions of the resulting TDM (Term-Document Matrix)?
A. 50 rows $\times$ 5,000 columns.
B. 5,000 rows $\times$ 50 columns.
C. 5,000 rows $\times$ 5,000 columns.
D. 50 rows $\times$ 50 columns.
Correct Answer: B
Type: Numerical
Difficulty: Medium

Unit: 3
CO: CO3
Q57. In the KerasNLP architecture provided for a BERT Classifier, which component is primarily responsible for converting input strings into dictionary representations before passing them to the backbone model?
A. BertBackbone.
B. BertClassifier `from_preset` method.
C. BertTokenizer, invoked within the BertPreprocessor.
D. Dense layer.
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q58. What role does the `Tokenizer` typically play in the TensorFlow/Keras NLP pipeline when preparing data for an `Embedding` layer?
A. It calculates the similarity matrix between tokens.
B. It converts raw words into numerical indices (IDs).
C. It learns the dense floating-point vector for each word.
D. It performs one-hot encoding.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q59. When setting up a deep learning model for text classification in Keras, which layer type maps the integer-encoded vocabulary produced by the Tokenizer/TextVectorization into dense vectors of trainable parameters?
A. `tf.keras.layers.Dense`.
B. `tf.keras.layers.Embedding`.
C. `tf.keras.layers.GlobalAveragePooling1D`.
D. `tf.keras.layers.LSTM`.
Correct Answer: B
Type: Code
Difficulty: Easy

Unit: 3
CO: CO3
Q60. Which key assumption of the Naive Bayes model can lead to the "double-counting" problem when features are highly correlated, reducing the model's justification?
A. The assumption that all words follow Zipf's law.
B. The assumption that all input features are continuous numerical values.
C. The assumption that every feature is entirely independent of every other feature, given the label.
D. The assumption that the model must maximize entropy.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q61. The Maximum Entropy (MaxEnt) classifier uses iterative optimization techniques to find a set of parameters that maximizes the performance of the classifier. What is the fundamental principle guiding this choice among distributions that are consistent with known information?
A. Maximizing the frequency of the training corpus.
B. Maximizing the likelihood ratio.
C. Maximizing entropy (making the fewest unwarranted assumptions).
D. Maximizing feature independence.
Correct Answer: C
Type: Theory
Difficulty: Hard

Unit: 3
CO: CO3
Q62. What is the primary purpose of creating a Custom Stop Word Dictionary?
A. To simplify lemmatization rules.
B. To include domain- or project-specific filler words that do not provide informative content for the analysis.
C. To limit the vocabulary size to the top 1000 words.
D. To ensure all words are converted to a sparse vector.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q63. If a training set contains 10,000 unique words (vocabulary size) and is processed by a Keras NLP pipeline, what should be the required value for the `input_dim` parameter of the `Embedding` layer?
A. 10,000 (Vocabulary size).
B. 100 (Sequence length).
C. 16 (Embedding dimension).
D. 1 (Output dimension).
Correct Answer: A
Type: Numerical
Difficulty: Hard

Unit: 3
CO: CO3
Q64. In a binary text classification problem using the Adam optimizer and BinaryCrossentropy loss, the initial training output shows: `loss: 0.6920 - accuracy: 0.5028 - val_loss: 0.6904 - val_accuracy: 0.4886`. What concept is immediately demonstrated if, after several epochs, the training accuracy continues to rise dramatically while the validation accuracy plateaus or declines?
A. Overfitting.
B. Underfitting.
C. Vanishing gradients.
D. Perfect model fit.
Correct Answer: A
Type: Code
Difficulty: Medium

Unit: 3
CO: CO3
Q65. In the context of the Bangla gender identification study, what class of features refers to those designed to capture the unique writing style of an author, including aspects like vocabulary richness, punctuation usage, and structural organization?
A. Transformer features.
B. Stylometric features.
C. Semantic features.
D. First-order logic features.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q66. Which technique provides a way to reduce a high-dimensional and sparse Document-Term Matrix (DTM) into a smaller approximation consisting of fewer latent dimensions, often for semantic space modeling?
A. Term Frequency-Inverse Document Frequency (TF-IDF).
B. Latent Semantic Analysis (LSA) using Singular Value Decomposition (SVD).
C. Consecutive Classification.
D. The Maximum Entropy Principle.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q67. What primary limitation of static word embeddings (like traditional Word2Vec or GloVe) did later token-level embeddings (like BERT) aim to address?
A. The inability to represent words as dense vectors.
B. The difficulty in learning weights via neural networks.
C. The fact that words with multiple meanings (polysemy/homonymy) were conflated into a single vector representation.
D. The constraint that embedding dimensions must be very high.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q68. Given the goal of sentiment classification in Keras, how would you retrieve the learned word embeddings from a trained model named `model` if the `Embedding` layer was explicitly named "embedding"?
A. `model.get_layer('embedding').weights`.
B. `model.layers.get_config()`.
C. `model.predict(input_data)`.
D. `model.compile().metrics`.
Correct Answer: A
Type: Code
Difficulty: Medium

Unit: 3
CO: CO3
Q69. What is the fundamental difference between word embeddings expressed as vectors of co-occurring words and embeddings expressed as vectors of linguistic contexts?
A. Co-occurrence vectors focus on direct semantic association; context vectors focus on syntactic parsing.
B. Co-occurrence vectors are sparse; context vectors are dense.
C. Co-occurrence vectors use only unsupervised learning; context vectors use only supervised learning.
D. They represent the same information and are mathematically equivalent.
Correct Answer: A
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q70. In the Keras NLP pipeline, the input to the `Embedding` layer is a batch of integer sequences of shape `(None, sequence_length)`. If the embedding dimension is `embedding_dim`, what is the matrix shape of the trainable weights learned within this layer?
A. (1, embedding\_dim).
B. (sequence\_length, embedding\_dim).
C. (vocab\_size, embedding\_dim).
D. (sequence\_length, vocab\_size).
Correct Answer: C
Type: Code
Difficulty: Medium

Unit: 3
CO: CO3
Q71. Which model, commonly used for sequence classification in NLP, finds the most likely label for the first input, uses that answer to help find the next label, and repeats this process until all inputs are labeled?
A. Maximum Entropy classifier.
B. Generative classifier.
C. Consecutive classification (greedy sequence classification).
D. Decision Tree.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q72. What is the term for the process in which an NLP system attempts to select the appropriate meaning for a word that has multiple dictionary definitions (e.g., distinguishing "bank" as a financial institution versus a river bank)?
A. Part-of-Speech Tagging.
B. Named Entity Recognition.
C. Coreference Resolution.
D. Word Sense Disambiguation.
Correct Answer: D
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q73. When training a sentiment classifier in Keras (as per the IMDb tutorial), the loss function used is `tf.keras.losses.BinaryCrossentropy(from_logits=True)`. This loss function is appropriate when the final Dense layer has:
A. No activation function (raw logits) and 1 output unit.
B. 'softmax' activation and 2 output units.
C. 'sigmoid' activation and 1 output unit.
D. 'tanh' activation and variable output units.
Correct Answer: A
Type: Code
Difficulty: Medium

Unit: 3
CO: CO3
Q74. For large-scale corpus processing in Python, what is the key benefit of using a generator expression like `(w.lower() for w in text)` inside a function call (e.g., `max()`), compared to first creating a full list comprehension `[w.lower() for w in text]`?
A. The list comprehension is more readable and easier to debug.
B. The generator expression streams data and does not require allocating memory for the full list object.
C. The list comprehension executes faster in all cases.
D. The generator expression automatically handles type checking.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q75. Which type of advanced deep learning model is characterized by being trained specifically to predict the next word in a sequence, marking a significant step in text generation ability?
A. Recurrent Neural Networks (RNN).
B. Sequence-to-Sequence (Seq2seq) models.
C. Autoregressive models.
D. Convolutional Neural Networks (CNN).
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q76. The term "Stylometric features" generally encompasses metrics related to four main categories of writing style. These categories typically include lexical features, structural features, syntactic features, and what other type of feature?
A. Gender-neutral features.
B. Context-specific features.
C. Positional features.
D. Global averaging features.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q77. A Keras classification model summary shows the following parameter counts:
*   Embedding layer: 160000 parameters
*   Dense (16 units): 272 parameters
*   Dense (1 unit): 17 parameters

What is the total number of trainable parameters in this model (assuming 0 non-trainable parameters)?
A. 160,000.
B. 160,289.
C. 160,272.
D. 17.
Correct Answer: B
Type: Code
Difficulty: Medium

Unit: 3
CO: CO3
Q78. In developing broad-coverage, rule-based grammars, why did the astronomical growth in the number of possible parse trees for longer sentences become a major efficiency problem?
A. The parsing algorithms themselves were only cubic complexity, making the task easy.
B. Lexical items could be made completely unambiguous.
C. Humans process long sentences effortlessly, making the computational models seem trivial.
D. Ambiguity increases with grammar coverage, leading to an intractable number of potential analyses (e.g., Catalan numbers).
Correct Answer: D
Type: Theory
Difficulty: Hard

Unit: 3
CO: CO3
Q79. When performing Latent Semantic Analysis (LSA), the choice of retaining $k$ singular vectors is a crucial decision. If $k$ is too low, what primary risk does the analyst face?
A. Overfitting the training data.
B. The complexity being too high.
C. Losing important semantic information.
D. SVD decomposition failure.
Correct Answer: C
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q80. In supervised classification, why is it vital to use a separate Dev-Test set (distinct from the final Test Set) during the model development process, especially when iterating on feature selection?
A. To guarantee the model achieves 100% accuracy on the final test set.
B. To prevent the classifier from reflecting idiosyncrasies of the Dev-Test set during feature refinement.
C. To reduce computational complexity by simplifying the feature set.
D. To ensure the final Test Set is as similar as possible to the training data.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q81. If a classification model correctly labels 75 documents out of a test set of 100, what is the accuracy of the classifier?
A. 75%.
B. 100%.
C. 25%.
D. 175%.
Correct Answer: A
Type: Numerical
Difficulty: Easy

Unit: 3
CO: CO3
Q82. The Word2Vec Skip-Gram model, designed for generating word embeddings, is characterized by taking the current word as input and predicting what component?
A. The semantic context of the word only.
B. The surrounding window of context words.
C. The raw count of the word in the corpus.
D. The Part-of-Speech tag of the word.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q83. Which Keras layer, applied immediately after an Embedding layer in a classification model, mathematically summarizes the input sequence by averaging across the temporal (sequence) dimension?
A. `tf.keras.layers.Dense`.
B. `tf.keras.layers.LSTM`.
C. `tf.keras.layers.GlobalAveragePooling1D`.
D. `tf.keras.layers.TextVectorization`.
Correct Answer: C
Type: Code
Difficulty: Easy

Unit: 3
CO: CO3
Q84. The Brill Tagger uses an analogy to explain its process of iteratively fixing errors. This analogy compares the tagging process to initially covering the entire canvas with a uniform color before applying successively finer details (correction rules). What activity is used in this analogy?
A. Database querying.
B. Scientific experimentation.
C. Painting a tree against a sky-blue background.
D. Binary searching for a word.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q85. In NLP, Convolutional Neural Networks (CNNs) are primarily used in text processing for which purpose?
A. Capturing long-term dependencies through gates.
B. Identifying local patterns, similar to filter application in image processing.
C. Transforming raw strings into integer indices.
D. Generating text one word at a time.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q86. When addressing the sparse data problem for N-gram taggers that encounter contexts unseen in the training data, a common method is Backoff. If backoff options are exhausted, the tagger often defaults to which final model type?
A. The most complex trigram model.
B. A regular expression tagger or a default tagger (e.g., tagging everything as NN).
C. Word embedding vectors.
D. A transformer model.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q87. Document 1 (D1) has a total of 5 terms. The word "this" appears once in D1. Using the raw count method, calculate the term frequency $tf(t, d)$ for the term "this" in Document 1.
A. 1.
B. 5.
C. $1/5 = 0.2$.
D. 0.
Correct Answer: C
Type: Numerical
Difficulty: Medium

Unit: 3
CO: CO3
Q88. Multi-sense embeddings (or contextually-meaningful embeddings like BERT) were developed to properly handle what specific linguistic challenge?
A. Arbitrary integer encoding.
B. Conflating words with multiple meanings (polysemy and homonymy) into a single vector.
C. The requirement for manual feature specification.
D. Low IDF scores.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q89. In KerasNLP, the `from_preset` method used in conjunction with a `BertClassifier` is typically responsible for:
A. Defining the text preprocessing steps like tokenization.
B. Loading up a ready-to-use model with pre-trained weights.
C. Calculating the Maximum Entropy of the input features.
D. Freezing the embedding weights.
Correct Answer: B
Type: Code
Difficulty: Medium

Unit: 3
CO: CO3
Q90. Which of the following machine learning techniques minimizes the total likelihood of the training corpus while maximizing the entropy of the probability distribution?
A. Naive Bayes Classifier.
B. Latent Semantic Analysis.
C. Maximum Entropy Classifier.
D. Decision Tree Classifier.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q91. What factor was identified in the Bangla gender identification study as being a major strength of TF-IDF over word embeddings like CBOW and Skip-Gram for that specific task?
A. Word embeddings require less training data.
B. Word embeddings excel at capturing subtle stylistic variations critical for gender classification.
C. TF-IDF's emphasis on term distinctiveness aligns better with the need to capture subtle stylistic variations in gendered language use.
D. TF-IDF models automatically integrate long-term contextual dependencies.
Correct Answer: C
Type: Theory
Difficulty: Hard

Unit: 3
CO: CO3
Q92. The TIMIT Corpus, cited as the first widely distributed annotated speech database, was structured to achieve a balance across multiple dimensions. These dimensions primarily included dialect regions, speakers, and what other type of material?
A. Financial reports.
B. Phonetically rich sentences.
C. Legal documentation.
D. Novel prose.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q93. The Keras `TextVectorization` layer uses `output_sequence_length` to ensure all input samples have a uniform length. If input sentences are shorter than this length, what typically happens?
A. They are truncated.
B. They are padded with zero values.
C. They are duplicated until the length is met.
D. An error is raised due to mismatched shapes.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q94. Which of the following is the standard formal notation for a production rule in a simple dependency grammar, where "shot" is the head and "I" is the dependent?
A. `S -> 'I' 'shot'`.
B. `'shot' -> 'I'`.
C. `V -> 'shot' NP`.
D. `'I' -> 'shot'`.
Correct Answer: B
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q95. When analyzing text, the goal of converting a word $w$ (e.g., 'running') into its dictionary form $l$ (e.g., 'run') is known as:
A. Tokenization.
B. Stemming.
C. Lemmatization.
D. Normalization (Stemming/Lemmatization).
Correct Answer: D
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q96. Document 2 (D2) has a total of 7 terms. The term "example" occurs 3 times. If $idf(\text{"example"}, D) = 0.301$, what is the final $tfidf(\text{"example"}, D2, D)$ using the formula $tf(t,d) \cdot idf(t,D)$?
A. $0$.
B. $0.429 \times 0.301 \approx 0.129$.
C. $3 \times 0.301 \approx 0.903$.
D. $7 \times 0.301 \approx 2.107$.
Correct Answer: B
Type: Numerical
Difficulty: Hard

Unit: 3
CO: CO3
Q97. In the NLP pipeline, the process of converting features extracted from text (like raw counts or TF-IDF weights) into structured, numerical data that models can process is known as:
A. Text Preprocessing.
B. Feature Extraction (or Vectorization).
C. Linguistic Analysis.
D. Model Deployment.
Correct Answer: B
Type: Theory
Difficulty: Easy

Unit: 3
CO: CO3
Q98. What machine learning concept is exemplified when a model is developed to learn multiple related tasks simultaneously, potentially reducing development time and improving performance?
A. Self-supervised learning.
B. Federated learning.
C. Multi-task learning.
D. Latent semantic indexing.
Correct Answer: C
Type: Theory
Difficulty: Medium

Unit: 3
CO: CO3
Q99. In the Keras code snippet provided for sentiment classification, the `Embedding` layer has `input_dim=vocab_size` and `output_dim=embedding_dim`. The `GlobalAveragePooling1D` layer reduces the sequence dimension. If this model were modified to use an `LSTM(32)` layer instead of pooling, what would be the output shape immediately following the `LSTM(32)` layer (assuming `return_sequences=False`)?
A. (None, 50, 32).
B. (None, 32).
C. (None, 100, 32).
D. (None, 100).
Correct Answer: B
Type: Code
Difficulty: Medium

Unit: 3
CO: CO3
Q100. In the process of developing a supervised classifier model, when should the final performance evaluation using the dedicated Test Set be conducted?
A. Continuously throughout training to monitor loss.
B. Only after the model training is complete and features have been finalized using the Dev-Test set.
C. Before any feature extraction begins.
D. During the initial data collection phase.
Correct Answer: B
Type: Theory
Difficulty: Medium