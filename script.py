import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample text for demonstration (you should use your own larger corpus)
text = """
Cochin University of Science and Technology (CUSAT) is a premier technical university located in Kerala, India.
CUSAT was established in 1971 and has emerged as a center of excellence for higher education and research.
The university offers undergraduate, postgraduate, and doctoral programs across various disciplines of science, engineering, and technology.
CUSAT is known for its strong focus on research, innovation, and industry collaboration in the field of technology.
The campus is situated in Kochi, also known as Cochin, which is a major port city on the south-west coast of India.
The university has several specialized departments including Ship Technology, Polymer Science, Marine Sciences, and Information Technology.
CUSAT has consistently ranked among the top technical universities in India for its academic excellence and research output.
Students at CUSAT benefit from state-of-the-art laboratories, a central library with extensive resources, and various extracurricular activities.
"""

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Create predictors and target
X, y = input_sequences[:, :-1], input_sequences[:, -1]

# Convert target to one-hot encoding
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Define model parameters
embedding_dim = 50
lstm_units = 100
learning_rate = 0.01

# Build the model
model = Sequential()
model.add(Embedding(total_words, embedding_dim, input_length=max_sequence_len-1))
model.add(LSTM(lstm_units))
model.add(Dense(total_words, activation='softmax'))

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Summarize the model
model.summary()

# Train the model (in actual implementation, use more epochs)
history = model.fit(X, y, epochs=100, verbose=1)

# Function to generate text
def generate_text(seed_text, next_words, model, tokenizer, max_sequence_len):
    """
    Generate text using the trained model
    
    Parameters:
    - seed_text: Initial text to start generation
    - next_words: Number of words to generate
    - model: Trained RNN model
    - tokenizer: Text tokenizer
    - max_sequence_len: Maximum length of sequences used in training
    
    Returns:
    - Generated text
    """
    generated_text = seed_text
    
    for _ in range(next_words):
        # Tokenize the current text
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        # Pad the sequence
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        # Predict the next word
        predicted = np.argmax(model.predict(token_list), axis=-1)
        
        # Get the word from index
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        
        # Add to the generated text
        seed_text += " " + output_word
        generated_text += " " + output_word
    
    return generated_text

# Example of generating text
print("\nText Generation Example")
print("This model can generate text continuations based on a seed phrase.")
print("For example, you could enter 'Cochin University of Science and Technology' to generate text about this prestigious institution.")
print("CUSAT is known for its excellence in technical education and research in Kerala, India.")

# Get seed text from user
seed_text = input("\nEnter your seed text: ")
generated_text = generate_text(
    seed_text, 
    next_words=15, 
    model=model, 
    tokenizer=tokenizer, 
    max_sequence_len=max_sequence_len
)

print("\nGenerated Text:")
print(generated_text)

# Additional information for presentation:
# 1. This is a simple character-level RNN for text generation
# 2. In real applications, you would use:
#    - Larger corpus of text
#    - More complex models (GRU, LSTM)
#    - Word embeddings (Word2Vec, GloVe)
#    - Attention mechanisms
# 3. Popular applications:
#    - Text completion
#    - Chatbots
#    - Content generation
#    - Translation