# Recurrent Neural Network Text Generation

This project demonstrates how to build and train a Recurrent Neural Network (RNN) using TensorFlow and Keras for text generation. The model is trained on a small corpus about Cochin University of Science and Technology (CUSAT) and can generate new text based on seed phrases.

## Features

- Text preprocessing and tokenization
- LSTM-based neural network model
- Text generation from seed phrases
- Interactive command-line interface

## Requirements

- Python 3.6+
- TensorFlow 2.5.0+
- NumPy 1.19.5+

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Recurrent-Neural-Network.git
   cd Recurrent-Neural-Network
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the script:
```
python script.py
```

The script will:
1. Train an RNN model on the provided text sample
2. Prompt you to enter a seed text
3. Generate text continuation based on your input

Example:
```
Enter your seed text: Cochin University
```

The model will then generate text based on this seed.

## Customization

To use your own text corpus:
1. Replace the `text` variable in `script.py` with your own text
2. Adjust model parameters as needed:
   - `embedding_dim`: Size of the embedding layer
   - `lstm_units`: Number of LSTM units
   - `learning_rate`: Learning rate for the optimizer
   - `epochs`: Number of training epochs

## Project Structure

- `script.py`: Main script containing the RNN model and text generation code
- `requirements.txt`: List of required Python packages

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow and Keras documentation
- Natural Language Processing community
 
