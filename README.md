# Translation Model With Transformer

This project implements a sequence-to-sequence translation model using Transformer architecture. It includes functionalities like beam and greedy search decoding, attention visualization, and efficient training. The project focuses on translating text from English to Arabic using advanced techniques.

## Features

- **Sequence-to-Sequence Translation**: Translates text from English to Arabic.
- **Greedy Search Decoding**: Efficient search for optimal translation sequences.
- **Attention Visualization**: Insights into the model's decision-making process.
- **Modular Design**: Configurable architecture for training and inference.
- **Pre-trained Weights**: Ready-to-use weights for quick deployment.

---

## Files and Directories

### Core Files
- **`model.py`**: Contains the Transformer model architecture and its components.
- **`dataset.py`**: Handles data loading, preprocessing, and tokenization.
- **`config.py`**: Configures hyperparameters, file paths, and other project settings.
- **`train.py`**: The main script for training the translation model.
- **`translate.py`**: Provides inference capabilities for translating input sentences.

### Jupyter Notebooks
- **`colab_training.ipynb`**: A Colab-compatible script for training the model.
- **`Kaggle-training-4.ipynb`**: A notebook for training on Kaggle's infrastructure.
- **`Beam_Search.ipynb`**: Demonstrates the beam search decoding process.
- **`attention_visual.ipynb`**: Visualizes attention weights in the Transformer model.

### Pre-trained Weights and Tokenizers
- **`weights/Tmodel_70.pt`**: Pre-trained model weights for deployment.
- **`tokenizer_text_ar.json`**: Arabic tokenizer configuration.
- **`tokenizer_text_en.json`**: English tokenizer configuration.

### Dependencies
- **`requirements.txt`**: Specifies required Python libraries and versions.
- 

## Download the Pre-trained Weights

To download the pre-trained model weights, use the following link:

https://drive.google.com/file/d/1-YSZq_vgRcl5eEMbcmdLpLhhDncBtnz4/view?usp=sharing
after downloading the file put it in the weights folder

## How to Run

1. Clone the repository:

```bash
git clone <repository-url>
```

2. Navigate to the project directory:

```bash
cd Machine-translation
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Run the program:

```bash
streamlit run translate.py
```

This will start the Streamlit UI.

## How It Works

1. **Tokenization**: The input text is preprocessed and tokenized using pre-trained tokenizers (tokenizer_text_en.json) and (tokenizer_text_ar.json).
2. **Encoding**: The source sequence is passed through the Transformer encoder, generating context-aware embeddings.
3. **Greedy Search Decoding:**: The decoder uses a greedy search strategy to find the most likely target word.
4. **Translation**: The tokenized target sequence is converted back into human-readable text.


## Contributing

If you'd like to contribute to this project, feel free to fork the repository, make changes, and submit a pull request. Contributions are always welcome!
