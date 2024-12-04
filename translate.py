import streamlit as st
from pathlib import Path
from dataset import causal_mask
from config import get_config, latest_weights_file_path 
from model import build_transformer
from tokenizers import Tokenizer
import torch
from googletrans import Translator
import inflect

def numbers_to_arabic_words(number):
    # Convert number to English words
    p = inflect.engine()
    english_words = p.number_to_words(number)
    
    # Translate English words to Arabic
    translator = Translator()
    arabic_translation = translator.translate(english_words, src='en', dest='ar').text
    
    return arabic_translation


# Translate function (same as the one you've written)
def translate(sentence: str):
    # Check if the input is numeric and return it directly
    if isinstance(sentence, int) or (isinstance(sentence, str) and sentence.isdigit()):
        return numbers_to_arabic_words(sentence)

    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = get_config()
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config["seq_len"], config['seq_len'], d_model=config['d_model']).to(device)

    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename, map_location=torch.device(f'{device}'))
    model.load_state_dict(state['model_state_dict'])

    # Translate the sentence
    seq_len = config['seq_len']
    model.eval()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        source = tokenizer_src.encode(sentence)
        source = torch.cat([ 
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)
        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
        encoder_output = model.encode(source, source_mask)

        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)

        # Generate the translation word by word
        translated_sentence = ""
        while decoder_input.size(1) < seq_len:
            # Build mask for target and calculate output
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # Project next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

            # Append the translated word
            translated_sentence += f"{tokenizer_tgt.decode([next_word.item()])} "

            # Stop if we reach the end-of-sentence token
            if next_word == tokenizer_tgt.token_to_id('[EOS]'):
                break

    # Return the translated sentence
    return translated_sentence.strip()



# Streamlit UI setup
def main():
    st.title("English to Arabic Machine Translation")
    
    # Inform the user about the language translation direction
    st.markdown("""
    This app uses a transformer-based model to translate sentences **from English to Arabic**. 
    Enter an English sentence below, and the model will generate the translation in Arabic.
    """)

    # Input box for user sentence
    sentence = st.text_area("Enter an English sentence to translate", "I love machine translation.")

    # Translate button
    if st.button("Translate"):
        st.write("Translating...")
        translated = translate(sentence)
        st.subheader("Translated Sentence (Arabic):")
        st.markdown(f"<h1 style='font-size:32px; color:#4682B4;'>{translated}</h1>", unsafe_allow_html=True)



# Run the app
if __name__ == "__main__":
    main()
