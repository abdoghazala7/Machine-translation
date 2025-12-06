import numpy as np
from datasets import Dataset
from tokenizers import Tokenizer 

def get_optimal_seq_lengths(ds_raw: Dataset, 
                            tokenizer_src: Tokenizer, 
                            tokenizer_tgt: Tokenizer, 
                            config: dict, 
                            percentile: float = 99.0):
    """
    Calculates the optimal sequence length for source and target datasets
    based on a specified percentile.
    
    This uses the fast dataset.map() method.
    
    Args:
        ds_raw: The raw Hugging Face Dataset.
        tokenizer_src: The source language tokenizer.
        tokenizer_tgt: The target language tokenizer.
        config: Dictionary containing 'lang_src' and 'lang_tgt' keys.
        percentile: The percentile to use for length calculation (e.g., 99.0).
        
    Returns:
        A tuple (seq_len_src, seq_len_tgt) containing the recommended lengths.
    """
    
    lang_src = config['lang_src']
    lang_tgt = config['lang_tgt']

    def _get_token_lengths(batch):
        """
        Internal helper function to be used with .map()
        Calculates token lengths for a batch.
        """
        src_sentences = [s if s else "" for s in batch[lang_src]]
        tgt_sentences = [t if t else "" for t in batch[lang_tgt]]
        
        src_encodings = tokenizer_src.encode_batch(src_sentences)
        tgt_encodings = tokenizer_tgt.encode_batch(tgt_sentences)
        
        return {
            'src_len': [len(e.ids) for e in src_encodings],
            'tgt_len': [len(e.ids) for e in tgt_encodings]
        }

    
    ds_with_lengths = ds_raw.map(
        _get_token_lengths, 
        batched=True, 
        num_proc=4 
    )
    
    all_src_lengths = ds_with_lengths['src_len']
    all_tgt_lengths = ds_with_lengths['tgt_len']
    
    seq_len_src = int(np.percentile(all_src_lengths, percentile))
    seq_len_tgt = int(np.percentile(all_tgt_lengths, percentile))
    
    max_src = np.max(all_src_lengths)
    max_tgt = np.max(all_tgt_lengths)

    print("\n--- Sequence Length Analysis ---")
    print(f"--- Source Language ({lang_src}) ---")
    print(f"Absolute max token length: {max_src}")
    print(f"{percentile}th percentile (Recommended seq_len): {seq_len_src}")
    
    print(f"\n--- Target Language ({lang_tgt}) ---")
    print(f"Absolute max token length: {max_tgt}")
    print(f"{percentile}th percentile (Recommended seq_len): {seq_len_tgt}")
    print("----------------------------------\n")
    
    return seq_len_src, seq_len_tgt