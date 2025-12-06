import torch
import torch.nn as nn
from torch.utils.data import Dataset


def causal_mask(size):
    # Shape: (size, size)
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.bool)
    return mask

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len_src, seq_len_tgt):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len_src = seq_len_src
        self.seq_len_tgt = seq_len_tgt

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        
        self.pad_token_src = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)
        self.pad_token_tgt = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]

        if src_text is None or tgt_text is None:
            return None  # Handled by custom_collate_fn

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # --- FIX 1: Handle Truncation and Padding ---

        # 1. For Encoder Input
        num_enc_tokens = len(enc_input_tokens)
        if num_enc_tokens > self.seq_len_src - 2: # -2 for [SOS] and [EOS]
            enc_input_tokens = enc_input_tokens[:self.seq_len_src - 2]
        
        enc_num_padding_tokens = self.seq_len_src - len(enc_input_tokens) - 2
        
        encoder_input = torch.cat(
            [
                self.sos_token, # Using tgt_sos, assuming same ID or standard practice
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token, # Using tgt_eos
                torch.tensor([self.pad_token_src] * enc_num_padding_tokens, dtype=torch.int64), # Use SRC pad
            ],
            dim=0,
        )

        # 2. For Decoder Input
        num_dec_tokens = len(dec_input_tokens)
        if num_dec_tokens > self.seq_len_tgt - 1: # -1 for [SOS]
            dec_input_tokens = dec_input_tokens[:self.seq_len_tgt - 1]

        dec_num_padding_tokens = self.seq_len_tgt - len(dec_input_tokens) - 1
        
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token_tgt] * dec_num_padding_tokens, dtype=torch.int64), # Use TGT pad
            ],
            dim=0,
        )

        # 3. For Label (shifted decoder input)
        # Use the *original* decoder tokens before truncation (or after, if we ensure label matches)
        # We need to ensure label matches the *original* sequence length for the loss
        
        label_tokens = dec_input_tokens 
        if len(label_tokens) > self.seq_len_tgt - 1: 
             label_tokens = label_tokens[:self.seq_len_tgt - 1] 
             
        label_num_padding_tokens = self.seq_len_tgt - len(label_tokens) - 1

        label = torch.cat(
            [
                torch.tensor(label_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token_tgt] * label_num_padding_tokens, dtype=torch.int64), 
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.seq_len_src
        assert decoder_input.size(0) == self.seq_len_tgt
        assert label.size(0) == self.seq_len_tgt

        # For PyTorch Transformer masks
        src_padding_mask = (encoder_input == self.pad_token_src.item())
        tgt_padding_mask = (decoder_input == self.pad_token_tgt.item())
        tgt_causal_mask = causal_mask(self.seq_len_tgt)

        # For the "scratch" model masks
        scratch_encoder_mask = (encoder_input != self.pad_token_src.item()).unsqueeze(0).unsqueeze(0).int() # (1, 1, seq_len)
        scratch_decoder_mask = (decoder_input != self.pad_token_tgt.item()).unsqueeze(0).int() & (causal_mask(decoder_input.size(0)).eq(0)) # (1, seq_len) & (1, seq_len, seq_len)

        return {
            "encoder_input": encoder_input, 
            "decoder_input": decoder_input,  
            "label": label,  
            "src_text": src_text,
            "tgt_text": tgt_text,

              # For the "scratch" model masks
            "scratch_encoder_mask": scratch_encoder_mask,
            "scratch_decoder_mask": scratch_decoder_mask,
             
              # For PyTorch Transformer masks
            "src_padding_mask": src_padding_mask,
            "tgt_padding_mask": tgt_padding_mask,
            "tgt_causal_mask": tgt_causal_mask
        }