import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from pathlib import Path
import os
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import (
    Sequence, 
    NFD, 
    Lowercase, 
    StripAccents, 
    NFKC, 
    Strip, 
    Replace
)
from tokenizers.pre_tokenizers import Whitespace
import warnings
from tqdm import tqdm
import os
import copy
from pathlib import Path
import sys
import numpy as np
import random

# Import the new config functions
from config import get_scratch_config, get_pytorch_config 
# Import the "manager" function (get_model)
from model import get_model 
# Import the dataset and the new causal_mask
from dataset import BilingualDataset, causal_mask 
# Import the function to get optimal sequence lengths
from get_optimal_seq_lengths import get_optimal_seq_lengths
# Import metrics
from torchmetrics.text import BLEUScore, CHRFScore


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
device = torch.device(device)



def build_arabic_normalizer():
    """
    Builds a custom normalizer for the Arabic language.
    This sequence is crucial for handling the complexities of Arabic text.
    """
    return Sequence([
        # 1. Apply NFKC normalization for general Unicode consistency
        NFKC(),
        
        # 2. Remove diacritics (Tashkeel/Harakat)
        # These are marks like Fatha, Damma, Kasra
        Replace(r"[\u064B-\u065F]", ""),
        
        # 3. Remove Tatweel (Kashida)
        # This is the 'ـ' character used for text elongation
        Replace(r"\u0640", ""),
        
        # 4. Unify Alef variants (Alef, Alef with Madda, Alef with Hamza)
        # Replaces (أ, إ, آ) with (ا)
        Replace(r"[\u0622\u0623\u0625]", "\u0627"),
        
        # 5. Unify Alef Maksura and Yeh
        # Replaces (ى) with (ي)
        Replace(r"\u0649", "\u064A"),
        
        # Note: We do NOT normalize Ta-Marbuta (ة) to Heh (ه)
        # because it holds important grammatical meaning for translation.
    ])

def get_all_sentences(ds, lang):
    """
    A generator function to yield sentences one by one from the dataset.
    This is memory-efficient as it doesn't load all sentences at once.
    """
    for item in ds:
        sentence = item.get(lang, None)
        if isinstance(sentence, str) and sentence.strip():
            yield sentence
        else:
            continue

def get_or_build_tokenizer(config, ds, lang):
    """
    Loads a tokenizer if it exists, or builds and trains a new one.
    
    config (dict): A configuration dictionary.
                   Expected keys: 'tokenizer_file', 'lang_src', 'lang_tgt', 'vocab_size'
    ds (Dataset): The dataset to train on.
    lang (str): The language to build the tokenizer for (e.g., 'en' or 'ar').
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not tokenizer_path.exists():
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        
        if lang == config['lang_src']:
            tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
            
        elif lang == config['lang_tgt']:
            tokenizer.normalizer = build_arabic_normalizer()
            
        else:
            tokenizer.normalizer = Lowercase()

        tokenizer.pre_tokenizer = Whitespace()
        
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )
        
        print(f"Starting tokenizer training for '{lang}'...")
        
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
        
        print(f"Tokenizer for '{lang}' trained and saved to {tokenizer_path}")

    else:
        print(f"Found existing tokenizer at {tokenizer_path}, loading it.")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset(f"{config['datasource']}", split='train')

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    split = ds_raw.train_test_split(test_size=0.1, shuffle=True, seed=42)
    train_ds_raw = split['train']   
    temp_ds_raw = split['test']     

    split = temp_ds_raw.train_test_split(test_size=0.5, shuffle=True, seed=42)
    val_ds_raw = split['train']     
    test_ds_raw = split['test']     

    seq_len_src, seq_len_tgt = get_optimal_seq_lengths(train_ds_raw, tokenizer_src, tokenizer_tgt, config) 
    config['seq_len_src'] = seq_len_src
    config['seq_len_tgt'] = seq_len_tgt

    # --- (A) Create the subset for METRIC calculation ---
    subset_size = config['validation_subset_size']
    if subset_size > 0 and subset_size < len(val_ds_raw):
        val_ds_subset_raw = val_ds_raw.select(range(subset_size))
    else:
        val_ds_subset_raw = val_ds_raw # Use full set if size is invalid

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], seq_len_src, seq_len_tgt )
    val_ds_full = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], seq_len_src, seq_len_tgt)
    val_ds_subset = BilingualDataset(val_ds_subset_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], seq_len_src, seq_len_tgt)
    test_ds = BilingualDataset(test_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], seq_len_src, seq_len_tgt)

        
    def custom_collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        return default_collate(batch)
    
    pin_memory = device.type == 'cuda'
    num_workers = 4 if device.type == 'cuda' else 0
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, collate_fn=custom_collate_fn, num_workers=num_workers, pin_memory=pin_memory,persistent_workers=True)

    # Loader for FAST loss calculation (batched)
    val_dataloader_full = DataLoader(val_ds_full, batch_size=config['batch_size'], shuffle=False, collate_fn=custom_collate_fn, num_workers=num_workers, pin_memory=pin_memory,persistent_workers=True)

    # Loader for SLOW metric calculation (bs=1)
    val_dataloader_subset = DataLoader(val_ds_subset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    return train_dataloader, val_dataloader_full, val_dataloader_subset, test_dataloader, tokenizer_src, tokenizer_tgt



# --- Weights File Path ---
def get_weights_file_path(config, best_model_tag: str) -> Path:
    model_folder = Path(config['model_folder'])
    model_folder.mkdir(parents=True, exist_ok=True)
    model_filename = f"{config['model_basename']}_{best_model_tag}.pt"
    return model_folder / model_filename



def greedy_decode(model_core, source, encoder_input_mask_batch, config, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    if config['model_type'] == 'scratch':
        encoder_mask = encoder_input_mask_batch
        encoder_output = model_core.encode(source, encoder_mask)
    else: 
        src_padding_mask = (encoder_input_mask_batch.squeeze(1).squeeze(1) == 0)
        encoder_output = model_core.encode(source, src_padding_mask)

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        if decoder_input.size(1) == max_len:
            break

        if config['model_type'] == 'scratch':
            tgt_len = decoder_input.size(1)
            scratch_causal_mask = (torch.triu(torch.ones((1, tgt_len, tgt_len)), diagonal=1).type(torch.int) == 0).to(device)
            decoder_mask = scratch_causal_mask
            out = model_core.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        
        else: 
            tgt_len = decoder_input.size(1)
            tgt_causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)
            src_padding_mask = (encoder_input_mask_batch.squeeze(1).squeeze(1) == 0)
            tgt_padding_mask = torch.zeros((1, tgt_len), device=device, dtype=torch.bool)

            out = model_core.decode(
                encoder_output, 
                src_padding_mask, 
                decoder_input, 
                tgt_padding_mask, 
                tgt_causal_mask
            )

        prob = model_core.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)


# --- run_validation_loss ---

# This function is FAST. It only calculates loss on the FULL dataset.
def run_validation_loss(
    model_core, val_dataloader_full, loss_fn, config, device, tokenizer_tgt
):
    model_core.eval()
    count = 0
    total_val_loss = 0.0

    with torch.no_grad():
        val_iterator = tqdm(val_dataloader_full, desc="Validating (Loss)")
        
        for batch in val_iterator:
            if batch is None: continue
            count += 1
            
            if config['model_type'] == 'scratch':
                encoder_input = batch['encoder_input'].to(device, non_blocking=True)
                decoder_input = batch['decoder_input'].to(device, non_blocking=True)
                encoder_mask = batch['scratch_encoder_mask'].to(device, non_blocking=True)
                decoder_mask = batch['scratch_decoder_mask'].to(device, non_blocking=True)
                encoder_output = model_core.encode(encoder_input, encoder_mask)
                decoder_output = model_core.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            else: # 'pytorch'
                encoder_input = batch['encoder_input'].to(device, non_blocking=True)
                decoder_input = batch['decoder_input'].to(device, non_blocking=True)
                src_padding_mask = batch['src_padding_mask'].to(device, non_blocking=True)
                tgt_padding_mask = batch['tgt_padding_mask'].to(device, non_blocking=True)
                tgt_seq_len = decoder_input.size(1)  
                tgt_causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device, non_blocking=True)
                encoder_output = model_core.encode(encoder_input, src_padding_mask)
                decoder_output = model_core.decode(encoder_output, src_padding_mask, decoder_input, tgt_padding_mask, tgt_causal_mask)
            
            proj_output = model_core.project(decoder_output)
            label = batch['label'].to(device, non_blocking=True)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            val_iterator.set_postfix({"val_loss": f"{loss.item():6.3f}"})
            total_val_loss += loss.item()

    if count == 0:
        return float('inf') # Return infinity if no batches
    
    return total_val_loss / count # Return average loss

# --- run_validation_metrics ---
def run_validation_metrics(
    model_core, val_dataloader_subset, 
    tokenizer_src, tokenizer_tgt, config,
    device, print_msg, 
    bleu_metric, chrf_metric,
    num_examples=4
):
    model_core.eval()
    count = 0
    # (This list will store the example translations)
    example_outputs = []

    try:
        console_width = os.get_terminal_size().columns
    except OSError:
        console_width = 80

    with torch.no_grad():
        val_iterator = tqdm(val_dataloader_subset, desc="Validating (Metrics)")
        
        for batch in val_iterator:
            if batch is None: continue
            count += 1 
            encoder_input = batch["encoder_input"].to(device, non_blocking=True)
            
            assert encoder_input.size(0) == 1, "Metric validation batch size must be 1"

            encoder_mask_for_greedy = batch['scratch_encoder_mask'].to(device, non_blocking=True)
            
            model_out_ids = greedy_decode(
                model_core, encoder_input, encoder_mask_for_greedy, 
                config, tokenizer_tgt, config['seq_len_tgt'], device
            )

            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out_ids.detach().cpu().tolist())
            pred_text_list = [model_out_text]
            ref_text_list = [[target_text]]
            
            bleu_metric.update(pred_text_list, ref_text_list)
            chrf_metric.update(pred_text_list, ref_text_list)
            
            if count <= num_examples:
                source_text = batch["src_text"][0]
                example_outputs.append('-'*console_width)
                example_outputs.append(f"{f'SOURCE: ':>20}{source_text}")
                example_outputs.append(f"{f'TARGET: ':>20}{target_text}")
                example_outputs.append(f"{f'PREDICTED: ':>20}{model_out_text}")
    
    # (This prevents them from mixing with the tqdm bar)
    for line in example_outputs:
        print_msg(line)

    if count == 0:
        print_msg("Warning: No valid batches found in metric validation set.")
        bleu_metric.reset()
        chrf_metric.reset()
        return 0.0, 0.0 # Return 0 for BLEU and ChrF

    final_bleu = bleu_metric.compute()
    final_chrf = chrf_metric.compute()
    bleu_metric.reset()
    chrf_metric.reset()
    
    print_msg('-'*console_width)
    
    return final_bleu, final_chrf



def train_model(config):

    print("Loading datasets and tokenizers...")
    train_dataloader, val_dataloader_full, val_dataloader_subset, test_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    src_pad_id = tokenizer_src.token_to_id("[PAD]")
    tgt_pad_id = tokenizer_tgt.token_to_id("[PAD]")
    src_vocab_len = tokenizer_src.get_vocab_size()
    tgt_vocab_len = tokenizer_tgt.get_vocab_size()

    config['src_vocab_size'] = src_vocab_len
    config['tgt_vocab_size'] = tgt_vocab_len

    print(f"Building model: {config['model_type']}")
    model = get_model(
        config, 
        src_vocab_len, 
        tgt_vocab_len, 
        src_pad_id, 
        tgt_pad_id
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device)

    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_pad_id, label_smoothing=0.1).to(device)

    d_model = config['d_model']
    warmup_steps = config['warmup_steps']
    
    def lr_lambda(step_num):
        step_num += 1
        arg1 = step_num ** -0.5
        arg2 = step_num * (warmup_steps ** -1.5)
        return (d_model ** -0.5) * min(arg1, arg2)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    bleu_metric = BLEUScore()
    chrf_metric = CHRFScore()

    initial_epoch = 0
    global_step = 0
    
    # (B) Initialize the two "best" trackers
    best_val_loss = float('inf')
    best_chrf_score = 0.0 
    
    patience = config['patience']
    patience_counter = 0
    chrf_patience_counter = 0

    print("Starting training...")
    for epoch in range(initial_epoch, config['num_epochs']):
        
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:03d}")
        total_train_loss = 0.0
        count_loss = 0
        for batch in batch_iterator:
            if batch is None: continue 
            count_loss += 1
            model_core = model.module if isinstance(model, nn.DataParallel) else model
            
            if config['model_type'] == 'scratch':
                encoder_input = batch['encoder_input'].to(device, non_blocking=True) 
                decoder_input = batch['decoder_input'].to(device, non_blocking=True) 
                encoder_mask = batch['scratch_encoder_mask'].to(device, non_blocking=True) 
                decoder_mask = batch['scratch_decoder_mask'].to(device, non_blocking=True) 
                
                encoder_output = model_core.encode(encoder_input, encoder_mask)
                decoder_output = model_core.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            
            else: # 'pytorch'
                encoder_input = batch['encoder_input'].to(device, non_blocking=True) 
                decoder_input = batch['decoder_input'].to(device, non_blocking=True) 
                src_padding_mask = batch['src_padding_mask'].to(device, non_blocking=True)
                tgt_padding_mask = batch['tgt_padding_mask'].to(device, non_blocking=True)
                tgt_seq_len = decoder_input.size(1) 
                tgt_causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device, non_blocking=True)
                
                encoder_output = model_core.encode(encoder_input, src_padding_mask)
                decoder_output = model_core.decode(encoder_output, src_padding_mask, decoder_input, tgt_padding_mask, tgt_causal_mask)
            
            proj_output = model_core.project(decoder_output)
            
            label = batch['label'].to(device, non_blocking=True)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            
            batch_iterator.set_postfix({"train_loss": f"{loss.item():6.3f}"})

            loss.backward()

             #  Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                model_core.parameters(), 
                config['gradient_clip']
            )
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / count_loss if count_loss > 0 else 0.0

        # --- Run the Two-Part Validation ---
        model_to_validate = model.module if isinstance(model, nn.DataParallel) else model
        
        # (1. Run FAST loss calculation on FULL val set)
        avg_val_loss = run_validation_loss(model_to_validate, val_dataloader_full, loss_fn, config, device, tokenizer_tgt)
        
        # (2. Run SLOW metric calculation on SUBSET val set)
        current_bleu, current_chrf = run_validation_metrics(
                            model_to_validate,
                            val_dataloader_subset,
                            tokenizer_src,
                            tokenizer_tgt,
                            config,
                            device,
                            lambda msg: batch_iterator.write(msg),
                            bleu_metric,
                            chrf_metric
                        )
        
        print(f"--- Epoch {epoch:03d} Summary ---")
        print(f"  Avg Train Loss: {avg_train_loss:6.3f}")
        print(f"  Avg Valid Loss: {avg_val_loss:6.3f}")
        print(f"  Valid BLEU:     {current_bleu.item():6.4f}")
        print(f"  Valid ChrF:     {current_chrf.item():6.4f}")
        print("-------------------------")

        writer.add_scalar('training/loss', avg_train_loss, epoch)
        writer.add_scalar('validation/loss', avg_val_loss, epoch)
        writer.add_scalar('validation/bleu', current_bleu, epoch)
        writer.add_scalar('validation/chrf', current_chrf, epoch)
        

        # ---  Dual Saving and Early Stopping Logic ---
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0 

            # Save the "best loss" model
            model_filename = get_weights_file_path(config, "best_loss") 
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_validate.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'best_val_loss': best_val_loss
            }, str(model_filename))
            print(f"Validation loss improved. Saving 'best_loss' model.")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Best is {best_val_loss:.4f}. Patience: {patience_counter}/{patience}")


        
        if isinstance(current_chrf, torch.Tensor):
            current_chrf_val = current_chrf.item()
        else:
            current_chrf_val = current_chrf

        if current_chrf_val > best_chrf_score:
            best_chrf_score = current_chrf_val
            chrf_patience_counter = 0

            # Save the "best quality" model
            model_filename = get_weights_file_path(config, "best_chrf")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_validate.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'best_chrf_score': best_chrf_score
            }, str(model_filename))
            print(f"Validation ChrF score improved. Saving 'best_chrf' model.")

        else:
            chrf_patience_counter += 1
            print(f"Validation ChrF score did not improve. Best is {best_chrf_score:.4f}. Patience: {chrf_patience_counter}/{patience}")


        # Trigger Early Stopping based on BOTH loss and ChrF patience
        if patience_counter >= patience and chrf_patience_counter >= patience:
            print(f"Early stopping triggered: Both loss and ChrF did not improve for {patience} epochs.")
            break
        elif patience_counter >= patience:
            print(f"Warning: Loss patience exhausted ({patience_counter}/{patience}), but ChrF still improving. Continuing...")
        elif chrf_patience_counter >= patience:
            print(f"Warning: ChrF patience exhausted ({chrf_patience_counter}/{patience}), but loss still improving. Continuing...")
                    
    writer.close()