from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 15,
        "lr": 10**-4,
        "seq_len": 50,
        "d_model": 512,
        "datasource": "ymoslem/CoVoST2-EN-AR-Text",
        "lang_src": "text_en",
        "lang_tgt": "text_ar",
        "model_folder": "weights",
        "model_basename": "Tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/Tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = Path(config['model_folder'])  # Convert to Path
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = Path(config['model_folder'])  
    model_filename = f"{config['model_basename']}*"  
    weights_files = list(model_folder.glob(model_filename))
    if not weights_files:  
        print("No weights files found in the folder.")
        return None
    weights_files.sort()  
    return str(weights_files[-1])  
