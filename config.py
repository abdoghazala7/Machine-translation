def get_scratch_config():
    return {
        "model_type": "scratch",
        
        "experiment_name": "runs/Transformer_Scratch",
        "model_folder": "weights/Transformer_Scratch",
        
        'batch_size': 256,
        "num_epochs": 50,   
        "lr": 1.0,
        "d_model": 512,
        "N": 6,
        "h": 8,
        "dropout": 0.3,
        "d_ff": 2048,
        "datasource": "ymoslem/CoVoST2-EN-AR-Text",
        "lang_src": "text_en",
        "lang_tgt": "text_ar",
        "tokenizer_file": "tokenizer_{0}.json",
        "model_basename": "Tmodel_",
        "warmup_steps": 4000,
        "gradient_clip": 1.0,
        "patience": 7,

        # How many samples to use for (slow) metric calculation
        # Set to -1 to use the full validation set
        "validation_subset_size": 1000
    }

def get_pytorch_config():
    config = get_scratch_config() 
    
    config["model_type"] = "pytorch"
    config["experiment_name"] = "runs/Transformer_PyTorch"
    config["model_folder"] = "weights/Transformer_PyTorch"
    
    return config