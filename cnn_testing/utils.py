import os
import random
import numpy as np
import torch
import logging

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    handler.setFormatter(formatter)
    
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console)
    logger.propagate = False
    
    return logger

def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pt'):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(output_dir, 'best_model.pt')
        torch.save(state, best_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint.get('epoch', 0), checkpoint.get('best_acc', 0.0)
