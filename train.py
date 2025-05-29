import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
import os
from tqdm import tqdm
import time
from datasets import create_data_loaders
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from models import get_model

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    predictions = []
    probabilities = []
    labels = []
    
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        data = {k: v.to(device) for k, v in batch['data'].items()}
        target = batch['label'].to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        probs = torch.softmax(output, dim=1)[:, 1] 
        predictions.extend(output.argmax(dim=1).cpu().numpy())
        probabilities.extend(probs.detach().cpu().numpy())
        labels.extend(target.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    metrics = calculate_metrics(
        y_true=np.array(labels), 
        y_pred=np.array(predictions),
        y_prob=np.array(probabilities)
    )
    metrics['loss'] = total_loss / len(loader)
    
    return metrics

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    probabilities = [] 
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating'):
            data = {k: v.to(device) for k, v in batch['data'].items()}
            target = batch['label'].to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            probs = torch.softmax(output, dim=1)[:, 1]
            predictions.extend(output.argmax(dim=1).cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            labels.extend(target.cpu().numpy())
    
    metrics = calculate_metrics(
        y_true=np.array(labels), 
        y_pred=np.array(predictions),
        y_prob=np.array(probabilities)
    )
    metrics['loss'] = total_loss / len(loader)
    
    return metrics

def calculate_metrics(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    try:
        auc = roc_auc_score(y_true, y_prob)
        metrics['auc'] = auc
    except:
        metrics['auc'] = 0
    
    return metrics

def train(config):
    run = wandb.init(
        project="RNA-Classification",
        name=f"try1",
        config=config
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    loaders = create_data_loaders(
        root_dir=config['data']['root_dir'],
        data_dir=config['data']['data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    
    sample_batch = next(iter(loaders['train']))
    input_dims = {rna_type: data.shape[1] 
                 for rna_type, data in sample_batch['data'].items()}
    model = get_model(
        model_name=config['model']['type'],
        input_dims=input_dims,
        **config['model'] 
    ).to(device)
    
    wandb.watch(model, log="all")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    
    best_val_auc = 0
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        train_metrics = train_epoch(model, loaders['train'], criterion, optimizer, device)
        
        val_metrics = validate(model, loaders['val'], criterion, device)

        wandb.log({
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
            "epoch": epoch
        })
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}, AUC: {train_metrics.get('auc', 0):.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics.get('auc', 0):.4f}")
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            model_path = os.path.join(config['save_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'config': config
            }, model_path)
            wandb.run.summary["best_val_auc"] = best_val_auc
            wandb.run.summary["best_epoch"] = epoch
    
    wandb.finish()

if __name__ == '__main__':
    base_config = {
        'data': {
            'root_dir': 'data/splits',
            'data_dir': 'data/expression_npy',
            'num_workers': 12
        },
        'model': {
            'type': 'ecoformer',
            'hidden_dim': 512,
            'num_heads': 8,
            'num_layers': 2,
            'dropout': 0.1
        },
        'training': {
            'epochs': 30,
            'batch_size': 32,
            'lr': 0.0002
        },
        'save_dir': 'results/try1'
    }
    import copy
    print(f"\n{'='*50}")
    print(f"{'='*50}\n")
    config = copy.deepcopy(base_config)
    os.makedirs(config['save_dir'], exist_ok=True)
    train(config)
