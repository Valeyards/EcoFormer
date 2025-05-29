import torch
import numpy as np
import os
from tqdm import tqdm
import wandb
from datasets import create_data_loaders, create_external_data_loader
from models import get_model
from train import validate, calculate_metrics
import json
from pathlib import Path
import pandas as pd
import argparse

def plot_roc_curve_and_confusion_matrix(y_true, y_prob, save_dir, dataset_name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, auc, confusion_matrix

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_dir / f'roc_curve_{dataset_name}.png')
    plt.close()
    
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_dir / f'confusion_matrix_{dataset_name}.png')
    plt.close()    

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    sample_loader = create_data_loaders(
        root_dir=config['data']['root_dir'],
        data_dir=config['data']['data_dir'],
        batch_size=1,
        num_workers=0,
        fold=config['data']['fold'],
        eval=True
    )
    sample_batch = next(iter(sample_loader['train']))
    input_dims = {rna_type: data.shape[1] 
                 for rna_type, data in sample_batch['data'].items()}
    
    model = get_model(
        model_name=config['model']['type'],
        input_dims=input_dims,
        **config['model']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config

def evaluate(model, loader, device, dataset_name="test"):
    model.eval()
    predictions = []
    probabilities = []
    labels = []
    sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Evaluating on {dataset_name}'):
            data = {k: v.to(device) for k, v in batch['data'].items()}
            target = batch['label'].to(device)
            
            output = model(data)
            probs = torch.softmax(output, dim=1)[:, 1]
            
            predictions.extend(output.argmax(dim=1).cpu().numpy().astype(int).tolist())
            probabilities.extend(probs.cpu().numpy().astype(float).tolist())
            labels.extend(target.cpu().numpy().astype(int).tolist())
            sample_ids.extend(batch['sample_id'])
    
    metrics = calculate_metrics(
        y_true=np.array(labels),
        y_pred=np.array(predictions),
        y_prob=np.array(probabilities)
    )
    metrics = {k: float(v) for k, v in metrics.items()}
    return metrics, predictions, probabilities, sample_ids

def main(model_path, external_data_config=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model, config = load_model(model_path, device)
    print("Model loaded successfully")
    
    save_dir = Path(model_path).parent / 'evaluation'
    save_dir.mkdir(exist_ok=True)
    
    val_loaders = create_data_loaders(
        root_dir=config['data']['root_dir'],
        data_dir=config['data']['data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        eval=True
    )
    
    train_metrics, train_preds, train_probs, train_ids = evaluate(model, val_loaders['train'], device, "train")
    train_results_df = pd.DataFrame({
        'sample_id': train_ids,
        'true_label': val_loaders['train'].dataset.labels,
        'predicted_label': train_preds,
        'probability': train_probs,
        'center': 'train'
    })
    train_results_df.to_csv(save_dir / 'train_results.csv', index=False)
    plot_roc_curve_and_confusion_matrix(
        y_true=np.array(val_loaders['train'].dataset.labels),
        y_prob=np.array(train_probs),
        save_dir=save_dir,
        dataset_name="train"
    )
    print("\nTrain Metrics:")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.4f}")

    val_metrics, val_preds, val_probs, val_ids = evaluate(model, val_loaders['val'], device, "validation")
    val_results_df = pd.DataFrame({
        'sample_id': val_ids,
        'true_label': val_loaders['val'].dataset.labels,
        'predicted_label': val_preds,
        'probability': val_probs,
        'center': 'val'
    })
    val_results_df.to_csv(save_dir / 'validation_results.csv', index=False)
    plot_roc_curve_and_confusion_matrix(
        y_true=np.array(val_loaders['val'].dataset.labels),
        y_prob=np.array(val_probs),
        save_dir=save_dir,
        dataset_name="validation"
    )
    print("\nValidation Metrics:")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")
    
    if external_data_config:
        external_loader = create_external_data_loader(
            csv_file=external_data_config['csv_file'],
            data_dir=external_data_config['data_dir'],
            batch_size=config['training']['batch_size'],
            num_workers=config['data']['num_workers']
        )
        
        ext_metrics, ext_preds, ext_probs, ext_ids = evaluate(
            model, external_loader, device, "external"
        )
        ext_results_df = pd.DataFrame({
            'sample_id': ext_ids,
            'true_label': external_loader.dataset.labels,
            'predicted_label': ext_preds,
            'probability': ext_probs,
            'center': 'external'
        })
        ext_results_df.to_csv(save_dir / 'external_results.csv', index=False)
        plot_roc_curve_and_confusion_matrix(
            y_true=np.array(external_loader.dataset.labels),
            y_prob=np.array(ext_probs),
            save_dir=save_dir,
            dataset_name="external"
        )
        print("\nExternal Dataset Metrics:")
        for k, v in ext_metrics.items():
            print(f"{k}: {v:.4f}")
    
    results = {
        'validation_metrics': val_metrics,
        'validation_predictions': val_preds,
        'validation_probabilities': val_probs,
    }
    
    if external_data_config:
        results.update({
            'external_metrics': ext_metrics,
            'external_predictions': ext_preds,
            'external_probabilities': ext_probs,
        })
    
    with open(save_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='results/try1/best_model.pth')
    parser.add_argument('--external_eval', type=bool, default=False)
    args = parser.parse_args()

    if args.external_eval:
        external_data_config = {
            'csv_file': 'data/external_samples.csv',
            'data_dir': 'data/expression_npy'
        }
    else:
        external_data_config = None
    main(args.model_path, external_data_config)