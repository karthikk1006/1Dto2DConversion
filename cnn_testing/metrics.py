import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def calculate_metrics(y_true, y_pred, num_classes):
    avg_setting = "binary" if num_classes == 2 else "macro"
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg_setting, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg_setting, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg_setting, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    return metrics

def save_metrics(metrics, output_path):
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()

def plot_training_curves(train_history, val_history, metric_name, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label=f'Train {metric_name}')
    plt.plot(val_history, label=f'Val {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Training and Validation {metric_name}')
    plt.legend()
    plt.savefig(output_path)
    plt.close()
