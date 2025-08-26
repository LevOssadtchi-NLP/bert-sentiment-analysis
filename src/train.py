import os
import torch
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW 
from utils import load_config
from data_loader import create_data_loaders
from model import get_model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def train_model():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, _ = create_data_loaders(config)
    train_loader = train_loader
    model = get_model(config)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=float(config['training']['learning_rate']))
    total_steps = len(train_loader) * config['training']['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    for epoch in range(config['training']['epochs']):
        print(f'Epoch {epoch + 1}/{config["training"]["epochs"]}')
        model.train()
        for d in tqdm(train_loader):
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Evaluation loop
        model.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():
            for d in val_loader:
                input_ids = d['input_ids'].to(device)
                attention_mask = d['attention_mask'].to(device)
                labels = d['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                _, preds = torch.max(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(true_labels, predictions)
        print(f'Validation Accuracy: {accuracy:.4f}')

    # Save the model
    output_dir = config['paths']['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == '__main__':
    train_model()
