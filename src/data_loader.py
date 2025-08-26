import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loaders(config):
    """
    Loads and prepares data for training and evaluation.
    This function assumes train.csv, val.csv, and test.csv exist.
    """
    from torch import manual_seed
    import torch

    manual_seed(42)
    
    tokenizer = BertTokenizer.from_pretrained(config['model']['name'])
    
    # Placeholder data creation (replace with your actual data loading)
    train_df = pd.read_csv(config['data']['train_path'])
    val_df = pd.read_csv(config['data']['val_path'])
    test_df = pd.read_csv(config['data']['test_path'])
    
    train_dataset = SentimentDataset(train_df.text.values, train_df.label.values, tokenizer, config['training']['max_len'])
    val_dataset = SentimentDataset(val_df.text.values, val_df.label.values, tokenizer, config['training']['max_len'])
    test_dataset = SentimentDataset(test_df.text.values, test_df.label.values, tokenizer, config['training']['max_len'])

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'])

    return train_loader, val_loader, test_loader
