import torch
from transformers import BertForSequenceClassification, BertTokenizer
from utils import load_config
from data_loader import create_data_loaders
from sklearn.metrics import accuracy_score

def predict_sentiment(text, model, tokenizer, device, max_len):
    """
    Predicts the sentiment of a single text input.
    """
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(output.logits, dim=1)
    
    return ["Negative", "Positive"][prediction.item()]

def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(config['model']['name'])
    model = BertForSequenceClassification.from_pretrained(config['paths']['output_dir'])
    model.to(device)
    model.eval()

    # Get test data loader
    _, _, test_loader = create_data_loaders(config)
    
    # Example prediction on a single sentence
    example_text = "This is an amazing product, I love it!"
    sentiment = predict_sentiment(example_text, model, tokenizer, device, config['training']['max_len'])
    print(f"Text: '{example_text}' -> Sentiment: {sentiment}")

    predictions = []
    true_labels = []
    with torch.no_grad():
        for d in test_loader:
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
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()
