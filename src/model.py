from transformers import BertForSequenceClassification, BertTokenizer
from torch import nn

def get_model(config):
    """
    Initializes and returns the pre-trained BERT model.
    """
    model = BertForSequenceClassification.from_pretrained(
        config['model']['name'],
        num_labels=config['model']['num_labels'],
        output_attentions=False,
        output_hidden_states=False,
    )
    return model
