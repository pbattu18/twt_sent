import torch
import torch.nn as nn
from transformers import RobertaModel

class TweetSentimentExtractor(nn.Module):
    def __init__(self):
        super(TweetSentimentExtractor, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.1)
        self.span_fc = nn.Linear(self.roberta.config.hidden_size, 2)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.last_hidden_state[:, 0]

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        span_logits = self.span_fc(sequence_output)
        start_logits, end_logits = span_logits.split(1, dim=2)
        sentiment_logits = self.classifier(pooled_output)

        return start_logits.squeeze(-1), end_logits.squeeze(-1), sentiment_logits
