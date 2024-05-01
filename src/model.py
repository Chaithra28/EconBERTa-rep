import torch
from src.config import dropout_rate
from torchcrf import CRF
from transformers import AutoModel

class CRFTagger(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        # Mask should be of type 'bool' in newer PyTorch versions
        mask = attention_mask.type(torch.bool) if hasattr(torch, 'bool') else attention_mask.byte()
        
        if labels is not None:
            loss = -self.crf(logits, labels, mask=mask, reduction='mean')
            return {'loss': loss, 'logits': logits, 'decoded': self.crf.decode(logits, mask=mask)}
        else:
            decoded_labels = self.crf.decode(logits, mask=mask)
            return {'decoded': decoded_labels, 'logits': logits}
        