from torch import nn
from transformers import BertForSequenceClassification


class CustomBERTModel(nn.Module):
    def __init__(
            self,
            dropout_prob,
            bert_model_name="bert-base-uncased",
            hidden_size=768,
            intermediate_dim=256,
        ):

        super(CustomBERTModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=1)
        self.fc1 = nn.Linear(hidden_size, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, 1)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_state = outputs.hidden_states[-1][:, 0, :]
        x = self.fc1(hidden_state)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
