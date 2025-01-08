import torch
from torch.utils.data import Dataset


class AICodeDataset(Dataset):
    def __init__(self, input_data, labels, tokenizer, max_len=512):
        self.input_data = input_data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_sample = self.input_data[idx]
        label = self.labels[idx]

        inputs = self.tokenizer(
            input_sample,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
        }
