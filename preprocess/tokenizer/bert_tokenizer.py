import pandas as pd
from transformers import BertTokenizerFast as BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

class TitleCategoryDataset(Dataset):
  def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, max_token_len: int = 40,
               x_col: str = "title", major_topics: list = ["math", "stat", "physics", "q-bio", "q-fin"]):
    self.data = data
    self.max_token_len = max_token_len
    self.x_col = x_col
    self.label_classes = major_topics
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]
    title = data_row[self.x_col]
    labels = data_row[self.label_classes].values.tolist()

    encoding = self.tokenizer.encode_plus(
        title,
        add_special_tokens=True,
        max_length=self.max_token_len,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True, # make sure that each sequence is of max_token_len
        return_attention_mask=True,
        return_tensors="pt" #to return tensors like pytorch
    )

    return dict(
        title=title,
        input_ids=encoding['input_ids'].flatten(),
        attention_mask=encoding['attention_mask'].flatten(),
        labels=torch.FloatTensor(labels) # required by the loss function
    )

  def return_tokenizer(self):
      return self.tokenizer