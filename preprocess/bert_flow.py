import os
import os.path as osp
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess.tokenizer.bert_tokenizer import TitleCategoryDataset
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast as BertTokenizer


class TitleCategoryModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, tokenizer: BertTokenizer,
                 batch_size=16, max_token_len=40):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.train_dataset = TitleCategoryDataset(
            self.train_df,
            self.tokenizer,
            self.max_token_len
        )

        self.val_dataset = TitleCategoryDataset(
            self.val_df,
            self.tokenizer,
            self.max_token_len
        )

        self.test_dataset = TitleCategoryDataset(
            self.test_df,
            self.tokenizer,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2  # feed more than one batch at a time
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2  # feed more than one batch at a time
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2  # feed more than one batch at a time
        )

class BertFlow:
    def __init__(self, data_loc, split: float = 0.2,
                 x_col: str = 'title',
                 label_columns: list = ['math', 'stat', 'physics', 'q-bio', 'q-fin']):
        self.RANDOM_SEED = 2021
        self.data_loc = data_loc
        self.label_cols = label_columns
        self.x_col = x_col
        # (test and val-size this split is then split into half)
        self.split = split
        self.df = self.read_csv()
        self.train_df, self.val_df, self.test_df = self.split_df()

    def read_csv(self):
        csv_s = [f_ for f_ in os.listdir(self.data_loc) if 'csv' in f_]
        df = pd.read_csv(osp.join(self.data_loc, csv_s[0]))
        return df

    def split_df(self):
        train_df, val_df = train_test_split(self.df, test_size=self.split, shuffle=True,
                                            random_state=self.RANDOM_SEED)
        val_df, test_df = train_test_split(val_df, test_size=0.5, shuffle=True,
                                           random_state=self.RANDOM_SEED)
        print(f'Number of training samples: {len(train_df)}')
        print(f'Number of validation samples: {len(val_df)}')
        print(f'Number of test samples: {len(test_df)}')
        return train_df, val_df, test_df

    def return_split(self):
        return self.train_df, self.val_df, self.test_df