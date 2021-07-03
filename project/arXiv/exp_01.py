import argparse
import math
import matplotlib.pyplot as plt
from model.bert_exp_01 import TitleClassifier
import numpy as np
import os
import os.path as osp
import pandas as pd
from preprocess.tokenizer.bert_tokenizer import TitleCategoryDataset
from preprocess.bert_flow import BertFlow, TitleCategoryModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import hamming_loss, accuracy_score, f1_score
from transformers import BertTokenizerFast as BertTokenizer
import torch
from tqdm import tqdm


class Exp01:
    def __init__(self, data_loc, model_loc):
        self.data_loc = data_loc
        self.model_loc = model_loc

        self.new_data_loc = osp.join(data_loc, 'clean')
        if not osp.exists(self.new_data_loc):
            os.mkdir(self.new_data_loc)

        # default x column and y columns
        self.major_topics = ['math', 'stat', 'physics', 'q-bio', 'q-fin']
        self.x_col = "title"
        self.tokenizer_name = "bert-base-uncased"
        self.N_EPOCHS = 1
        self.BATCH_SIZE = 1
        self.P_threshold = 0.5
        self.GPUs = 0

    def split_df(self):
        flow = BertFlow(self.data_loc, x_col=self.x_col, label_columns=self.major_topics)
        self.train_df, self.val_df, self.test_df = flow.return_split()

    def load_dataset(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        self.train_dataset = TitleCategoryDataset(self.train_df, tokenizer=self.tokenizer)
        self.val_dataset = TitleCategoryDataset(self.val_df, tokenizer=self.tokenizer)
        self.test_dataset = TitleCategoryDataset(self.test_df, tokenizer=self.tokenizer)

    def load_module(self):
        self.data_module = TitleCategoryModule(self.train_df, self.val_df, self.test_df,
                                               tokenizer=self.tokenizer, batch_size=self.BATCH_SIZE)
        self.data_module.setup()

    def initiate_clsasifier(self):
        self.model = TitleClassifier(
            n_classes=len(self.major_topics),
            steps_per_epoch=len(self.train_df) // self.BATCH_SIZE,
            n_epochs=self.N_EPOCHS, )

    def train(self):
        try:
            resume_from_checkpoint = osp.join(self.model_loc, "last.ckpt")
            self.trainer = pl.Trainer(default_root_dir=self.model_loc,
                                      resume_from_checkpoint=resume_from_checkpoint,
                                      max_epochs=self.N_EPOCHS, gpus=self.GPUs, progress_bar_refresh_rate=30,
                                      callbacks=[EarlyStopping(monitor='val_loss', patience=3),
                                                 ModelCheckpoint(dirpath=osp.join(self.model_loc, "bert.ckpt"),
                                                                 filename="bert", monitor="val_loss", mode="min",
                                                                 save_last=True, period=1, save_top_k=1)])
            self.trainer.fit(self.model, self.data_module)

        except:
            self.trainer = pl.Trainer(default_root_dir=self.model_loc,
                                      max_epochs=self.N_EPOCHS, gpus=self.GPUs, progress_bar_refresh_rate=30,
                                      callbacks=[EarlyStopping(monitor='val_loss', patience=3),
                                                 ModelCheckpoint(dirpath=osp.join(self.model_loc, "bert.ckpt"),
                                                                 filename="bert", monitor="val_loss", mode="min",
                                                                 save_last=True, period=1, save_top_k=1)])
            self.trainer.fit(self.model, self.data_module)

    def test(self):
        max_token_len = 40
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trained_model = TitleClassifier.load_from_checkpoint(
            self.trainer.checkpoint_callback.best_model_path,
            n_classes=len(self.major_topics))
        trained_model.eval()
        trained_model.freeze()
        trained_model = trained_model.to(device)

        predictions = []
        labels = []

        test_dataset = TitleCategoryDataset(
            self.test_df,
            tokenizer=self.tokenizer,
            max_token_len=max_token_len
        )

        for i, item in enumerate( tqdm(test_dataset) ):
            if i < len(self.test_df):
                _, prediction = trained_model(
                    item["input_ids"].unsqueeze(dim=0).to(device),
                    item["attention_mask"].unsqueeze(dim=0).to(device)
                )
                predictions.append(prediction.flatten())
                labels.append(item["labels"].int())
            else:
                break

        predictions = torch.stack(predictions).detach().cpu()
        labels = torch.stack(labels).detach().cpu()
        one_hot_predictions = np.array([(p.numpy() > self.P_threshold).astype('int32') for p in predictions])
        one_hot_labels = labels.numpy()
        accuracy = (one_hot_labels == one_hot_predictions).sum() / (len(one_hot_labels) * len(one_hot_labels[0]))
        print(f'Binary Accruacy of the model is: {accuracy * 100}%')
        hamming_loss_ = hamming_loss(one_hot_labels, one_hot_predictions)
        f1_macro = f1_score(one_hot_labels, one_hot_predictions, average='macro')
        f1_weighted = f1_score(one_hot_labels, one_hot_predictions, average='weighted')
        f1_scores = f1_score(one_hot_labels, one_hot_predictions, average=None)
        print(f'Hamming loss of the model is {hamming_loss_}')
        print(f'F1-macro score of the model is {f1_macro}')
        print(f'F1-weighted score of the model is {f1_weighted}')
        print(f"Per class F1-score of the model is {[f'{class_}: {score}' for class_, score in zip(self.major_topics, f1_scores)]}")
        multilabel_CMs = multilabel_confusion_matrix(one_hot_labels, one_hot_predictions)
        norm_multilabel_CMs = np.array([(CM.astype('float') / CM.sum(axis=1)[:, np.newaxis]) for CM in multilabel_CMs])
        fig = plt.figure(figsize=(15, 15))
        rows = int(math.ceil(len(self.major_topics) / 2))
        for i, CM in enumerate(multilabel_CMs):
            axes = fig.add_subplot(rows, 2, i + 1)
            df_cm = pd.DataFrame(CM, index=[i for i in [1, 0]],
                                 columns=[i for i in [1, 0]])
            sns.heatmap(df_cm, annot=True, ax=axes, fmt='d')
            axes.set_title(f'Confusion Matrix for class {self.major_topics[i]}')
        fig.suptitle('{} binary accuracy score: {:.2f}%'.format('BERT', accuracy * 100))
        plt.savefig(osp.join(self.model_loc, "exp_01_multilabel_CM.png"), dpi=100)
        plt.show()
        plt.close()

        fig = plt.figure(figsize=(15, 15))
        for i, CM in enumerate(norm_multilabel_CMs):
            axes = fig.add_subplot(rows, 2, i + 1)
            df_cm = pd.DataFrame(CM, index=[i for i in [1, 0]],
                                 columns=[i for i in [1, 0]])
            sns.heatmap(df_cm, annot=True, ax=axes)
            axes.set_title(f'Confusion Matrix for class {self.major_topics[i]}')
        fig.suptitle('{} binary accuracy score: {:.2f}%'.format('BERT-normalized', accuracy * 100))
        plt.savefig(osp.join(self.model_loc, "exp_01_normalized_multilabel_CM.png"), dpi=100)
        plt.show()
        plt.close()


def arg_parse():
    parser = argparse.ArgumentParser(description='Input Variables')
    parser.add_argument(
        "--data_loc", "-i",
        required=True,
        help="data location"
    )
    parser.add_argument(
        "--model_loc", "-o",
        required=False,
        help="directory to save the output in"
    )
    args = parser.parse_args()
    return (
        args.data_loc,
        args.model_loc,
    )


if __name__ == "__main__":
    data_loc, model_loc = arg_parse()

    # exp
    exp = Exp01(data_loc, model_loc)
    exp.split_df()
    exp.load_dataset()
    exp.load_module()
    exp.initiate_clsasifier()
    exp.train()
    exp.test()
