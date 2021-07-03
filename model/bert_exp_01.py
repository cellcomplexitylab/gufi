import torch
import torch.nn as nn
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1, auroc

criterion = nn.BCELoss()


class TitleClassifier(pl.LightningModule):
    def __init__(self, n_classes: int = 5, steps_per_epoch: int = None, n_epochs: int = None,
                 BERT_MODEL_NAME: str = "bert-base-uncased",
                 LABEL_COLUMNS: list = ["math", "stat", "physics", "q-bio", "q-fin"]):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)  # hidden size for bert is 768
        self.LABEL_COLUMNS = LABEL_COLUMNS
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        '''
          Compute loss in the forward step if labels are given
          (Standard in fine-tuning bert)
        '''
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)

        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        self.labels = batch["labels"]
        loss, self.outputs = self(input_ids, attention_mask, self.labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {"test_loss": loss, "test_predictions": self.outputs, "test_labels": self.labels}

    def training_epoch_end(self, outputs):
        '''
          Computing ROC-AUC score for the prediction at the end of training
        '''
        labels = []
        predictions = []
        for output in outputs:
            # detach from CPU because we are training using GPU
            for output_labels in output["labels"].detach().cpu():
                labels.append(output_labels)

            for output_predictions in output["predictions"].detach().cpu():
                predictions.append(output_predictions)

        labels = torch.stack(labels).detach().cpu()
        predictions = torch.stack(predictions).detach().cpu()
        # per class AUC-ROC score because we are doing multilabel classification
        # its like a binary classification for all the classes
        for i, name in enumerate(self.LABEL_COLUMNS):
            roc_score = auroc(preds=predictions[:, i], target=labels[:, i].type('torch.IntTensor'))
            # adding results to tensorboard
            self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", roc_score, self.current_epoch)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)  # original recommended fine-tune lr from bert-paper

        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )

    def predict(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch)