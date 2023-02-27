import pytorch_lightning as pl
import torch
from transformers import AdamW
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration


class DualAEModel(pl.LightningModule):
    def __init__(self, token_embeddings_size, cache_dir_base: str, model_name: str):
        super().__init__()
        self.kgqt_model = MT5ForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=f"{cache_dir_base}/{model_name}",
            return_dict=True)
        # 先扩容到应有的大小
        self.kgqt_model.resize_token_embeddings(token_embeddings_size)
        self.kgqg_model = MT5ForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=f"{cache_dir_base}/{model_name}",
            return_dict=True)
        self.kgqg_model.resize_token_embeddings(token_embeddings_size)

    def forward(self, input_ids, attention_mask, labels=None):
        with torch.autograd.set_grad_enabled(True):
            # 理论上，根据不同的输入，共有三种计算形式。
            # MODE 1: NLQ1 -> TSQ1 -> NLQ2 (NLQ-AutoEncoder)
            # MODE 2: TSQ1 -> NLQ2 -> TSQ2 (TSQ-AutoEncoder)
            # MODE 3: NLQ1 -> TSQ1 -> NLQ2 -> TSQ2 (Throughout the Information Flow) 实装
            kgqt_output1 = self.kgqt_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = kgqt_output1.loss

            kgqg_outputs1 = self.kgqg_model(
                input_ids=kgqt_output1.logits.argmax(-1),
                labels=input_ids
            )

            loss += kgqg_outputs1.loss

            # 第三步，再次kgqt
            kgqt_output2 = self.kgqt_model(
                input_ids=kgqg_outputs1.logits.argmax(-1),
                attention_mask=attention_mask,
                labels=labels
            )

            loss += kgqt_output2.loss

        return loss, kgqt_output2.logits

    def predict_KGQT(self, input_ids, attention_mask):
        # NLQ -> TSQ
        return self.kgqt_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=64
        )

    def predict_KGQG(self, input_ids, attention_mask):
        # TSQ -> NLQ
        return self.kgqg_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=64,
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-3)
