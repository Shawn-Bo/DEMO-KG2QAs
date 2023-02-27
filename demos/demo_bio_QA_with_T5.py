"""
    本历程展示了使用 pytorch_lightning 和 transformers 实现 T5 模型预训练的过程。
"""

import json
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer


# Data
# 预览
# with Path("BioASQ/BioASQ-train-factoid-4b.json").open() as json_file:
#     data = json.load(json_file)
#     # print(data.keys())
#     questions = data["data"][0]["paragraphs"]
#     # print(questions[0])


def extract_questions_and_answers(fact_oid_path: Path):
    with fact_oid_path.open() as json_file:
        data = json.load(json_file)
    questions = data["data"][0]["paragraphs"]
    data_rows = []
    for question in questions:
        context = question["context"]
        for q_a in question["qas"]:
            question = q_a["question"]
            answers = q_a["answers"]
            for ans in answers:
                ans_text = ans["text"]
                ans_start = ans["answer_start"]
                ans_end = ans_start + len(ans_text)

                data_rows.append({
                    "question": question,
                    "context": context,
                    "answer_text": ans_text,
                    "answer_start": ans_start,
                    "answer_end": ans_end
                })
    return pd.DataFrame(data_rows)


class BioQADataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: T5Tokenizer,
            source_max_token_len: int = 396,
            target_max_token_len: int = 32
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        source_encoding = self.tokenizer(
            data_row["question"],
            data_row["context"],
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation="only_second",
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            data_row["answer_text"],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        labels = target_encoding["input_ids"]
        labels[labels == 0] = -100  # todo 为什么输入不用？
        return dict(
            question=data_row["question"],
            context=data_row["context"],
            answer_text=data_row["answer_text"],
            input_ids=source_encoding["input_ids"].flatten(),
            attention_mask=source_encoding["attention_mask"].flatten(),
            labels=labels.flatten()
        )


class BioQADatasetMoudle(pl.LightningDataModule):
    def __init__(
            self,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame,
            tokenizer: T5Tokenizer,
            batch_size: int = 8,
            source_max_token_len: int = 396,
            target_max_token_len: int = 32
    ):
        super().__init__()
        self.batch_size = batch_size
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.tokenizer = tokenizer
        self.train_df = train_df
        self.test_df = test_df

    def setup(self):
        self.train_dataset = BioQADataset(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

        self.test_dataset = BioQADataset(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=1
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=4
        )


# 尝试使用T5


# model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)

# input_ids = tokenizer(
#     "translate English to German: I talk a lot, so I've learned to tune myself out.",
#     return_tensors="pt"
#     ).input_ids

# generated_ids = model.generate(input_ids=input_ids)

# preds = [
#     tokenizer.decode(gen_id)
#     for gen_id in generated_ids
# ]

# print("".join(preds))

# 定义一波模型
class BioQAModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            cache_dir=f"E:\\LaBarn\\checkpoints\\{MODEL_NAME}",
            return_dict=True)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output.loss, output.logits

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
        return AdamW(self.parameters(), lr=0.0001)


def generate_answer(question):
    source_encoding = tokenizer(
        question["question"],
        question["context"],
        max_length=396,
        padding="max_length",
        truncation="only_second",
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    generated_ids = trained_model.kgqt_model.generate(
        input_ids=source_encoding["input_ids"],
        attention_mask=source_encoding["attention_mask"],
        num_beams=1,
        max_length=80,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True
    )

    preds = [
        tokenizer.decode(
            generated_id,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        for generated_id in generated_ids
    ]

    return "".join(preds)


if __name__ == "__main__":

    pl.seed_everything(seed=42)

    # print(extract_questions_and_answers(Path("BioASQ/BioASQ-train-factoid-4b.json")).head(5))

    factoid_paths = sorted(list(Path("E:\\LaBarn\\datasets\\BioBERT\\BioASQ").glob("BioASQ-train-*")))

    df = pd.concat([extract_questions_and_answers(path) for path in factoid_paths])
    df = df.drop_duplicates(subset=["context"]).reset_index(drop=True)
    sample_question = df.iloc[32]

    # print(df)
    # sample_dataset = BioQADataset(df, tokenizer)

    # for data in sample_dataset:
    #     print(data["question"])
    #     print(data["answer_text"])
    #     print(data["input_ids"][:10])
    #     print(data["labels"][:10])
    #     exit()
    # print("daw")

    train_df, val_df = train_test_split(df, test_size=0.05)

    # print(df.head(5))
    # print(df.shape)
    # print(len(df["context"].unique()))
    # print(df.iloc[0])
    # print(df.iloc[[0]])

    MODEL_NAME = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(
        MODEL_NAME,
        model_max_length=512,
        cache_dir=f"E:\\LaBarn\\checkpoints\\{MODEL_NAME}")

    question_encoding = tokenizer(
        sample_question["question"],
        sample_question["context"],
        max_length=396,
        padding="max_length",
        truncation="only_second",
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    # print(tokenizer.decode(encoding["input_ids"].squeeze()))
    # print(tokenizer.special_tokens_map)
    # print(tokenizer.eos_token_id)

    # 准备标签(答案)
    answer_encoding = tokenizer(
        sample_question["answer_text"],
        max_length=32,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    # print(tokenizer.decode(answer_encoding["input_ids"].squeeze()))

    # T5模型不会忽略id=0，文档要求设置为-100
    # 但是我们这个就不一定了！！！

    labels = answer_encoding["input_ids"]
    labels[labels == 0] = -100

    BATCH_SIZE = 1
    N_EPOCHS = 5

    data_module = BioQADatasetMoudle(train_df, val_df, tokenizer, batch_size=BATCH_SIZE)
    data_module.setup()

    model = BioQAModel()

    # 开始训练
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    logger = TensorBoardLogger("training-logs", name="bio-qa")

    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=N_EPOCHS,
        gpus=1,
        progress_bar_refresh_rate=30
    )

    trainer.fit(model, data_module)
    trainer.test()

    # 预测

    trained_model = BioQAModel.load_from_checkpoint("checkpoints/best-checkpoint.ckpt")
    trained_model.freeze()

    # 最终实现QA的预测！
    for i in range(5):
        q = val_df.iloc[i]
        print(q)
        print(generate_answer(q))
