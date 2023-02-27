from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from DualAE.module_DualAE_model import DualAEModel
from shared.t5_pegasus_tokenizer import T5PegasusTokenizer


def extract_questions_and_answers(question_path: Path):
    with question_path.open("r", encoding="utf-8") as f:
        data = [line.replace("\n", "").split("\t") for line in f.readlines()]
        # for line in data:
        #     print(line)
    return pd.DataFrame(data, columns=["masked_question", "origin_question"])


class DualAEDataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: T5PegasusTokenizer,
            source_max_token_len: int = 64,
            target_max_token_len: int = 64
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
            data_row[0],
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            data_row[1],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        labels = target_encoding["input_ids"]
        return dict(
            masked_question=data_row[0],
            origin_question=data_row[1],
            input_ids=source_encoding["input_ids"].flatten(),
            attention_mask=source_encoding["attention_mask"].flatten(),
            labels=labels.flatten()
        )


class DualAEDatasetMoudle(pl.LightningDataModule):
    def __init__(
            self,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame,
            tokenizer: T5PegasusTokenizer,
            batch_size: int = 8,
            source_max_token_len: int = 64,
            target_max_token_len: int = 64
    ):
        super().__init__()
        self.test_dataset = None
        self.train_dataset = None
        self.batch_size = batch_size
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.tokenizer = tokenizer
        self.train_df = train_df
        self.test_df = test_df

    def setup(self):
        self.train_dataset = DualAEDataset(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

        self.test_dataset = DualAEDataset(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # 不再重新打乱数据，目的是便于分析训练过程。
            num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=8,
            num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=8,
            num_workers=0
        )


if __name__ == "__main__":

    MODEL_NAME = "imxly/t5-pegasus-small"
    CACHE_DIR_BASE = "E:/LaBarn/checkpoints"
    TRAINING_FILE = "../data/family_train_variant.txt"

    BATCH_SIZE = 16
    N_EPOCHS = 20000

    LOGGING_DIR = "training-logs"
    LOGGING_NAME = "DualAutoEncoder"

    pl.seed_everything(seed=42)
    df = extract_questions_and_answers(Path(TRAINING_FILE))
    train_df, val_df = train_test_split(df, test_size=0.005)

    DualAE_tokenizer = T5PegasusTokenizer.from_pretrained(
        MODEL_NAME,
        model_max_length=64,
        cache_dir=f"{CACHE_DIR_BASE}/{MODEL_NAME}"
    )
    DualAE_tokenizer.add_tokens(["【事实提问】", "【问题解析】", "【-】", "【?】"])

    # 加载数据模块
    data_module = DualAEDatasetMoudle(train_df, val_df, DualAE_tokenizer, batch_size=BATCH_SIZE)
    data_module.setup()

    # 加载模型
    # model = QGModel.load_from_checkpoint("checkpoints/last.ckpt")
    model = DualAEModel(
        token_embeddings_size=len(DualAE_tokenizer),
        cache_dir_base=CACHE_DIR_BASE,
        model_name=MODEL_NAME
    )

    # 开始训练
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_dae",
        filename="best-checkpoint",
        verbose=True,
        save_last=True,
    )

    logger = TensorBoardLogger(LOGGING_DIR, name=LOGGING_NAME)

    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=N_EPOCHS,
        gpus=3,
        progress_bar_refresh_rate=30,
        accelerator="ddp"
    )

    trainer.fit(model, data_module)
    trainer.test()
