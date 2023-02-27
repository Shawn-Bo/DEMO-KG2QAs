import pandas as pd
pd.options.display.float_format = '{:.5f}'.format
import pytorch_lightning as pl
from transformers import AdamW
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
import torch
from shared.t5_pegasus_tokenizer import T5PegasusTokenizer

MODEL_NAME = "imxly/t5-pegasus-small"
MODEL_BASE = "E:/LaBarn/checkpoints"

tokenizer = T5PegasusTokenizer.from_pretrained(
    MODEL_NAME,
    model_max_length=64,
    cache_dir=f"{MODEL_BASE}/{MODEL_NAME}")

tokenizer.add_tokens(["【恢复问句】", "[MSK]", "【恢复问诊】,【恢复诊断】", "【事实提问】", "【-】", "【问题解析】", "【?】"])


class QGModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            cache_dir=f"{MODEL_BASE}/{MODEL_NAME}",
            return_dict=True)
        self.model.resize_token_embeddings(len(tokenizer))

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


def generate_answer(questions):
    source_encoding = tokenizer.batch_encode_plus(
        questions,
        max_length=64,
        padding="max_length",
        truncation="only_second",
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    # print(source_encoding["input_ids"].shape)

    generated_ids_list = trained_model.kgqt_model.generate(
        input_ids=source_encoding["input_ids"],
        attention_mask=source_encoding["attention_mask"],
        num_beams=5,
        max_length=64,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True,
        num_return_sequences=3,
        output_scores=True,
        return_dict_in_generate=True
    )

    # 接下来，计算每个候选序列的概率
    result_token_ids = generated_ids_list.sequences
    result_scores = generated_ids_list.scores
    result_sequences_scores = torch.exp(generated_ids_list.sequences_scores)

    # transition_scores = trained_model.model.compute_transition_scores(
    #     generated_ids_list.sequences, generated_ids_list.scores, normalize_logits=False
    # )
    # output_length = 1 + np.sum(transition_scores.numpy() < 0, axis=1)
    # length_penalty = trained_model.model.generation_config.length_penalty
    # reconstructed_scores = transition_scores.sum(axis=1) / (output_length ** length_penalty)

    conclusions = []
    result_possibilities = []
    for token_position_index in range(len(result_scores)):
        # 每个项目都是需要执行一遍softmax,保证可以获取概率
        result_possibilities.append(torch.nn.functional.softmax(result_scores[token_position_index], dim=1))

    for return_sequence_index in range(result_token_ids.shape[0]):
        # return_sequence_index 可以取10个
        return_sequence = result_token_ids[return_sequence_index]  # 目前是(64个int组成的列表)
        # 处理 result_scores 将分数转为概率
        sequence_possibility = 1.0
        for token_position_index, token_id in enumerate(return_sequence[:]):
            if token_position_index == 0:
                continue
            if token_id == 102:
                break  # 已生成到结束符
            # 这里要注意的是一个匹配问题
            token_possibility = result_possibilities[token_position_index-1][return_sequence_index][token_id]  # 生成这个token的概率
            sequence_possibility *= float(token_possibility.numpy())
        # 差不多了就输出结果吧
        conclusions.append([
            # float(transition_scores[return_sequence_index].numpy()),
            # sequence_possibility,
            tokenizer.decode(return_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True),
            float(result_sequences_scores[return_sequence_index].numpy())
        ])

    # 如果不考虑概率的话，可以用这段
    # ans = tokenizer.batch_decode(
    #     generated_ids_list[0],  # ans[0]是可以用于解码的token_ids，解码之后就得到了
    #     skip_special_tokens=True,
    #     clean_up_tokenization_spaces=True
    # )

    return conclusions


trained_model = QGModel.load_from_checkpoint(f"{MODEL_BASE}/KBQGT5/kgqa.ckpt")
trained_model.freeze()


# 最终实现QA的预测！
qs = [
    "1【-】失眠【-】忌吃【-】【?】",
    "2【-】失眠【-】忌吃【-】【?】",
    "3【-】失眠【-】忌吃【-】【?】",
    "4【-】失眠【-】忌吃【-】【?】",
    "5【-】失眠【-】忌吃【-】【?】",
    "6【-】失眠【-】忌吃【-】【?】",
    "7【-】失眠【-】忌吃【-】【?】",
    "8【-】失眠【-】忌吃【-】【?】"  # 这个算是额外测试！
]

for q in qs:
    conclusions = generate_answer([q])
    print(q)
    print(pd.DataFrame(conclusions, columns=["答案", "分数"]).sort_values("分数", ascending=False).drop_duplicates(subset=["答案"]).reset_index(drop=True))

