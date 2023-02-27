"""
    本代码用于测试KGQG模型，并从结果中生成模板。
"""
import pytorch_lightning as pl
import torch
from transformers import AdamW
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration

from shared.t5_pegasus_tokenizer import T5PegasusTokenizer

MODEL_NAME = "imxly/t5-pegasus-small"
MODEL_BASE = "E:/LaBarn/checkpoints"

tokenizer = T5PegasusTokenizer.from_pretrained(
    MODEL_NAME,
    model_max_length=128,
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
        max_length=396,
        padding="max_length",
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    # print(source_encoding["input_ids"].shape)

    generated_ids_list = trained_model.model.generate(
        input_ids=source_encoding["input_ids"],
        attention_mask=source_encoding["attention_mask"],
        num_beams=1,
        max_length=396,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True
    )

    ans = tokenizer.batch_decode(
        generated_ids_list,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return ans


trained_model = QGModel.load_from_checkpoint(f"{MODEL_BASE}/KBQGT5/kgqt.ckpt")
trained_model.freeze()


# 最终实现QA的预测！
qs = [
    "【问题解析】得了感冒怎么办",
    "【问题解析】患上了感冒怎么办？",
    "【问题解析】治疗心脏病要多少钱？",
    "【问题解析】缓解阳痿吃什么菜？",
    "【问题解析】得了抽动障碍该吃什么药？",
    "【问题解析】什么年龄容易患高血压？"
]


print(qs)
ans = generate_answer(qs)
print(ans)

exit()

# 接下来，我将从输入和输出中构建一个“模板”出来

def generate_template(triple_str, question_str):
    S, P, O = triple_str.split("【-】")  # 从triple_str中获取主语、谓语、宾语
    if "【?】" in S:  # 主语是待预测的，则替换宾语
        S = S[6:]

        # 已有P和O，下一步是从question_str中找到最相似O的若干个token
        # 首先获取O中所有token的embedding
        source_encoding = tokenizer.encode(
            triple_str,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        SPO_len = torch.where(source_encoding == 102)[1]

        O_encoding = tokenizer.encode(
            O,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        O_len = torch.where(O_encoding == 102)[1] - 1  # 101_

        labels = tokenizer("", return_tensors="pt").input_ids

        outputs = trained_model.model(input_ids=source_encoding, labels=labels, return_dict=True)

        # 在outputs.encoder_last_hidden_state[0]的形状就是512（长度）*512（隐藏层）了
        SPO_embedding = outputs.encoder_last_hidden_state[0][:SPO_len]
        O_embedding = SPO_embedding[-O_len:SPO_len]  # S的嵌入表示对应的索引和向量都有了

        # 然后获取Q中所有token的embedding
        source_encoding = tokenizer.encode(
            question_str,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        Q_len = torch.where(source_encoding == 102)[1] - 1  # 101
        labels = tokenizer("", return_tensors="pt").input_ids

        outputs = trained_model.model(input_ids=source_encoding, labels=labels, return_dict=True)

        # 在outputs.encoder_last_hidden_state[0]的形状就是512（长度）*512（隐藏层）了
        Q_embedding = outputs.encoder_last_hidden_state[0][1:1 + Q_len]

        # 对a和b进行矩阵乘法操作，得到(m, n)的张量
        similarity_matrix = torch.matmul(O_embedding, Q_embedding.transpose(1, 0))
        # 为每个token寻找对应的最大分值
        max_values, max_indices = torch.max(similarity_matrix, dim=1)
        # 其中的最低分者为掩码阈值
        threshold_score = torch.min(max_values)
        # 筛选出大于阈值分数的元素索引
        indexes_for_template = torch.argwhere(threshold_score <= similarity_matrix)[:, 1:]
        # 找出最......

        # 目前获得的需要舍弃的index都还好，下面在Q句中将对应的token替换为特殊token
        input_encoding = torch.index_put_(source_encoding[0], [indexes_for_template + 1],
                                          torch.tensor(tokenizer.added_tokens_encoder["[msk]"]))
        template = tokenizer.decode(input_encoding, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        template = template.replace(" ", "")
        for _ in range(O_len):
            template = template.replace("[msk][msk]", "[msk]")
        print(f"{S}【-】{P}【-】【O】 -> {template}")
    elif O == "【?】":  # 宾语是待预测的，则替换主语
        # 已有S和P，下一步是从question_str中找到最相似S的若干个token
        # 首先获取S中所有token的embedding
        source_encoding = tokenizer.encode(
            triple_str,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        SPO_len = torch.where(source_encoding == 102)[1]

        S_encoding = tokenizer.encode(
            S,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        S_len = torch.where(S_encoding == 102)[1] - 2  # 101 【事实提问】

        labels = tokenizer("", return_tensors="pt").input_ids

        outputs = trained_model.model(input_ids=source_encoding, labels=labels, return_dict=True)

        # 在outputs.encoder_last_hidden_state[0]的形状就是512（长度）*512（隐藏层）了
        SPO_embedding = outputs.encoder_last_hidden_state[0][:SPO_len]
        S_embedding = SPO_embedding[2:2 + S_len]  # S的嵌入表示对应的索引和向量都有了

        # 然后获取Q中所有token的embedding
        source_encoding = tokenizer.encode(
            question_str,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        Q_len = torch.where(source_encoding == 102)[1] - 1  # 101
        labels = tokenizer("", return_tensors="pt").input_ids

        outputs = trained_model.model(input_ids=source_encoding, labels=labels, return_dict=True)

        # 在outputs.encoder_last_hidden_state[0]的形状就是512（长度）*512（隐藏层）了
        Q_embedding = outputs.encoder_last_hidden_state[0][1:1 + Q_len]

        # 对a和b进行矩阵乘法操作，得到(m, n)的张量
        similarity_matrix = torch.matmul(S_embedding, Q_embedding.transpose(1, 0))
        # 为每个token寻找对应的最大分值
        max_values, max_indices = torch.max(similarity_matrix, dim=1)
        # 其中的最低分者为掩码阈值
        threshold_score = torch.min(max_values)
        # 筛选出大于阈值分数的元素索引
        indexes_for_template = torch.argwhere(threshold_score <= similarity_matrix)[:, 1:]
        # 找出最......

        # 目前获得的需要舍弃的index都还好，下面在Q句中将对应的token替换为特殊token
        input_encoding = torch.index_put_(source_encoding[0], [indexes_for_template + 1],
                                          torch.tensor(tokenizer.added_tokens_encoder["[msk]"]))
        template = tokenizer.decode(input_encoding, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        template = template.replace(" ", "")
        for _ in range(S_len):
            template = template.replace("[msk][msk]", "[msk]")
        print(f"【S】【-】{P}【-】{O} -> {template}")


generate_template("【事实提问】【?】【-】症状【-】头晕恶心四肢无力", "头晕恶心四肢无力是得了什么病？")
generate_template("【事实提问】感冒【-】治疗方法【-】【?】", "感冒要怎么治？")





