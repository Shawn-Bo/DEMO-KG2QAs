"""
    DualAE 的测试模块 同时提供了例程
"""
from typing import List
from DualAE.module_DualAE_model import DualAEModel
from shared.t5_pegasus_tokenizer import T5PegasusTokenizer


def predict_KGQT(nlqs: List[str], tokenizer: T5PegasusTokenizer, no_space=False):
    nlq_encodings = tokenizer.batch_encode_plus(
        nlqs,
        max_length=64,
        padding="max_length",
        truncation="only_second",
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    tsq_token_id_sequences = DualAE_model.predict_KGQT(
        input_ids=nlq_encodings["input_ids"],
        attention_mask=nlq_encodings["attention_mask"]
    )
    if no_space:
        return [tokenizer.decode(sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(" ", "")
                for sequence in tsq_token_id_sequences]
    else:
        return [tokenizer.decode(sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for sequence in tsq_token_id_sequences]


def predict_KGQG(tsqs: List[str], tokenizer: T5PegasusTokenizer, no_space=False):
    tsq_encodings = tokenizer.batch_encode_plus(
        tsqs,
        max_length=64,
        padding="max_length",
        truncation="only_second",
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    nlq_token_id_sequences = DualAE_model.predict_KGQG(
        input_ids=tsq_encodings["input_ids"],
        attention_mask=tsq_encodings["attention_mask"]
    )
    if no_space:
        return [tokenizer.decode(sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(" ", "")
                for sequence in nlq_token_id_sequences]
    else:
        return [tokenizer.decode(sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for sequence in nlq_token_id_sequences]


if __name__ == "__main__":

    checkpoint_path = "E:/LaBarn/checkpoints/KBQGT5/DualAutoEncoder.ckpt"
    MODEL_NAME = "imxly/t5-pegasus-small"
    CACHE_DIR_BASE = "E:/LaBarn/checkpoints"

    DualAE_tokenizer = T5PegasusTokenizer.from_pretrained(
        MODEL_NAME,
        model_max_length=64,
        cache_dir=f"{CACHE_DIR_BASE}/{MODEL_NAME}"
    )
    DualAE_tokenizer.add_tokens(["【事实提问】", "【问题解析】", "【-】", "【?】"])

    DualAE_model = DualAEModel(
        token_embeddings_size=len(DualAE_tokenizer),
        cache_dir_base=CACHE_DIR_BASE,
        model_name=MODEL_NAME
    )
    tsq = predict_KGQT(["【问题解析】得了感冒要怎么治疗？"], tokenizer=DualAE_tokenizer, no_space=True)
    print(tsq)
    nlq = predict_KGQG(["【事实提问】感冒【-】症状【-】【?】"], tokenizer=DualAE_tokenizer, no_space=True)
    print(nlq)
