"""
    用于KGQG KGQT的数据标注处理。处理标注后的csv文件，分割为训练数据和测试数据。
"""
import pandas as pd
from pathlib import Path


def split_dataset(path):
    df = pd.read_csv(path, encoding="utf-8")
    unmarked = df[df.isnull().sum(axis=1) >= 2]  # 未标注QG数据
    marked = df[df.isnull().sum(axis=1) == 1]  # 已标注QG数据
    return unmarked, marked


def generate_marked_CMQG_dataset(df: pd.DataFrame, out_path):
    df = df.fillna("【?】", inplace=False)
    df['S'] = df['S'].str.replace('\n', '')
    df['P'] = df['P'].str.replace('\n', '')
    df['O'] = df['O'].str.replace('\n', '')
    df['Q'] = df['Q'].str.replace('\n', '')
    with Path(out_path).open(encoding="utf-8", mode="w") as f:
        for index, row in df.iterrows():
            S, P, O, Q = row
            f.write(f"【事实提问】{S}【-】{P}【-】{O}\t{Q}\n")


def generate_marked_CMQT_dataset(df: pd.DataFrame, out_path):
    df = df.fillna("【?】", inplace=False)
    df['S'] = df['S'].str.replace('\n', '')
    df['P'] = df['P'].str.replace('\n', '')
    df['O'] = df['O'].str.replace('\n', '')
    df['Q'] = df['Q'].str.replace('\n', '')
    with Path(out_path).open(encoding="utf-8", mode="w") as f:
        for index, row in df.iterrows():
            S, P, O, Q = row
            f.write(f"【问题解析】{Q}\t{S}【-】{P}【-】{O}\n")


def generate_unmarked_CMQT_dataset(df, out_path):
    df = df.fillna("【?】", inplace=False)
    df['Q'] = df['Q'].str.replace('\n', '')
    with Path(out_path).open(encoding="utf-8", mode="w") as f:
        for index, row in df.iterrows():
            S, P, O, Q = row
            f.write(f"【问题解析】{Q}\n")


def generate_CSV_to_label(df: pd.DataFrame, out_path):
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    for path in Path("../data/").glob("preprocessed_table.csv"):
        file_stem = path.stem
        print(file_stem)
        unmarked, marked = split_dataset(f"data/{file_stem}.csv")
        if marked.shape[0] > 0:
            # 已标注数据生成训练集
            generate_marked_CMQT_dataset(marked, f"./data/train_CMQT_{file_stem}.txt")
            generate_marked_CMQG_dataset(marked, f"./data/train_CMQG_{file_stem}.txt")
        if unmarked.shape[0] > 0:
            # 未标注数据生成所谓测试集
            generate_unmarked_CMQT_dataset(unmarked, f"./data/predict_CMQT_{file_stem}.txt")
            # 生成适合标注数据集
            generate_CSV_to_label(unmarked, f"./data/unmarked_{file_stem}.csv")
