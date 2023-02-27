"""
    本代码曾用于标注训练数据。
"""

import pandas
exit()  # 注意保存数据
all_table = pandas.read_csv("../data/preprocessed_table.csv", encoding="utf-8", dtype=str)
all_table = all_table.drop(all_table[all_table["Q"].str.len() <= 6].index).reset_index(drop=True)
all_table = all_table.sample(frac=1)

all_table.loc[all_table["Q"].fillna("").str.contains("怎样治|如何诊|方法治|有办法|有哪些办法|怎么去|咋回事|什么病|治疗好|怎么解决|怎么办|怎样诊治|怎么样治|如何医|怎么治|治疗办法|怎么样去治疗|如何治疗|怎么治疗|怎样治疗|治疗方法|如何救治|彻底改善"), "P"] = "治疗方法"
all_table.loc[all_table["Q"].fillna("").str.contains("原因|病因"), "P"] = "病因"
all_table.loc[all_table["Q"].fillna("").str.contains("是不是|症状|怎么回事|什么原因|正常吗|什么表现"), "P"] = "症状"
all_table.loc[all_table["Q"].fillna("").str.contains("如何诊断"), "P"] = "诊断方法"
all_table.loc[all_table["Q"].fillna("").str.contains("吃什么药"), "P"] = "好评药物"
all_table.loc[all_table["Q"].fillna("").str.contains("预防"), "P"] = "预防方法"
all_table.loc[all_table["Q"].fillna("").str.contains("需要注意什么|该注意哪些"), "P"] = "生活事宜"
all_table.loc[all_table["Q"].fillna("").str.contains("能不能改善|能治好|能康|能不能治"), "P"] = "治疗机会"
all_table.loc[all_table["Q"].fillna("").str.contains("什么价格|治疗费用|花费|多少钱|价格多少"), "P"] = "治疗费用"
all_table.loc[all_table["Q"].fillna("").str.contains("可以吃|饮食该注意|能否吃|不能吃"), "P"] = "忌食"
all_table.loc[all_table["Q"].fillna("").str.contains("平时应该多吃什么"), "P"] = "宜吃"
all_table.loc[all_table["Q"].fillna("").str.contains("什么是"), "P"] = "描述"

all_table.loc[all_table["Q"].fillna("").str.contains("什么地方|哪里手术|哪里治|医院好|哪里看"), "P"] = "治疗医院"

all_table.to_csv("data/preprocessed_table.csv", encoding="utf-8", index=False)
