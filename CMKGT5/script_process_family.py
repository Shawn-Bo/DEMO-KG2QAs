"""
    从family中生成CMKGT5的训练数据。
"""
import pathlib
import pandas as pd

if __name__ == "__main__":
    # 打开csv文件
    family_df = pd.read_csv("E:\\LaBarn\\datasets\\Haveadoctortohelp\\family\\family.csv", names=["S", "P", "O"],
                            dtype=str)
    max_prompt = -1
    existing_pattern_list = []

    # 输入形式： A【-】B【-】【?】 或者 【？】A【-】B
    with pathlib.Path("../data/family_train_variant.txt").open(mode="w", encoding="utf-8") as f:
        for index, row in family_df.iterrows():
            S, P, O = row
            S = str(S).replace("\n", "").replace("\t", "")
            P = str(P).replace("\n", "").replace("\t", "")
            O = str(O).replace("\n", "").replace("\t", "")

            # 先解决O的模式
            pattern_for_O = f"{S}【-】{P}【-】【?】"
            if pattern_for_O not in existing_pattern_list:
                # 加入字典
                existing_pattern_list.append(pattern_for_O)
                # 寻找所有的数据
                pattern_df = family_df[(family_df["S"] == S) & (family_df["P"] == P)].reset_index(drop=True)
                # 添加变体
                for pindex, (pS, pP, pO) in pattern_df.iterrows():
                    pS = str(pS).replace("\n", "").replace("\t", "")
                    pP = str(pP).replace("\n", "").replace("\t", "")
                    pO = str(pO).replace("\n", "").replace("\t", "")
                    f.write(f"{pindex}【-】{pS}【-】{pP}【-】【?】\t{pO}\n")
                    max_prompt = max(pindex, max_prompt)
            else:
                # 删除已有的模式
                pass

            # 再解决S的模式
            pattern_for_S = f"【?】【-】{P}【-】{O}"
            if pattern_for_S not in existing_pattern_list:
                # 加入字典
                existing_pattern_list.append(pattern_for_S)
                # 寻找所有的数据
                pattern_df = family_df[(family_df["O"] == O) & (family_df["P"] == P)].reset_index(drop=True)
                # 添加变体
                for pindex, (pS, pP, pO) in pattern_df.iterrows():
                    pS = str(pS).replace("\n", "").replace("\t", "")
                    pP = str(pP).replace("\n", "").replace("\t", "")
                    pO = str(pO).replace("\n", "").replace("\t", "")
                    f.write(f"{pindex}【-】【?】【-】{pP}【-】{pO}\t{pS}\n")
                    max_prompt = max(pindex, max_prompt)
            else:
                pass

    print("max_prompt:", max_prompt)  # 6904
