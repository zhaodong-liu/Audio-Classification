import os
import pandas as pd
import numpy as np


submission_df = pd.DataFrame({
    "id": range(2447),  # 创建一个从 0 到 预测数-1 的序列
    "category": np.zeros((2447, 0))  # 使用预测结果填充类别列
})

# 保存 DataFrame 到 CSV 文件
submission_file_path = '/zeros.csv'  # 指定文件保存路径
submission_df.to_csv(submission_file_path, index=False)  # 确保不保存索引列