import os
import pandas as pd

# 文件路径
label_file = './AAAI_3_4_labels.csv'
data_dir = './data_raw/downstreamIV'

# 读取标签文件
labels_df = pd.read_csv(label_file)

# 要统计的标签列
label_columns = ['DIEINHOSPITAL', 'Readmission_30', 'Multiple_ICUs', 'sepsis_all', 'FirstICU24_AKI_ALL']

# 仅保留存在对应数据文件的样本
available_ids = []
for icu_id in labels_df['ICUSTAY_ID']:
    file_path = os.path.join(data_dir, f'{icu_id}.csv')
    if os.path.exists(file_path):
        available_ids.append(icu_id)

filtered_labels = labels_df[labels_df['ICUSTAY_ID'].isin(available_ids)].copy()

# 统计每个标签的0和1比例
stats = []
for col in label_columns:
    total = len(filtered_labels)
    count_1 = filtered_labels[col].sum()
    count_0 = total - count_1
    ratio_1 = count_1 / total if total > 0 else 0
    ratio_0 = count_0 / total if total > 0 else 0
    stats.append({
        'Label': col,
        'Total': total,
        'Count_0': int(count_0),
        'Count_1': int(count_1),
        'Ratio_0': round(ratio_0, 4),
        'Ratio_1': round(ratio_1, 4)
    })

# 生成结果表
result_df = pd.DataFrame(stats)

# 打印结果
print("\n=== 标签 0/1 比例统计结果 ===")
print(result_df)

# 如果希望保存结果到文件
result_df.to_csv('./label_ratio_stats.csv', index=False)
print("\n结果已保存到 label_ratio_stats.csv")
