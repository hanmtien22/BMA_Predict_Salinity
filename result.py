import pandas as pd
import openpyxl 

# 1. Đọc dữ liệu
df_my = pd.read_csv('result.csv')
df_paper = pd.read_csv('result_paper.csv')

# 2. Đặt cột 'Model' làm index cho cả 2 dataframe để gộp theo đúng dòng
df_my = df_my.set_index('Model')
df_paper = df_paper.set_index('Model')

# 3. Ghép 2 DataFrame lại với MultiIndex ở mức Cột (Columns)
df_combined = pd.concat([df_paper, df_my], axis=1, keys=['Paper', 'Ket qua cua em'])

print(df_combined)
df_combined.to_excel('combined_results1.xlsx')