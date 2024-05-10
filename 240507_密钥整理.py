import pandas as pd

# 读取CSV文件
df = pd.read_csv('exp4.csv', encoding='ISO-8859-1')

# 根据D、H、I、J列进行分组，并且删除重复的行
df_no_duplicates = df.drop_duplicates(subset=['name', 'login_uri', 'login_username', 'login_password'])

# 将处理后的数据保存到新的CSV文件中
df_no_duplicates.to_csv('exp4_3.csv', index=False)
