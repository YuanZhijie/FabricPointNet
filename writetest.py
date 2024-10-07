import random
import os

# 源文件路径
source_file = 'data\\modelnet40_normal_resampled\\modelnet10_test.txt'
# 目标文件路径
target_file = 'data\\modelnet40_normal_resampled\\dataset02_test.txt'


# 打开源文件和目标文件
with open(source_file, 'r') as file

    lines = file.readlines()

with open(target_file, 'w') as file:
    # 每五行随机选择一行并写入目标文件
    for i in range(4, len(lines), 5):
        random_line = random.choice(lines[i-4:i+1])
        file.write(random_line)

print("复制完成！")

# 读取目标文件
with open(target_file, 'r') as file:
    target_lines = file.readlines()

# 删除源文件中对应的内容
with open(source_file, 'w') as file:
    file.writelines([line for line in lines if line not in target_lines])

print("删除完成！")
