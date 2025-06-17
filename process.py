from dataset.augmentation import process_dataset
import numpy as np
import os

print(1)

process_dataset(
    data_dir = 'data/train/vroid',
    output_dir = 'newdata/train/vroid',
    is_train = True
)

process_dataset(
    data_dir = 'data/train/mixamo',
    output_dir = 'newdata/train/mixamo',
    is_train = True
)

print(2)

# # 跟踪集处理
# process_dataset(
#     data_dir = 'data/track',
#     output_dir = 'newdata/track',
#     is_train = False
# )

# print(3)
# print(4)

# 测试集处理
process_dataset(
    data_dir = 'data/test/mixamo',
    output_dir = 'newdata/test/mixamo',
    is_train = False
)

print(5)

process_dataset(
    data_dir = 'data/test/vroid',
    output_dir = 'newdata/test/vroid',
    is_train = False
)

def update_train_list(train_list_path):
    """
    读取train_list.txt，为每个训练数据文件添加_aug后缀的新文件名。
    """
    with open(train_list_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    new_lines = []
    for line in lines:
        new_lines.append(line)
        if line.startswith('train/vroid/') or line.startswith('train/mixamo/'):
            base_name, ext = os.path.splitext(line)
            new_lines.append(f"{base_name}_aug{ext}")

    # 将更改写入新的文件
    output_train_list_path = os.path.join(os.path.dirname(train_list_path), 'train_list_1.txt')
    with open(output_train_list_path, 'w') as f:
        for line in new_lines:
            f.write(line + '\n')
    print(f"Updated {output_train_list_path} with augmented file names.")

# 调用函数更新 train_list.txt
update_train_list('data/train_list.txt')

print('done.')