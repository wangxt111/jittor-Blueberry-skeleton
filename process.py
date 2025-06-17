from dataset.augmentation import process_dataset
import numpy as np

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

print('done.')