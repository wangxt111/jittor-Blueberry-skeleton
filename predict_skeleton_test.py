import jittor as jt
import numpy as np
import os
import argparse
from tqdm import tqdm
import random

from dataset.dataset import get_dataloader, transform
from dataset.sampler import SamplerMix
from models.skeleton import create_model
from models.metrics import J2J

# python predict_skeleton.py --predict_data_list data/test_list.txt --data_root data --model_name pct --pretrained_model /home/ubuntu/jittor2025_skeleton/output/base+J2Jloss+no_epochdata/skeleton/best_model.pkl --predict_output_dir predict1 --batch_size 16

def predict_and_eval(args, seed):
    print(f"Running prediction with seed {seed} ...")
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = create_model(
        model_name=args.model_name,
        model_type=args.model_type
    )
    sampler = SamplerMix(num_samples=1024, vertex_samples=512)
    if args.pretrained_model:
        model.load(args.pretrained_model)
    predict_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.predict_data_list,
        train=False,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        transform=transform,
    )
    model.eval()

    total_j2j_loss = 0.0
    count = 0
    for batch_idx, data in tqdm(enumerate(predict_loader)):
        vertices, joints, cls, ids = data['vertices'], data['joints'], data['cls'], data['id']

        if vertices.ndim == 3:
            vertices = vertices.permute(0, 2, 1)  # (B, 3, N)

        outputs = model(vertices)  # 输出形状假设为 (B, J*3) 或 (B, J, 3)
        B = outputs.shape[0]
        if outputs.ndim == 2:
            outputs = outputs.reshape(B, -1, 3)

        for i in range(B):
            j2j_loss = J2J(outputs[i].reshape(-1, 3), joints[i].reshape(-1, 3)).item()
            total_j2j_loss += j2j_loss
            count += 1

    avg_j2j = total_j2j_loss / count
    print(f"Seed {seed} average Chamfer Distance J2J: {avg_j2j:.6f}")
    return avg_j2j

def main():
    parser = argparse.ArgumentParser(description='Predict with multiple seeds and compute CD-J2J')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--predict_data_list', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='pct',
                        choices=['pct', 'pct2', 'custom_pct', 'skeleton'])
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'enhanced'])
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seeds', type=int, nargs='+', default=[123, 456, 789],
                        help='List of random seeds to run prediction with')

    args = parser.parse_args()

    jt.flags.use_cuda = 1
    cd_list = []
    for seed in args.seeds:
        avg_j2j = predict_and_eval(args, seed)
        cd_list.append(avg_j2j)

    print("All seeds CD-J2J results:", cd_list)
    print("Mean CD-J2J:", np.mean(cd_list))

if __name__ == '__main__':
    main()
