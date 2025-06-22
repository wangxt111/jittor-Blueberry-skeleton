from dataset.dataset import get_dataloader, transform
from dataset.sampler_new import SamplerEven
import os

sampler = SamplerEven(num_samples=2048,export_path='sampled_points.ply')
train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=sampler,
        transform=transform,
    )