import argparse
import os
import warnings

import torch
import torch.nn as nn
import yaml
from ml_collections import ConfigDict
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MSSDatasets
from tfc_tdf_v3 import TFC_TDF_net
from utils import manual_seed

warnings.filterwarnings("ignore")

# I get "Out Of Memory" error when trying to continue training from ckpt. Changing the batch size to 1 did not help.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:100'  # Add this line to avoid "Out Of Memory" error


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    parser.add_argument("--model_path", type=str, help="path to model checkpoint folder (containing config.yaml)")
    parser.add_argument("--data_root", type=str, help="path to folder containing all training datasets")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader num_workers")
    parser.add_argument("--pin_memory", type=bool, default=False, help="dataloader pin_memory")
    parser.add_argument("--num_steps", type=int, default=0, help="total number of training steps")
    parser.add_argument("--ckpt", type=str, help="path to checkpoint file")
    args = parser.parse_args()

    manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    with open(args.model_path + '/config.yaml') as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
    config.training.num_steps = args.num_steps

    trainset = MSSDatasets(config, args.data_root)

    train_loader = DataLoader(
        trainset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    model = TFC_TDF_net(config)
    model.train()

    device_ids = args.device_ids
    if type(device_ids) == int:
        device = torch.device(f'cuda:{device_ids}')
        model = model.to(device)
    else:
        device = torch.device(f'cuda:{device_ids[0]}')
        model = nn.DataParallel(model, device_ids=device_ids).to(device)

    optimizer = Adam(model.parameters(), lr=config.training.lr)

    # Define step
    step = 0

    # Load checkpoint if provided
    if args.ckpt:
        checkpoint = torch.load(args.ckpt)
        model_state_dict = checkpoint['model_state_dict']

        # If the model is wrapped with nn.DataParallel, add 'module.' prefix
        if isinstance(model, nn.DataParallel):
            model_state_dict = {'module.' + k: v for k, v in model_state_dict.items()}

        model.load_state_dict(model_state_dict)

        # Move optimizer state to correct device
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        for state in optimizer_state_dict['param_groups']:
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        optimizer.load_state_dict(optimizer_state_dict)

        # Load step number from checkpoint
        step = checkpoint.get('step', 0)

        # Print the step number at which the checkpoint was saved
        print(f'Loaded checkpoint from {args.ckpt} at step {step}')

    print('Train Loop')
    scaler = GradScaler()

    try:
        for batch in tqdm(train_loader):
            y = batch.to(device)
            x = y.sum(1)  # mixture
            if config.training.target_instrument is not None:
                i = config.training.instruments.index(config.training.target_instrument)
                y = y[:, i]
            with torch.cuda.amp.autocast():
                y_ = model(x)
                loss = nn.MSELoss()(y_, y)

            scaler.scale(loss).backward()
            if config.training.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # Increment step after processing each batch
            step += 1

    except KeyboardInterrupt:
        print('Interrupted, saving model...')
    finally:
        state_dict = model.state_dict() if type(device_ids) == int else model.module.state_dict()
        # Save both model state and optimizer state
        ckpt_path = args.model_path + '/ckpt'
        if os.path.exists(ckpt_path):
            overwrite = input(f'Checkpoint file {ckpt_path} already exists. Do you want to overwrite it? (y/n): ')
            if overwrite.lower() != 'y':
                print('Model not saved.')
                return
        # Save the current step number along with the model and optimizer states
        torch.save({
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
        }, ckpt_path)
        print('Model saved.')


if __name__ == "__main__":
    train()
