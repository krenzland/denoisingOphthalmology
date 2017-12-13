#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable

from model import LapSRN
from loss import CharbonnierLoss
from dataset import Dataset, get_lr_transform, get_hr_transform

def save_checkpoint(epoch, model, optimizer, filename):
    state = {
        'epoch': epoch + 1,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
    }
    torch.save(state, filename)

def train(epoch, model, criterion, optimizer, train_data):
    cum_loss = 0.0
    for it, batch in enumerate(train_data):
        lr, hr2, hr4, hr8 = [Variable(b).cuda(async=True) for b in batch]        
        hr2_hat, hr4_hat, hr8_hat = model(lr)
        
        error_1 = criterion(hr2_hat, hr2)
        error_2 = criterion(hr4_hat, hr4)
        error_3 = criterion(hr8_hat, hr8)        
        loss = error_1 + error_2 + error_3 
        cum_loss += loss.data[0]
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 10) # Fixes NaN in SGD for first iteration with high LR.
        optimizer.step()
        
        if it > 0 and it % 10 == 0:
            print(f"Epoch={epoch}, Batch={it}/{len(train_data)}, Avg. loss = {cum_loss/it}")
 
def validate(model, validation_data):
    cum_psnr2, cum_psnr4, cum_psnr8 = 0.0, 0.0, 0.0
    
    for batch in validation_data:
        lr, hr2, hr4, hr8 = [Variable(b).cuda(async=True) for b in batch]
        hr2_hat, hr4_hat, hr8_hat = model(lr)
        
        mse = nn.MSELoss().cuda()
        error_1 = mse(hr2_hat, hr2)
        error_2 = mse(hr4_hat, hr4)
        error_3 = mse(hr8_hat, hr8)      
        
        get_psnr = lambda e: -10 * np.log10(e.data[0])
        cum_psnr2 += get_psnr(error_1)
        cum_psnr4 += get_psnr(error_2)
        cum_psnr8 += get_psnr(error_3)
        
    print(f"Avg. PSNR: {cum_psnr2/len(validation_data)}, {cum_psnr4/len(validation_data)},{cum_psnr8/len(validation_data)}.")



def main():
    # Currently only support CUDA, not CPU backend..
    assert(torch.cuda.is_available)

    parser = argparse.ArgumentParser(description="Run LapSRN training.")
    parser.add_argument('--checkpoint', help="Path to checkpoint file.")
    parser.add_argument('--seed', type=int,
                        help="Value for random seed. Default: Random.")
    # ---------------------- Model settings ---------------------------------------
    parser.add_argument('--depth', type=int, default=5,
                        help="Set number of convolution layers for each feature extraction stage.")
    # ------------------ Optimizer settings ---------------------------------------
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Value for learning rate. Default: 1e-4")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Set batch size. Default: 32")
    parser.add_argument('--num_epochs', type=int, default=50,
                        help="Set number of epochs. Default: 50")
    parser.add_argument('--checkpoint_every', type=int, default=10,
                        help="Sets how often a checkpoint gets written. Default: 10")

    args = parser.parse_args()
    print(f"Called with args={args}")

    # Generate and print random seed:
    if args.seed:
        seed = int(args.seed)
    else:
        seed = np.random.randint(0, 10000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Using seed={seed}.")
    
    model = LapSRN(depth=args.depth).cuda()
    # Paper uses SGD with LR=1e-4, doesnt work here for some reason.
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
    criterion = CharbonnierLoss().cuda()

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optim_state'])
        start_epoch = checkpoint['epoch']
        print("Model succesfully loaded from checkpoint")
    else:
        start_epoch = 0

    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5, last_epoch = start_epoch) # See paper

    # Load datasets and set transforms.
    # TODO: Make dataset paths configurable!
    train_data_path = '../data/raw/all-images/'
    validation_data_path = '../data/raw/DRIVE/training/images'

    # Set needed data transformations.
    CROP_SIZE = 128 # maybe 256?
    hr_transform = get_hr_transform(CROP_SIZE)
    lr_transforms = [get_lr_transform(CROP_SIZE, factor) for factor in [8, 4, 2]]

    train_dataset = Dataset(train_data_path, hr_transform=hr_transform, lr_transforms=lr_transforms)
    train_data = data.DataLoader(dataset=train_dataset, num_workers=4,\
                                 batch_size=args.batch_size, shuffle=True, pin_memory=True)
    validation_dataset = Dataset(validation_data_path, hr_transform=hr_transform,\
                                 lr_transforms=lr_transforms)
    validation_data = data.DataLoader(dataset=validation_dataset, num_workers=4,\
                                      batch_size=args.batch_size, shuffle=True, pin_memory=True)

    for epoch in range(start_epoch, args.num_epochs+1):
        print(f"Started epoch num={epoch}.")   
        scheduler.step()
        train(epoch, model, criterion, optimizer, train_data)
        validate(model, validation_data)
    
        if (epoch % args.checkpoint_every) == 0:
            checkpoint_name = f'../checkpoints/srn_{epoch}.pt'
            save_checkpoint(epoch, model, optimizer, checkpoint_name)

if __name__ == '__main__':
    main()
