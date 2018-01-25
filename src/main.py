#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from PIL import Image
from torchvision.transforms import Resize

from model import LapSRN
from loss import CharbonnierLoss
from dataset import Dataset, Split, SplitDataset, get_lr_transform, get_hr_transform

def save_checkpoint(epoch, model, optimizer, filename):
    state = {
        'epoch': epoch + 1,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
    }
    torch.save(state, filename)

def train(epoch, model, criterion, optimizer, writer, train_data):
    cum_loss = 0.0
    for it, batch in enumerate(train_data):
        lr, hr2, hr4, hr8 = [Variable(b).cuda(async=True) for b in batch]        
        
        optimizer.zero_grad()
        hr2_hat, hr4_hat, hr8_hat = model(lr)

        error_1 = criterion(hr2_hat, hr2)
        error_2 = criterion(hr4_hat, hr4)
        error_3 = criterion(hr8_hat, hr8)        
        loss = error_1 + error_2 + error_3
        cum_loss += loss.data[0]
        
        loss.backward()
        # Maybe clip gradient adaptively as in https://arxiv.org/pdf/1511.04587.pdf
        nn.utils.clip_grad_norm(model.parameters(), 1000) # Fixes NaN in SGD for first iteration with high LR.
        #clip = 0.01 / optimizer.param_groups[0]['lr']
        #for p in model.parameters():
        #    p.grad.data.clamp_(-clip, clip)
        optimizer.step()
        
        if it > 0 and (len(train_data) < 10 or it % (len(train_data)//10) == 0):
            print(f"Epoch={epoch}, Batch={it}/{len(train_data)}, Avg. loss = {cum_loss/(it+1)}")

    writer.add_scalar('data/training_error', cum_loss, epoch)
 
def validate(epoch, model, writer, validation_data, validation_dataset):
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
    writer.add_scalar('data/validation_2', cum_psnr2/len(validation_data), epoch)
    writer.add_scalar('data/validation_4', cum_psnr4/len(validation_data), epoch)
    writer.add_scalar('data/validation_8', cum_psnr8/len(validation_data), epoch)

   # Upscale one image for testing:
    lr, _hr2, _hr4, _hr_8 = validation_dataset[0]
    out = model(Variable(lr).cuda().unsqueeze(0))
    for factor, img in enumerate(out):
        upscaled = img.cpu().data
        writer.add_image(f'images/sr_{2**(factor + 1)}', upscaled, epoch)

    return cum_psnr2 + cum_psnr4

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
    parser.add_argument('--lr', type=float, default=1e-5,
                        help="Value for learning rate. Default: 1e-5")
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
    np.random.seed(seed)
    print(f"Using seed={seed}.")
    
    model = LapSRN(depth=args.depth).cuda().train()
    # Paper uses SGD with LR=1e-5
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4, lr=1e-3)
    optimizer = optim.SGD(model.parameters(), weight_decay=1e-4, lr=args.lr, momentum=0.9, nesterov=True)
    criterion = CharbonnierLoss().cuda()

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optim_state'])
        start_epoch = checkpoint['epoch']
        print("Model succesfully loaded from checkpoint")
    else:
        start_epoch = 0

    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5, last_epoch = start_epoch - 1) # See paper
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, cooldown=0, mode='max')

    # Load datasets and set transforms.
    data_path = '../data/processed/eyepacs/train'
    data_path = '../data/raw/all-images/'
    
    # Set needed data transformations.
    CROP_SIZE = 256 # was 128 in paper
    hr_transform = get_hr_transform(CROP_SIZE, random=True)
    lr_transforms = [get_lr_transform(CROP_SIZE, factor, random=True) for factor in [2, 4, 8]]

    dataset = Dataset(data_path, hr_transform=hr_transform, lr_transforms=lr_transforms, verbose=True)
    train_dataset = SplitDataset(dataset, Split.TRAIN, 0.8) 
    validation_dataset = SplitDataset(dataset, Split.TEST, 0.8)
    train_data = data.DataLoader(dataset=train_dataset, num_workers=8,\
                                 batch_size=args.batch_size, shuffle=True, pin_memory=True)
    validation_data = data.DataLoader(dataset=validation_dataset, num_workers=8,\
                                      batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # Initialise logging
    writer = SummaryWriter()

    for epoch in range(start_epoch, args.num_epochs+1):
        print(f"Started epoch num={epoch}.")
        scheduler.step()
        print('Learning rate is {:.7E}'.format(optimizer.param_groups[0]['lr']))
        writer.add_scalar('hyper/lr', optimizer.param_groups[0]['lr'], epoch)
        train(epoch, model, criterion, optimizer, writer, train_data)
        cum_psnr = validate(epoch, model, writer, validation_data, validation_dataset)
        #scheduler.step(cum_psnr)

        if (epoch % args.checkpoint_every) == 0:
            checkpoint_name = f'../checkpoints/srn_{epoch}.pt'
            print(f"Wrote checkpoint {checkpoint_name}!")
            save_checkpoint(epoch, model, optimizer, checkpoint_name)

if __name__ == '__main__':
    main()
