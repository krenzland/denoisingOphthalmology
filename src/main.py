#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import autograd
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from PIL import Image
from torchvision.transforms import Resize

from model import LapSRN, PatchD
from loss import CharbonnierLoss, make_vgg16_loss
from dataset import Dataset, Split, SplitDataset, get_lr_transform, get_hr_transform
from gan import GAN

def save_checkpoint(epoch, generator, discriminator, optimizer_generator, optimizer_disc, filename, use_adversarial):
    state = {
        'epoch': epoch + 1,
        'model_state': generator.state_dict(),
        'optim_state': optimizer_generator.state_dict(),
    }
    if use_adversarial:
        state['model_state_discriminator'] = discriminator.state_dict()
        state['optim_state_discriminator'] = optimizer_disc.state_dict()

    torch.save(state, filename)


def train(epoch, generator, gan, criterion, optimizer_generator, writer, train_data):
    use_adversarial = gan is not None
    
    cum_image_loss = 0.0
    cum_loss = 0.0
    if use_adversarial:
        cum_critic_loss = 0.0
        cum_adversarial_loss = 0.0
        
    for it, batch in enumerate(train_data):
        lr, hr2, hr4 = [Variable(b).cuda(async=True) for b in batch]        

        if use_adversarial:
            # Update critic network
            critic_loss = gan.update(generator=generator,
                                     lr=lr,
                                     hr4=hr4)           
            cum_critic_loss += critic_loss.data[0]

        optimizer_generator.zero_grad()
        hr2_hat, hr4_hat = generator(lr)
        
        # Compute pixel-wise/perceptual loss for both output imgs.
        loss_hr2 = criterion(hr2_hat, hr2)
        loss_hr4 = criterion(hr4_hat, hr4)

        if use_adversarial:
            adversarial_loss = gan.get_generator_loss(hr4_hat=hr4_hat)
            cum_adversarial_loss += adversarial_loss.data[0]
        else:
            adversarial_loss = 0.0
        image_loss = loss_hr2 + loss_hr4
        cum_image_loss += image_loss.data[0]

        loss = image_loss + adversarial_loss
        cum_loss += loss.data[0]
        
        loss.backward()
        optimizer_generator.step()
        
        if ((it + 1)% (len(train_data)//10) == 0):
            print("Epoch={}, Batch={}/{}, Avg. loss = {}".format(epoch, it + 1, len(train_data),
                                                                 cum_loss/(it+1)))
    print("Epoch={}, image loss={}, total Loss={}".format(
        epoch,
        cum_image_loss/len(train_data),
        cum_loss/len(train_data)))
    writer.add_scalar('data/image_loss', cum_image_loss/len(train_data), epoch)
    writer.add_scalar('data/total_loss', cum_loss, epoch/len(train_data))

    if use_adversarial:
        print("Negative critic loss = {}, Adversarial loss={}".format(
            -cum_critic_loss/len(train_data),
            cum_adversarial_loss/len(train_data)))
        writer.add_scalar('data/neg_critic_loss', -cum_critic_loss/len(train_data), epoch)
        writer.add_scalar('data/adversarial_loss', cum_adversarial_loss/len(train_data), epoch)
 
def validate(epoch, generator, gan, criterion, writer, validation_data):
    use_adversarial = gan is not None
    
    cum_psnr2, cum_psnr4, cum_hr2_loss, cum_hr4_loss = 0.0, 0.0, 0.0, 0.0
    if use_adversarial:
        cum_critic_loss = 0.0
    
    for batch in validation_data:
        lr, hr2, hr4 = [Variable(b).cuda(async=True) for b in batch]
        hr2_hat, hr4_hat = generator(lr)
        
        mse = nn.MSELoss().cuda()
        mse_1 = mse(hr2_hat, hr2)
        mse_2 = mse(hr4_hat, hr4)
        
        get_psnr = lambda e: -10 * np.log10(e.data[0])
        cum_psnr2 += get_psnr(mse_1)
        cum_psnr4 += get_psnr(mse_2)

        # Compute pixel-wise/perceptual loss for both output imgs.
        loss_hr2 = criterion(hr2_hat, hr2)
        loss_hr4 = criterion(hr4_hat, hr4)

        cum_hr2_loss += loss_hr2.data[0]
        cum_hr4_loss += loss_hr4.data[0]

        if use_adversarial:
            critic_loss = gan.get_discriminator_loss(hr4=hr4, hr4_hat=hr4_hat)
            cum_critic_loss += critic_loss.data[0]


    print("Validation Avg. PSNR: {}, {}, Validation Image Loss: {} {}.".format(
        cum_psnr2/len(validation_data),
        cum_psnr4/len(validation_data),
        cum_hr2_loss/len(validation_data),
        cum_hr4_loss/len(validation_data)
    ))

    if use_adversarial:
        print("Validation Neg. Critic Loss = {}".format(-cum_critic_loss))
        writer.add_scalar('data/validation_neg_critic_loss', -cum_critic_loss/len(validation_data), epoch)
        
    writer.add_scalar('data/validation_psnr_2', cum_psnr2/len(validation_data), epoch)
    writer.add_scalar('data/validation_psnr_4', cum_psnr4/len(validation_data), epoch)
    writer.add_scalar('data/validation_image_loss_2', cum_hr2_loss/len(validation_data), epoch)
    writer.add_scalar('data/validation_image_loss_4', cum_hr4_loss/len(validation_data), epoch)

    # Upscale one image for testing:
    lr, _hr2, _hr4 = validation_data.dataset[np.random.randint(0, len(validation_data.dataset))]
    out = generator(Variable(lr).cuda().unsqueeze(0))
    for factor, img in enumerate(out):
        out = img.data.clone()

        # Remove normalisation from image.
        mean= [0.485, 0.456, 0.406]
        std= [0.229, 0.224, 0.225]

        # Undo input[channel] = (input[channel] - mean[channel]) / std[channel]
        for t, m, s in zip(out, mean, std):
            t.mul_(s)
            t.add_(m)
        upscaled = out.cpu().clamp(0,1)
        writer.add_image('images/sr_{}'.format(2**(factor+1)), upscaled, epoch)

    return cum_psnr2 + cum_psnr4

def main():
    # Currently only support CUDA, not CPU backend..
    assert(torch.cuda.is_available)

    parser = argparse.ArgumentParser(description="Run LapSRN training.")
    parser.add_argument('--seed', type=int,
                        help="Value for random seed. Default: Random.")
    # ---------------------- Model settings ---------------------------------------
    parser.add_argument('--depth', type=int, default=10,
                        help="Set number of convolution layers for each feature extraction stage.")
    # ------------------ Optimizer settings ---------------------------------------
    parser.add_argument('--lr', type=float, default=1e-5,
                        help="Value for learning rate. Default: 1e-5")
    parser.add_argument('--batch-size', type=int, default=64,
                        help="Set batch size. Default: 32")
    parser.add_argument('--num-epochs', type=int, default=16670,
                        help="Set number of epochs. Default: 16670")
    parser.add_argument('--perceptual', action='store_true',
                        help="If true, use perceptual loss.")
    parser.add_argument('--adversarial', action='store_true',
                        help="If true, use adversarial loss.")
    parser.add_argument('--wgan', action='store_true', dest='use_wgan',
                        help="If true, use wgan-gp instead of standard GAN.")
    parser.add_argument('--adversarial-weight', type=float, default=1.0,
                         help="Sets weight of adversarial loss. Default: 1.0")
    # Directories
    parser.add_argument('--data-dir', default='../data/processed/messidor',
                        help="Sets tensorboard run directory, default ../data/processed/messidor")
    parser.add_argument('--tensorboard-dir', default='../runs',
                        help="Sets tensorboard run directory, default ../runs/")
    parser.add_argument('--checkpoint', help="Path to checkpoint file.")
    parser.add_argument('--checkpoint-dir', default='../checkpoints',
                        help="Sets checkpoint directory, default ../checkpoints/")
    parser.add_argument('--checkpoint-every', type=int, default=333,
                        help="Sets how often a checkpoint gets written. Default: 333")

    args = parser.parse_args()
    print("Called with args={}".format(args))

    # Generate and print random seed:
    if args.seed:
        seed = int(args.seed)
    else:
        seed = np.random.randint(0, 10000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print("Using seed={}.".format(seed))
    
    # Set up networks
    generator = LapSRN(depth=args.depth).cuda().train()
    print(generator)
    if args.adversarial:
        if args.use_wgan:
            discriminator = PatchD(use_sigmoid=False, num_layers=4).cuda().train()
        else:
           discriminator = PatchD(use_sigmoid=True, num_layers=4).cuda().train() 
        print(discriminator)
    else:
        assert(not args.use_wgan)
        discriminator = None

    # Setup optimizers
    if args.adversarial:
        if args.use_wgan:
            optimizer_generator = optim.Adam(generator.parameters(), betas=(0.0, 0.9), lr=args.lr)
            optimizer_discriminator = optim.Adam(discriminator.parameters(), betas=(0.0, 0.9), lr=args.lr)
        else:
            optimizer_generator = optim.Adam(generator.parameters(), weight_decay=1e-4, lr=args.lr)
            optimizer_discriminator = optim.Adam(discriminator.parameters(), weight_decay=1e-4, lr=args.lr)
    else:
        # Paper uses SGD with LR=1e-5
        optimizer_generator = optim.SGD(generator.parameters(), weight_decay=1e-4, lr=args.lr, momentum=0.9)
        optimizer_generator = optim.Adam(generator.parameters(), weight_decay=1e-4, lr=args.lr)
        optimizer_discriminator = None

    # Set up losses
    if args.perceptual:
        criterion = make_vgg16_loss(nn.MSELoss().cuda()).cuda()
    else:
        criterion = CharbonnierLoss().cuda()
    if args.adversarial:
        gan = GAN(discriminator=discriminator,
                  optimizer=optimizer_discriminator,
                  adversarial_weight=args.adversarial_weight,
                  use_wgan=args.use_wgan)
    else:
        gan = None

    # Load from checkpoint.
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        generator.load_state_dict(checkpoint['model_state'])
        optimizer_generator.load_state_dict(checkpoint['optim_state'])
        start_epoch = checkpoint['epoch']
        if args.adversarial:
            if 'optim_state_optimizer' in checkpoint:
                optimizer_discriminator.load_state_dict(checkpoint['optim_state_optimizer'])
            else:
                print("Warning: Only loading generator from checkpoint!")
                print("Resetting epoch to 0 and learning rates to --lr argument.")
                start_epoch = 0
                for param_group in optimizer_generator.param_groups:
                    param_group['lr'] = args.lr
                
        print("Model succesfully loaded from checkpoint")
    else:
        start_epoch = 0

    scheduler_generator = lr_scheduler.StepLR(optimizer_generator, step_size=3334, gamma=0.5, last_epoch = start_epoch - 1) # See LapSRN paper
    scheduler_discriminator = lr_scheduler.StepLR(optimizer_generator, step_size=3334, gamma=0.5, last_epoch = start_epoch - 1)

    # Set needed data transformations.
    CROP_SIZE = 128 # is 128 in paper
    hr_transform = get_hr_transform(CROP_SIZE, random=True)
    lr_transforms = [get_lr_transform(CROP_SIZE, factor, random=True) for factor in [2, 4]]

    # Load datasets and set transforms.
    dataset = Dataset(args.data_dir, hr_transform=hr_transform, lr_transforms=lr_transforms, verbose=True)
    train_dataset = SplitDataset(dataset, Split.TRAIN, 0.8) 
    validation_dataset = SplitDataset(dataset, Split.TEST, 0.8)
    train_data = data.DataLoader(dataset=train_dataset, num_workers=6,\
                                 batch_size=args.batch_size, shuffle=True, pin_memory=True)
    validation_data = data.DataLoader(dataset=validation_dataset, num_workers=6,\
                                      batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # Initialise logging
    writer = SummaryWriter(args.tensorboard_dir)

    for epoch in range(start_epoch, args.num_epochs+1):
        print("Started epoch num={}.".format(epoch))
        scheduler_generator.step()
        scheduler_discriminator.step()
        print('Learning rate is {:.7E}'.format(optimizer_generator.param_groups[0]['lr']))
        writer.add_scalar('hyper/lr', optimizer_generator.param_groups[0]['lr'], epoch)
        train(epoch, generator, gan, criterion, optimizer_generator, writer, train_data)

        validate_every = 67 # epochs
        if (epoch % validate_every) == 0 or (epoch == args.num_epochs):
            cum_psnr = validate(epoch, generator, gan, criterion, writer, validation_data)

        if (epoch % args.checkpoint_every) == 0 or (epoch == args.num_epochs):
            checkpoint_name = str(Path(args.checkpoint_dir) / 'srn_{}.pt'.format(epoch))
            print("Wrote checkpoint {}!".format(checkpoint_name))
            save_checkpoint(epoch, generator, discriminator, optimizer_generator, optimizer_discriminator, checkpoint_name, args.adversarial)

if __name__ == '__main__':
    main()
