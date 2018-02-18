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

def calc_gradient_penalty(critic, real_data, fake_data):
    LAMBDA = 10 # standard value from paper

    batch_size = real_data.size(0)
    # One number between 0 and 1 for each pair (real, fake)
    alpha = torch.rand(batch_size, 1).cuda()
    alpha = alpha.expand(batch_size,real_data.nelement()// batch_size).contiguous().view(*real_data.shape
    )
    
    # Linear interpolation between real and fake data with random weights
    interpolate = alpha * real_data + ((1 - alpha) * fake_data)
    interpolate = Variable(interpolate, requires_grad=True).cuda()
    
    critic_interpolate = critic(interpolate)
    
    ones = torch.ones(critic_interpolate.size()).cuda()    
    
    # Retain graph needed for higher derivatives
    grad = autograd.grad(outputs=critic_interpolate, inputs=interpolate,
                        grad_outputs=ones, create_graph=True, retain_graph=True,
                        only_inputs=True)[0]
    
    # We want the gradient for each image of batch individually.
    grad = grad.view(grad.size(0), -1)
    grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return grad_penalty
    

def update_critic(generator, critic, optimizer_critic, lr, hr4):       
    NUM_CRITIC_ITERS = 5
    cum_critic_loss = 0.0
    
    # Freeze generator here
    input_lr = Variable(lr.data, volatile=True)

    # Compute fake data
    _, hr4_hat = generator(input_lr)
    # Remove volatile from output
    hr4_hat = Variable(hr4_hat.data)    

    for _ in range(NUM_CRITIC_ITERS):
        critic.zero_grad()

        # Train with real data (= hr4)
        critic_real = critic(hr4).mean()

        # Train with fake data (= hr4_hat)
        critic_fake  = critic(hr4_hat).mean()

        # Train with gradient penalty
        gradient_penalty = calc_gradient_penalty(critic, 
                                                 real_data=hr4.data,
                                                 fake_data=hr4_hat.data)

        wasserstein_distance = critic_real - critic_fake
        critic_loss = -wasserstein_distance + gradient_penalty
        cum_critic_loss += critic_loss        

        critic_loss.backward()
        optimizer_critic.step()
    
    return cum_critic_loss/NUM_CRITIC_ITERS
    
def get_adversarial_loss(critic, hr4_hat):
    # Freeze weights of critic for efficiency
    for p in critic.parameters():
        p.requires_grad = False
    
    adversarial_loss = -critic(hr4_hat).mean()
    
    # Unfreeze weights of critic
    for p in critic.parameters():
        p.requires_grad = True
        
    return adversarial_loss


def train(epoch, generator, discriminator, criterion, optimizer_generator, optimizer_disc, writer, train_data):
    use_adversarial = discriminator is not None
    
    cum_image_loss = 0.0
    cum_loss = 0.0
    if use_adversarial:
        ADVERSARIAL_WEIGHT = 0.01
        cum_critic_loss = 0.0
        cum_adversarial_loss = 0.0
        
    for it, batch in enumerate(train_data):
        lr, hr2, hr4 = [Variable(b).cuda(async=True) for b in batch]        

        if use_adversarial:
            # Update critic network
            critic_loss = update_critic(generator=generator, critic=discriminator, 
                                        optimizer_critic=optimizer_disc,
                                        lr=lr,
                                        hr4=hr4)           
            cum_critic_loss += critic_loss.data[0]

        optimizer_generator.zero_grad()
        hr2_hat, hr4_hat = generator(lr)
        
        # Compute pixel-wise/perceptual loss for both output imgs.
        loss_hr2 = criterion(hr2_hat, hr2)
        loss_hr4 = criterion(hr4_hat, hr4)

        if use_adversarial:
            adversarial_loss = ADVERSARIAL_WEIGHT * get_adversarial_loss(discriminator, hr4_hat)
            cum_adversarial_loss += adversarial_loss.data[0]
        else:
            adversarial_loss = 0.0
        image_loss = loss_hr2 + loss_hr4
        cum_image_loss += image_loss.data[0]

        #loss = image_loss + adversarial_loss
        loss = image_loss + adversarial_loss
        cum_loss += loss.data[0]
        
        loss.backward()
        optimizer_generator.step()
        
        if ((it + 1)% (len(train_data)//10) == 0):
            print("Epoch={}, Batch={}/{}, Avg. loss = {}".format(epoch, it + 1, len(train_data),
                                                                 cum_loss/(it+1)))
    print("Epoch={}, image loss={}, total Loss={}".format(epoch, cum_image_loss, cum_loss))
    writer.add_scalar('data/image_loss', cum_image_loss, epoch)
    writer.add_scalar('data/total_loss', cum_loss, epoch)

    if use_adversarial:
        print("Negative critic loss = {}, Adversarial loss={}".format(-cum_critic_loss, cum_adversarial_loss))
        writer.add_scalar('data/neg_critic_loss', -cum_critic_loss, epoch)
        writer.add_scalar('data/adversarial_loss', cum_adversarial_loss, epoch)
 
def validate(epoch, model, writer, validation_data):
    cum_psnr2, cum_psnr4 = 0.0, 0.0
    
    for batch in validation_data:
        lr, hr2, hr4 = [Variable(b).cuda(async=True) for b in batch]
        hr2_hat, hr4_hat = model(lr)
        
        mse = nn.MSELoss().cuda()
        error_1 = mse(hr2_hat, hr2)
        error_2 = mse(hr4_hat, hr4)
        
        get_psnr = lambda e: -10 * np.log10(e.data[0])
        cum_psnr2 += get_psnr(error_1)
        cum_psnr4 += get_psnr(error_2)

    print("Avg. PSNR: {}, {}.".format(cum_psnr2/len(validation_data),cum_psnr4/len(validation_data)))
    writer.add_scalar('data/validation_2', cum_psnr2/len(validation_data), epoch)
    writer.add_scalar('data/validation_4', cum_psnr4/len(validation_data), epoch)

   # Upscale one image for testing:
    lr, _hr2, _hr4 = validation_data.dataset[np.random.randint(0, len(validation_data.dataset))]
    out = model(Variable(lr).cuda().unsqueeze(0))
    for factor, img in enumerate(out):
        out = img.data.clone()
        # Remove normalisation from image.

        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

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
    
    generator = LapSRN(depth=args.depth).cuda().train()
    print(generator)
    if args.adversarial:
        discriminator = PatchD(use_sigmoid=False, num_layers=4).cuda().train()
        print(discriminator)

    # Paper uses SGD with LR=1e-5
    if args.adversarial:
        optimizer_generator = optim.Adam(generator.parameters(), betas=(0.0, 0.9), lr=args.lr)
        optimizer_disc = optim.Adam(discriminator.parameters(), betas=(0.0, 0.9), lr=args.lr)
    else:
        optimizer_generator = optim.SGD(generator.parameters(), weight_decay=1e-4, lr=args.lr, momentum=0.9)
        optimizer_generator = optim.Adam(generator.parameters(), weight_decay=1e-4, lr=args.lr)

    if args.perceptual:
        criterion = make_vgg16_loss(nn.MSELoss().cuda()).cuda()
    else:
        criterion = CharbonnierLoss().cuda()

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        generator.load_state_dict(checkpoint['model_state'])
        optimizer_generator.load_state_dict(checkpoint['optim_state'])
        if args.adversarial:
            optimizer_disc.load_state_dict(checkpoint['optim_state_optimizer'])
        start_epoch = checkpoint['epoch']
        print("Model succesfully loaded from checkpoint")
    else:
        start_epoch = 0

    scheduler = lr_scheduler.StepLR(optimizer_generator, step_size=3334, gamma=0.5, last_epoch = start_epoch - 1) # See paper

    # Set needed data transformations.
    CROP_SIZE = 128 # is 128 in paper
    hr_transform = get_hr_transform(CROP_SIZE, random=True)
    lr_transforms = [get_lr_transform(CROP_SIZE, factor, random=True) for factor in [2, 4]]

    # Load datasets and set transforms.
    dataset = Dataset(args.data_dir, hr_transform=hr_transform, lr_transforms=lr_transforms, verbose=True)
    train_dataset = SplitDataset(dataset, Split.TRAIN, 0.8) 
    validation_dataset = SplitDataset(dataset, Split.TEST, 0.8)
    train_data = data.DataLoader(dataset=train_dataset, num_workers=2,\
                                 batch_size=args.batch_size, shuffle=True, pin_memory=True)
    validation_data = data.DataLoader(dataset=validation_dataset, num_workers=2,\
                                      batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # Initialise logging
    writer = SummaryWriter(args.tensorboard_dir)

    for epoch in range(start_epoch, args.num_epochs+1):
        print("Started epoch num={}.".format(epoch))
        scheduler.step()
        print('Learning rate is {:.7E}'.format(optimizer_generator.param_groups[0]['lr']))
        writer.add_scalar('hyper/lr', optimizer_generator.param_groups[0]['lr'], epoch)
        train(epoch, generator, discriminator, criterion, optimizer_generator, optimizer_disc, writer, train_data)

        validate_every = 67 # epochs
        if (epoch % validate_every) == 0 or (epoch == args.num_epochs):
            cum_psnr = validate(epoch, generator, writer, validation_data)

        if (epoch % args.checkpoint_every) == 0 or (epoch == args.num_epochs):
            checkpoint_name = str(Path(args.checkpoint_dir) / 'srn_{}.pt'.format(epoch))
            print("Wrote checkpoint {}!".format(checkpoint_name))
            save_checkpoint(epoch, generator, discriminator, optimizer_generator, optimizer_disc, checkpoint_name, args.adversarial)

if __name__ == '__main__':
    main()
