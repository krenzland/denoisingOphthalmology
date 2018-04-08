#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import time

# Training utilities
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils import data
from torch.autograd import Variable
from tensorboardX import SummaryWriter

# Models
from models.unet import UNet
from models.lap_srn import LapSRN
from models.patch_discriminator import PatchD

# Loss functions
from loss import CharbonnierLoss, CombinedLoss, SaliencyLoss, make_vgg16_loss, WeightedLoss
from gan import GAN

# Data handling
from dataset import Dataset, Split, SplitDataset
from augmentations import HrTransform, LrTransform

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
    
    cum_image_loss = Variable(torch.zeros(1).cuda()) #0.0
    cum_loss = Variable(torch.zeros(1).cuda())#0.0
    if use_adversarial:
        cum_critic_loss = 0.0
        cum_adversarial_loss = 0.0
        
    start = time.perf_counter()
    for it, batch in enumerate(train_data):
        # TODO: Only if using saliency loss.
        imgs, saliencies = batch
        lr, *ground_truth = [Variable(b).cuda(async=True) for b in imgs]
        saliencies = [Variable(s).cuda(async=True) for s in saliencies]

        if use_adversarial:
            # Update critic network
            critic_loss = gan.update(generator=generator,
                                     lr=lr,
                                     hr4=ground_truth[-1])           
            cum_critic_loss += critic_loss

        optimizer_generator.zero_grad()
        out = generator(lr)
        
        # Compute pixel-wise/perceptual loss for both output imgs.
        if len(saliencies) > 0:
            loss_hr = [criterion(a, b, saliency) for (a,b, saliency) in zip(out, ground_truth, saliencies)]
        else:
            loss_hr = [criterion(a, b) for (a,b) in zip(out, ground_truth)] 

        if use_adversarial:
            adversarial_loss = gan.get_generator_loss(hr4_hat=out[-1])
            cum_adversarial_loss += adversarial_loss
        else:
            adversarial_loss = 0.0

        image_loss = sum(loss_hr)
        cum_image_loss += image_loss

        loss = image_loss + adversarial_loss
        cum_loss += loss
        
        loss.backward()
        optimizer_generator.step()
        
        if ((it + 1)% (len(train_data)//10) == 0):
            print("Epoch={}, Batch={}/{}".format(epoch, it + 1, len(train_data)))
            
    end = time.perf_counter()
    print("Epoch took {:2.3f}s (train)".format(end-start))
    print("Epoch={}, image loss={}, total Loss={}".format(
        epoch,
        cum_image_loss.data[0]/len(train_data),
        cum_loss.data[0]/len(train_data)))
    writer.add_scalar('data/image_loss', cum_image_loss.data[0]/len(train_data), epoch)
    writer.add_scalar('data/total_loss', cum_loss.data[0]/len(train_data), epoch)

    if use_adversarial:
        print("Negative critic loss = {}, Adversarial loss={}".format(
            -cum_critic_loss.data[0]/len(train_data),
            cum_adversarial_loss.data[0]/len(train_data)))
        writer.add_scalar('data/neg_critic_loss', -cum_critic_loss.data[0]/len(train_data), epoch)
        writer.add_scalar('data/adversarial_loss', cum_adversarial_loss.data[0]/len(train_data), epoch)
 
def validate(epoch, generator, gan, criterion, writer, validation_data):
    use_adversarial = gan is not None
    
    # TODO: Adjust for unet denoising!
    cum_psnr = np.array([0.0, 0.0])
    cum_hr_loss = np.array([0.0, 0.0])

    if use_adversarial:
        cum_critic_loss = 0.0
    
    for batch in validation_data:
        # Need no gradient here!
        imgs, saliencies = batch
        lr, *ground_truth = [Variable(b, volatile=True).cuda(async=True) for b in imgs]
        saliencies = [Variable(s, volatile=True).cuda(async=True) for s in saliencies]

        out = generator(lr)
        
        mse = nn.MSELoss().cuda()

        mse_loss = np.array([mse(a, b).data[0] for (a,b) in zip(ground_truth, out)])
        
        get_psnr = lambda e: -10 * np.log10(e)
        cum_psnr += np.array([get_psnr(m) for m in mse_loss])

        # Compute pixel-wise/perceptual loss for both output imgs.
        if len(saliencies) > 0:
            cum_hr_loss += np.array([criterion(a,b, saliency).data[0] for (a,b, saliency)
                                 in zip(ground_truth, out, saliencies)])
        else:
            cum_hr_loss += np.array([criterion(a,b).data[0] for (a,b)
                                 in zip(ground_truth, out)])

        if use_adversarial:
            critic_loss = gan.get_discriminator_loss(hr4=ground_truth[-1], hr4_hat=out[-1])
            cum_critic_loss += critic_loss.data[0]


    print("Validation Avg. PSNR: {}, Validation Image Loss: {}.".format(
        cum_psnr/len(validation_data),
        cum_hr_loss/len(validation_data)
    ))

    if use_adversarial:
        print("Validation Neg. Critic Loss = {}".format(-cum_critic_loss))
        writer.add_scalar('data/validation_neg_critic_loss', -cum_critic_loss/len(validation_data), epoch)
        
    for i, (psnr, hr_loss) in enumerate(zip(cum_psnr, cum_hr_loss)):
        scale = 2**(i+1)
        writer.add_scalar('data/validation_psnr_{}'.format(scale), psnr/len(validation_data), epoch)
        writer.add_scalar('data/validation_image_loss_{}'.format(scale), hr_loss/len(validation_data), epoch)

    # Upscale one image for testing:
    imgs, _  = validation_data.dataset[np.random.randint(0, len(validation_data.dataset))]
    lr, *_ = imgs
    out = generator(Variable(lr, volatile=True).cuda().unsqueeze(0))
    for factor, img in enumerate(out):
        out = img.data.clone()

        # Remove normalisation from image.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Undo input[channel] = (input[channel] - mean[channel]) / std[channel]
        for t, m, s in zip(out, mean, std):
            t.mul_(s)
            t.add_(m)
        upscaled = out.cpu().clamp(0,1)
        writer.add_image('images/sr_{}'.format(2**(factor+1)), upscaled, epoch)

    return cum_psnr.sum()

def main():
    # Currently only support CUDA, not CPU backend..
    assert(torch.cuda.is_available)

    parser = argparse.ArgumentParser(description="Run LapSRN training.")
    parser.add_argument('--seed', type=int,
                        help="Value for random seed. Default: Random.")
    # ---------------------- Model settings ---------------------------------------
    parser.add_argument('--mode', type=str, default='sr', choices=['sr', 'denoise'],
                        help="Set operation mode, either upscaling (=sr) or deblurring (=denoise)")
    parser.add_argument('--model', type=str, default='LapSRN', choices=['LapSRN', 'UNet'],
                        help="Set model that should be used, choices are LapSRN and UNet (for denoising only). Default: LapSRN")
    parser.add_argument('--depth', type=int, default=10,
                        help="Set number of convolution layers for each feature extraction stage.")
    # ------------------ Optimizer settings ---------------------------------------
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Value for learning rate. Default: 1e-3")
    parser.add_argument('--batch-size', type=int, default=64,
                        help="Set batch size. Default: 32")
    parser.add_argument('--num-epochs', type=int, default=3333*3,
                        help="Set number of epochs. Default: 3333*4")
    # ------------------ Loss functions ------------------------------------------
    parser.add_argument('--mse', type=float, default=0.0,
                        help="Set weight of mse loss. Default=0.0")
    parser.add_argument('--l1', type=float, default=0.0,
                        help="Set weight of l1 loss. Default=0.0")
    parser.add_argument('--saliency', type=float, default=0.0,
                        help="Set weight of saliency loss. Default=0.0")
    parser.add_argument('--perceptual', type=float, default=0.0,
                        help="Set weight of perceptual loss. Default=0.0")
    parser.add_argument('--adversarial', type=float, default=0.0,
                        help="Set weight of adversarial loss. Default=0.0")
    parser.add_argument('--wgan', action='store_true', dest='use_wgan',
                        help="If true, use wgan-gp instead of standard GAN.")
    # ------------------ Directories ----------------------------------------------
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

    global args # todo: remove global?
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
    if args.mode == 'sr':
        assert(args.model == 'LapSRN')
        generator = LapSRN(depth=args.depth, upsample=True).cuda().train()
    else:
        if args.model == 'LapSRN':
            generator = LapSRN(depth=args.depth, upsample=False).cuda().train()
        else:
            generator = UNet(num_classes=3).cuda().train()

    print(generator)
    if args.adversarial > 0.0:
        if args.use_wgan:
            discriminator = PatchD(use_sigmoid=False, num_layers=4).cuda().train()
        else:
           discriminator = PatchD(use_sigmoid=True, num_layers=4).cuda().train() 
        print(discriminator)
    else:
        assert(not args.use_wgan)
        discriminator = None

    # Use data parallelism if more than one GPU.
    if torch.cuda.device_count() > 1:
        generator = nn.DataParallel(generator)
        if discriminator is not None:
            discriminator = nn.DataParallel(discriminator)

    # Setup optimizers
    if args.adversarial > 0:
        if args.use_wgan:
            optimizer_generator = optim.Adam(generator.parameters(), betas=(0.0, 0.9), lr=args.lr)
            optimizer_discriminator = optim.Adam(discriminator.parameters(), betas=(0.0, 0.9), lr=args.lr)
        else:
            #optimizer_generator = optim.Adam(generator.parameters(), weight_decay=1e-4, lr=args.lr)
            optimizer_generator = optim.Adam(generator.parameters(), betas=(0.5, 0.999), lr=args.lr)
            optimizer_discriminator = optim.Adam(discriminator.parameters(), weight_decay=0.0, lr=args.lr)
    else:
        optimizer_generator = optim.Adam(generator.parameters(), weight_decay=1e-4, lr=args.lr)
        optimizer_discriminator = None

    # Set up losses
    criterions = []
    if args.mse > 0.0:
        criterions.append(WeightedLoss(nn.MSELoss().cuda(), args.mse))
    if args.l1 > 0.0:
        criterions.append(WeightedLoss(CharbonnierLoss().cuda(), args.l1))
    if args.saliency > 0.0:
        criterions.append(WeightedLoss(SaliencyLoss().cuda(), args.saliency))
    if args.perceptual > 0.0:
        criterions.append(WeightedLoss(make_vgg16_loss(nn.MSELoss().cuda()).cuda(),
                                       args.perceptual))
                          
    # Combine all losses into one.
    #assert(len(criterions) > 0)
    print(criterions)
    criterion = CombinedLoss(criterions).cuda()

    # Handle adversarial loss seperately (backprob not trivial)
    if args.adversarial > 0.0:
        gan = GAN(discriminator=discriminator,
                  optimizer=optimizer_discriminator,
                  adversarial_weight=args.adversarial,
                  use_wgan=args.use_wgan)
    else:
        gan = None

    # Load from checkpoint.
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        generator.load_state_dict(checkpoint['model_state'])
        start_epoch = checkpoint['epoch']
        if args.adversarial:
            if 'optim_state_optimizer' in checkpoint:
                optimizer_generator.load_state_dict(checkpoint['optim_state'])
                optimizer_discriminator.load_state_dict(checkpoint['optim_state_optimizer'])
            else:
                print("Warning: Only loading generator from checkpoint!")
                print("Resetting epoch to 0 and learning rates to --lr argument.")
                print("Resetting optimizer for generator completely.")
                start_epoch = 0
                for param_group in optimizer_generator.param_groups:
                    param_group['lr'] = args.lr
        else:
            optimizer_generator.load_state_dict(checkpoint['optim_state'])
            
                
        print("Model succesfully loaded from checkpoint")
    else:
        start_epoch = 0

    # After 10e5 gradient updates lower LR once.
    scheduler_generator = lr_scheduler.StepLR(optimizer_generator, step_size=3333, gamma=0.1, last_epoch = start_epoch - 1)
    scheduler_discriminator = lr_scheduler.StepLR(optimizer_generator, step_size=3333, gamma=0.1, last_epoch = start_epoch - 1)

    # Set needed data transformations.
    CROP_SIZE = 128 # is 128 in paper
    hr_transform = HrTransform(CROP_SIZE, random=True)
    if args.mode == 'sr':
        resize_factors = [2, 4]
        max_blur = 2
    else:
        resize_factors = [1, 1] # no resizing here
        max_blur = 3
    if args.model == 'UNet':
        resize_factors = resize_factors[-1:] # TODO !

    print(resize_factors)
    lr_transform = LrTransform(crop_size=CROP_SIZE,
                               factors=resize_factors,
                               max_blur=max_blur)

    # Load datasets and set transforms.
    dataset = Dataset(args.data_dir,
                      hr_transform=hr_transform,
                      lr_transform=lr_transform,
                      use_saliency=True,
                      verbose=True)
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
        start = time.perf_counter()
        scheduler_generator.step()
        scheduler_discriminator.step()
        print('Learning rate is {:.7E}'.format(optimizer_generator.param_groups[0]['lr']))
        writer.add_scalar('hyper/lr', optimizer_generator.param_groups[0]['lr'], epoch)
        train(epoch, generator, gan, criterion, optimizer_generator, writer, train_data)

        validate_every = 10 # epochs
        if (epoch % validate_every) == 0 or (epoch == args.num_epochs):
            cum_psnr = validate(epoch, generator, gan, criterion, writer, validation_data)

        if (epoch % args.checkpoint_every) == 0 or (epoch == args.num_epochs):
            checkpoint_name = str(Path(args.checkpoint_dir) / 'srn_{}.pt'.format(epoch))
            print("Wrote checkpoint {}!".format(checkpoint_name))
            save_checkpoint(epoch, generator, discriminator, optimizer_generator, optimizer_discriminator, checkpoint_name, args.adversarial)

        end = time.perf_counter()
        print("Epoch took {:2.3f}s (total)".format(end-start))

if __name__ == '__main__':
    main()
