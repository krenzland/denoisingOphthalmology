import torch
from torch import autograd
from torch.autograd import Variable

class GAN(object):
    def __init__(self, discriminator, optimizer, adversarial_weight, use_wgan=True):
        self.discriminator = discriminator
        self.optimizer = optimizer
        self.adversarial_weight = adversarial_weight
        self.use_wgan = use_wgan
        self.critic_iterations = 5 if use_wgan else 1

    def get_gradient_penalty(self, real_data, fake_data):
        # Only for WGAN-GP!
        assert(self.use_wgan)
        
        LAMBDA = 10 # standard value from paper

        batch_size = real_data.size(0)
        # One number between 0 and 1 for each pair (real, fake)
        alpha = torch.rand(batch_size, 1).cuda()
        alpha = alpha.expand(batch_size,real_data.nelement()// batch_size).contiguous().view(*real_data.shape
        )

        # Linear interpolation between real and fake data with random weights
        interpolate = alpha * real_data + ((1 - alpha) * fake_data)
        interpolate = Variable(interpolate, requires_grad=True).cuda()

        critic_interpolate = self.discriminator(interpolate)

        ones = torch.ones(critic_interpolate.size()).cuda()    

        # Retain graph needed for higher derivatives
        grad = autograd.grad(outputs=critic_interpolate, inputs=interpolate,
                            grad_outputs=ones, create_graph=True, retain_graph=True,
                            only_inputs=True)[0]

        # We want the gradient for each image of batch individually.
        grad = grad.view(grad.size(0), -1)
        grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return grad_penalty

    def get_discriminator_loss(self, hr4, hr4_hat):
        # Train with real data (= hr4)
        discriminator_real = self.discriminator(hr4)

        # Train with fake data (= hr4_hat)
        discriminator_fake  = self.discriminator(hr4_hat)

        if self.use_wgan:
            # Train with gradient penalty
            gradient_penalty = self.get_gradient_penalty(real_data=hr4.data,
                                                          fake_data=hr4_hat.data)

            wasserstein_distance = discriminator_real.mean() - discriminator_fake.mean()
            discriminator_loss = -wasserstein_distance + gradient_penalty
        else:
            discriminator_loss = (-torch.log(discriminator_real) - torch.log(1 - discriminator_fake)).mean()

        return discriminator_loss

    def update(self, generator, lr, hr4):       
        # Unfreeze weights of critic
        for p in self.discriminator.parameters():
            p.requires_grad = True

        cum_critic_loss = 0.0

        # Freeze generator here
        input_lr = Variable(lr.data, volatile=True)

        # Compute fake data
        # TODO: Why compute it twice?
        hr4_hat = generator(input_lr)[-1]
        # Remove volatile from output
        hr4_hat = Variable(hr4_hat.data)    

        for _ in range(self.critic_iterations):
            self.discriminator.zero_grad()

            critic_loss = self.get_discriminator_loss(hr4=hr4, hr4_hat=hr4_hat)
            cum_critic_loss += critic_loss        

            critic_loss.backward()
            self.optimizer.step()

        return cum_critic_loss/self.critic_iterations

    def get_generator_loss(self, hr4_hat):
        # Freeze weights of critic for efficiency
        for p in self.discriminator.parameters():
            p.requires_grad = False

        if self.use_wgan:
            generator_loss = -self.discriminator(hr4_hat).mean()
        else:
            generator_loss = -torch.log(self.discriminator(hr4_hat)).mean()

        return self.adversarial_weight * generator_loss

   
