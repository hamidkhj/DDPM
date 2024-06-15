import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset import MnistDataset
from torch.utils.data import DataLoader
from Unet import Unet
from noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



im_path = 'data/train/images'


num_timesteps  = 1000
beta_start  = 0.0001
beta_end  = 0.02


im_channels  = 1
im_size  = 28
down_channels  = [32, 64, 128, 256]
mid_channels  = [256, 256, 128]
down_sample  = [True, True, False]
time_emb_dim  = 128
num_down_layers  = 2
num_mid_layers  = 2
num_up_layers  = 2
num_heads  = 4


task_name = 'default'
batch_size = 64
num_epochs = 40
num_samples  = 100
num_grid_rows  = 10
lr = 0.0001
ckpt_name = 'ddpm_ckpt.pth'


def train(num_epochs = 40):
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=num_timesteps,
                                     beta_start=beta_start,
                                     beta_end=beta_end)
    
    # Create the dataset
    mnist = MnistDataset('train', im_path=im_path)
    mnist_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Instantiate the model
    model = Unet(
                im_channels,
                down_channels,
                mid_channels,
                time_emb_dim,
                down_sample,
                num_down_layers,
                num_mid_layers,
                num_up_layers
    ).to(device)
    model.train()
    
    # Create output directories
    if not os.path.exists(task_name):
        os.mkdir(task_name)
    
    # Load checkpoint if found
    if os.path.exists(os.path.join(task_name,ckpt_name)):
        print('Loading checkpoint as found one')
        model.load_state_dict(torch.load(os.path.join(task_name,
                                                      ckpt_name), map_location=device))
    # Specify training parameters
    num_epochs = num_epochs
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    # Run training
    for epoch_idx in range(num_epochs):
        losses = []
        for im in tqdm(mnist_loader):
            optimizer.zero_grad()
            im = im.float().to(device)
            
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            
            # Sample timestep
            t = torch.randint(0, num_timesteps, (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses),
        ))
        torch.save(model.state_dict(), os.path.join(task_name,
                                                    ckpt_name))
    
    print('Done Training ...')
    

if __name__ == '__main__':
    train()