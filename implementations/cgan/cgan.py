'''
    Libraries
'''
import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from torch import optim

import random as rand

import seaborn as sb

import pandas as pd
import scipy as sp
from scipy import interpolate

from matplotlib import pyplot as plt

from parcels_v2 import *
from u_v import *
from data_pre_processing import *
from loss_L2 import *
from metrics import *

cuda = True if torch.cuda.is_available() else False


'''
    Args parse from command line
'''
#from datetime import datetime, timedelta
parser = argparse.ArgumentParser(description='Run Mercator simulation at N days')

parser.add_argument("epochs", help="specify number of epochs", type=int)
parser.add_argument("lr", help="sepcify learning rate", type=float)
parser.add_argument("batch_size", help="specify number of batches", type=int)
parser.add_argument("days", help="specify number of days", type=int)
parser.add_argument("N", help="specify number of trajectories", type=int)


parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")

args = parser.parse_args()

'''
    Init parameters
'''
batch_size = args.batch_size
w_entropy = 1
w_L2 = 0
loss = nn.BCELoss()
N=args.N
days=args.batch_size.days
steps=5*days
lr = args.lr
num_epochs = args.epochs
delta_t = 6 * 60 * 60
mu = 0
std = 1

'''
    Gaussian noise generation ####!!!! To do outside of generator class
'''
def gen_z(steps, mu, std):
    mu=torch.Tensor((mu,mu)) # Mean of noise. We defined a 2d mean to be generic
    std=torch.Tensor((std,std)) # Std of noise. We defined a 2d std to be generic
    z=mu+std*torch.randn(steps,2) # Gaussian noise generation
    return z

'''
    Data init
'''
data_train = prepare_data('data_train_1_day.csv', N)

# We initialize 1d lengths of drifter for positions and velocities
data_train_1day = np.zeros([data_train.shape[0]//steps, steps, 2])

'''
    Divide trajectories one by one
'''
for i in range(0, data_train.shape[0]//steps):  
    data_train_1day[i] = data_train[i*steps:i*steps+steps]
    #data_pos_1day[i] = data_pos[i*steps:i*steps+steps]

'''
    Init data loader, learning rate, number of epochs
'''
train_loader = torch.utils.data.DataLoader(data_train_1day, batch_size=batch_size, shuffle=True)

'''
    Save loss at each epoch to show it's evolution
'''
loss_D = num_epochs * [0.]
loss_G = num_epochs * [0.]

'''
    Init real and fake samples labels
'''
real_samples_labels = torch.ones((batch_size, 1))
generated_samples_labels = torch.zeros((batch_size, 1))

'''
    Define model
    Discriminator - CNN structure, LeakyReLU activation function and Sigmoid at the output to transform it to the probability space
'''
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(in_channels = 5, out_channels = 16, kernel_size = 2), # !!!! Attention now in_channels here is 5!!!!
                nn.LeakyReLU(),
                nn.Dropout(0.3),
                '''nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 1),
                nn.LeakyReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 1),
                nn.LeakyReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(in_channels = 64, out_channels = 32, kernel_size = 1),
                nn.LeakyReLU(),
                nn.Conv1d(in_channels = 32, out_channels = 2, kernel_size = 1), 
                nn.LeakyReLU(),
                nn.Conv1d(in_channels = 2, out_channels = 1, kernel_size = 1), 
                nn.LeakyReLU(),'''
                nn.Sigmoid()
            )
            
    def forward(self, inp, condition):
        output = self.model(inp)
        return output

'''
    Define model
    Generator - LSTM (RNN) structure, each cell outputs a velocity for the next time step (delta_t = 6 hours) 
    Input condition - GLORYS12 model velocity field at a given position (updated after time step delta_t)
'''
class Generator(nn.Module):
    def __init__(self, Ndays, batch_size):
        super(Generator, self).__init__()
        self.batch_size = batch_size
        # The size of the LSTM inputs is 4 : condition (u,v) and noise (z1,z2). The size of the LSTM output is 2 (u_corr,v_corr)
        self.lstmcell = nn.LSTMCell(4, 2)
        # The size of the linear inputs and outputs is 2 : (u_corr,v_corr)
        self.linear = nn.Linear(2, 2)
        self.days = Ndays
        

    def forward(self, inp, z):
        # We initialize the outputs
        outputs = []
        hx = torch.zeros(1, 2, dtype=torch.float) # Maybe it is better to initialize these values randomly. For the moment we let them as 0.
        cx = torch.zeros(1, 2, dtype=torch.float)
        '''generated_u = self.days * steps * [0.]
        generated_v = self.days * steps * [0.]'''
        generated_TS = torch.zeros((self.batch_size, steps, 2))
        parcels_TS = torch.zeros((self.batch_size, steps, 2))
        print('generated TS init :')
        print(generated_TS)
        
        '''
            Iterate through the batch  ###!!!! Some lines below should be moved out of the generator class
        '''
        for b in range(0, self.batch_size):
            date_start = str(inp.loc[b*steps, 'date'])[:10].split('-')
            date_end = str(inp.loc[b*steps + steps-1, 'date'])[:10].split('-')
            start_lon =  inp.loc[b*steps, 'lon']
            start_lat = inp.loc[b*steps, 'lat']
            
            # Initalize lon_step_i and lat_step_i
            lon_step_i=np.zeros((steps+1))
            lat_step_i=np.zeros((steps+1))
            
            '''
                Iterate through day steps (5 positions, velocities etc)
            '''    
            for i in range(0, steps):
                lon_step_i[i] = start_lon
                lat_step_i[i] = start_lat
                #Call Parcels simulation tool for batch b to get U and V at lon_0 and lat_0 (start point of CMEMS drifter) and given start date
                lon_traj_model, lat_traj_model = parcels(int(date_start[0]), int(date_start[1]), int(date_start[2]), int(date_end[0]), int(date_end[1]), int(date_end[2]), lon_step_i[i], lat_step_i[i])
                # We obtain the i+1 position of Parcels to later obtain the velocities at time 0
                lon_step_i[i+1] = lon_traj_model[:, i + 1]
                lat_step_i[i+1] = lat_traj_model[:, i + 1]
                # We obtain the velocities at time i zero from Parcels
                u, v = get_Ui_Vi(lon_step_i[i], lat_step_i[i], lon_step_i[i+1], lat_step_i[i+1], delta_t)
                # We define the input (condition + Noise) of the LSTM. size 4
                lstm_input = torch.Tensor([u[0], v[0],z[i,0], z[i,1]])
                hx, cx = self.lstmcell(lstm_input, (hx, cx))
                #save generated time series
                generated_TS[b][i] = [hx.detach().numpy()[0][0], hx.detach().numpy()[0][1]]
                #save parcels velocities to put the corresponding condition to the discriminator
                parcels_TS[b][i] = [u, v]
                #update lon and lat, new generated particle position
                start_lon, start_lat = data.loc[b*5, 'lon'] + delta_t * u_corr / 1000 * (1/111),\ 
                data.loc[b*5, 'lat'] + delta_t * v_corr / 1000 * (1/111)
                output = self.linear(hx)
                outputs += [output]            
        return generated_TS, parcels_TS
    
#Init G and D for optimizers
discriminator = Discriminator()
generator = Generator(1, batch_size)

#Init optimizers
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

#Cuda
if cuda:
    generator.cuda()
    discriminator.cuda()
    loss.cuda()

writer = SummaryWriter()
    
'''
    Train
'''
for epoch in range(num_epochs):

    print('*******************************************')
    print('Epoch '+str(epoch))
    print('*******************************************')
    for batch, real_samples in enumerate(train_loader):
        inp = data_train_1day[batch*batch_size*5 : batch*batch_size*5 + batch_size*4].reset_index(drop=True)
        #inp_u_v = inp_generated[batch*batch_size : batch*batch_size + batch_size]
        # we use the generator de define a first set of trajectories before starting the training
        generated_samples = generator(inp, gen_z(steps, mu, std))
        generated_samples = torch.Tensor(generated_samples[0]).double()
        condition_parcels_generated = torch.Tensor(generated_samples[1]).double()
        condition_parcels_real = torch.Tensor(data_cmems.loc[batch*batch_size*5 : batch*batch_size*5 + batch_size*4, 've':'vn'].to_numpy())
       
        # Discriminator training
        optimizer_discriminator.zero_grad() 
        out_discr_real = discriminator(real_samples.float(), condition_parcels_real) 
        D_x = out_discr_real.mean() # D_x is initialized as zero, pas initializer ou l'initializer dans la boucle n
        print('D_x = '+str(D_x))
        out_discr_fake = discriminator(generated_samples.float(), condition_parcels_generated)
        D_G_z = out_discr_fake.mean()
        print('D_G_z = '+str(D_G_z))
        #out_discr_fake = torch.unsqueeze(torch.squeeze(out_discr_fake), dim = 1) ## Netoyer
        loss_discriminator = -(loss(out_discr_real, real_samples_labels) + loss(out_discr_fake, generated_samples_labels))
        loss_G_G_z = -loss(out_discr_fake, generated_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()
        
        '''print('********************************')
        print('Generator training')
        print('********************************')'''
        # Generator training
        generator = Generator(1, batch_size)
        optimizer_generator.zero_grad()
        generated_samples = generator(inp, gen_z(steps, mu, std))
        generated_samples = torch.Tensor(generated_samples[0]).double()
        condition_parcels_generated = torch.Tensor(generated_samples[1]).double()
        output_discriminator_generated = discriminator(generated_samples.float(), condition_parcels_generated)
        loss_generator = -w_entropy * loss(output_discriminator_generated, real_samples_labels) #+ w_L2 * L2()
        loss_generator.backward()
        optimizer_generator.step()
        
    # Output value of loss function
    if (epoch % 10 == 0):
        metrics = metrics()
        rmse = metrics[0]
        print(f"Epoch: {epoch} Loss D.: {loss_discriminator} Loss D_G_z : {loss_G_G_z} D_x : {D_x} D_G_z : {D_G_z}")
        print(f"Epoch: {epoch} Loss G.: {loss_generator}")
        print('RMSE between v_generated and v_cmems : {rmse}')
        loss_D[epoch] = loss_discriminator
        loss_G[epoch] = loss_generator
        '''plt.plot(loss_D)
        plt.plot(loss_G)'''
