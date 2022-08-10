
import sys


n_chars = 42
seq_len = 14
batch_size=64
hidden = 1024
nz = 42
num_epochs= 120
LEAK = 0.2
LAMBDA = 10       # strength of gradient penalty

BATCH_SIZE = 64
ITER = 200000
C_ITERS = int(sys.argv[2])       # critic iterations
EG_ITERS = 1      # encoder / generator iterations     # strength of gradient penalty
LEARNING_RATE = float(sys.argv[1])
BETA1 = 0.5
BETA2 = 0.9
name_f = sys.argv[3]
from decoder_mc import v_mainchain, show_mutiply_structures,get_sidechain_angles,get_1d_mc_sd


from  mc_sc_to_residues  import lst_residue,c_pdb,rosetta_score
import pickle


def get_batch_r_score(g_data):
    score_lst = []
    for i in range(len(g_data)):

        data = g_data[i].detach().numpy()

        seq, mc, sd = get_1d_mc_sd(data)
#print(seq,mc,sd)
#lst_residue("mc_test_len14.pdb", sd_coords)

        c_pdb(seq,mc,sd,"../tmp/test_file"+name_f+"_Cov2_h1024.pdb")
        score =rosetta_score("../tmp/test_file"+name_f+"_Cov2_h1024.pdb")
        score_lst.append(score)
    return score_lst




import argparse
import os
import random
import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pickle

import torch.autograd as autograd
from torch.autograd import Variable
#from utils.torch_utils import *

from torch.optim import Adam
#from torch.utils import data
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, LeakyReLU, ReLU, Tanh
#from util import DeterministicConditional, GaussianConditional, JointCritic, WALI
from torchvision import datasets, transforms, utils



class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out

class ConvBlock(nn.Module):
    def __init__(self, hidden):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.conv_block(input)
        return input + (0.3*output)





class Generator_helix(nn.Module):
    def __init__(self, n_chars, seq_len, batch_size, hidden):
        super(Generator_helix, self).__init__()
        self.fc1 = nn.Linear(nz, hidden*seq_len)
        self.block = nn.Sequential(
            ConvBlock(hidden),
            ConvBlock(hidden),
           # ConvBlock(hidden),
           # ConvBlock(hidden),
           # ConvBlock(hidden),
        )
        self.conv1 = nn.Conv1d(hidden, n_chars, 1)
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden = hidden

    def forward(self, noise):
        output = self.fc1(noise)
        output = output.view(-1, self.hidden, self.seq_len) # (BATCH_SIZE, DIM, SEQ_LEN)
        output = self.block(output)
        output = self.conv1(output)
       # output = output.transpose(1, 2)
        shape = output.size()
        #output = output.contiguous()
        #output = output.view(self.batch_size*self.seq_len, -1)
        #output = gumbel_softmax(output, 0.5)
        #return output.view(shape) # (BATCH_SIZE, SEQ_LEN, len(charmap))
        return output

class Encoder_helix(nn.Module):
    def __init__(self, n_chars, seq_len, batch_size, hidden):
        super(Encoder_helix, self).__init__()
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden = hidden
        self.block = nn.Sequential(
            ConvBlock(hidden),
            ConvBlock(hidden),
           # ConvBlock(hidden),
           # ConvBlock(hidden),
           # ConvBlock(hidden),
        )
        self.conv1d = nn.Conv1d(n_chars, hidden, 1)
        self.linear = nn.Linear(self.seq_len*self.hidden, 42)

    def forward(self, input):
        #output = input.transpose(1, 2) # (BATCH_SIZE, len(charmap), SEQ_LEN)
        output = self.conv1d(input)
        output = self.block(output)
       # print(output.size())
        output = output.view(-1, self.seq_len*self.hidden)
        output = self.linear(output)
        #output = nn.Sigmoid()(output)
        return output

#A joint Wasserstein critic function.
#    Args:
#      x_mapping: An nn.Sequential module that processes x.
#      z_mapping: An nn.Sequential module that processes z.
#      joint_mapping: An nn.Sequential module that process the output of x_mapping and z_mapping.

class JointCritic(nn.Module):
    def __init__(self, x_mapping, z_mapping, joint_mapping):
        super().__init__()
        self.x_net = x_mapping
        self.z_net = z_mapping
        self.joint_net = joint_mapping

    def forward(self, x, z):
        assert x.size(0) == z.size(0)
        x_out = self.x_net(x)   #x_net takes in x which is data. this should be like discriminator
        z_out = self.z_net(z)   #
        joint_input = torch.cat((x_out, z_out), dim=1)
        output = self.joint_net(joint_input)
        return output

def create_critic():
    x_mapping = nn.Sequential(
    nn.Conv1d(n_chars, 2*hidden, 1),
   # LeakyReLU(LEAK),
    nn.Conv1d(2*hidden, 1*hidden, 1),
   # LeakyReLU(LEAK)
    )

    z_mapping = nn.Sequential(
    nn.Linear(nz, 2*hidden*seq_len),
    View([2*hidden, seq_len]),
    nn.Conv1d(2*hidden, 1*hidden, 1),
    LeakyReLU(LEAK)
    )

    joint_mapping = nn.Sequential(
    nn.Conv1d(1*hidden+1*hidden,1*hidden,1), LeakyReLU(LEAK),
    #nn.Conv1d(1*hidden, 1*hidden, 1), LeakyReLU(LEAK),
    ConvBlock(hidden),
    ConvBlock(hidden),
    nn.Conv1d(1*hidden, 1, 1))

    return JointCritic(x_mapping, z_mapping, joint_mapping)




cri = create_critic()




#     """ Adversarially learned inference (a.k.a. bi-directional GAN) with Wasserstein critic.
#     Args:
#       E: Encoder p(z|x).
#       G: Generator p(x|z).
#       C: Wasserstein critic function f(x, z).
#     """
class WALI(nn.Module):
    def __init__(self, E, G, C):
        super().__init__()
        self.E = E
        self.G = G
        self.C = C
    def get_encoder_parameters(self):
        return self.E.parameters()

    def get_generator_parameters(self):
        return self.G.parameters()

    def get_critic_parameters(self):
        return self.C.parameters()

    def encode(self, x):
        return self.E(x)

    def generate(self, z):
        return self.G(z)

    def reconstruct(self, x):
        return self.generate(self.encode(x))

    def criticize(self, x, z_hat, x_tilde, z):
        input_x = torch.cat((x, x_tilde), dim=0)
        input_z = torch.cat((z_hat, z), dim=0)
        output = self.C(input_x, input_z)
        data_preds, sample_preds = output[:x.size(0)], output[x.size(0):]
        return data_preds, sample_preds

    def calculate_grad_penalty(self, x, z_hat, x_tilde, z):
        bsize = x.size(0)
        eps = torch.rand(bsize, 1, 1).to(x.device) # eps ~ Unif[0, 1]
        intp_x = eps * x + (1 - eps) * x_tilde
       # print("intp_x",intp_x.size())
        eps_z = eps.view(bsize,1)
        intp_z = eps_z * z_hat + (1 - eps_z) * z
       # print("intp_z",intp_z.size())
        intp_x.requires_grad = True
        intp_z.requires_grad = True
        C_intp_loss = self.C(intp_x, intp_z).sum()
        grads = autograd.grad(C_intp_loss, (intp_x, intp_z), retain_graph=True, create_graph=True)
        grads_x, grads_z = grads[0].view(bsize, -1), grads[1].view(bsize, -1)
        grads = torch.cat((grads_x, grads_z), dim=1)
        grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty

    def forward(self, x, z, lamb=10):
        z_hat, x_tilde = self.encode(x), self.generate(z)
#        print("z_hat",z_hat.size())
#        print("x_tilde",x_tilde.size())
        data_preds, sample_preds = self.criticize(x, z_hat, x_tilde, z)
#        print("data_preds",data_preds.size())
#        print("sample_preds",sample_preds.size())
        EG_loss = torch.mean(data_preds - sample_preds)
#        print("x.data",x.data.size())
#        print("z_hat.data",z_hat.data.size())
#        print("x_tilde.data",x_tilde.data.size())
#        print("z.data",z.data.size())
        C_loss = -EG_loss + lamb * self.calculate_grad_penalty(x.data, z_hat.data, x_tilde.data, z.data)
        return C_loss, EG_loss


G = Generator_helix(n_chars, seq_len, batch_size, hidden)
E = Encoder_helix(n_chars, seq_len, batch_size, hidden)
C = create_critic()


wali = WALI(E, G, C)

# load dataset. Change data from numpy to tensor. use dataloader
workers = 1
with open('../../data/numpy_filter_rosettascore100_trainingdata_0512.pickle', 'rb') as f:
    np_encoded_a1_h_dataset_10_20 = pickle.load(f)

tensor_x_update = torch.from_numpy(np_encoded_a1_h_dataset_10_20[60000:])

tensor_x_update = tensor_x_update.permute([0,2,1])

my_dataset = TensorDataset(tensor_x_update) # create your datset
 # create your dataloader

dataloader = DataLoader(my_dataset, batch_size=batch_size,
                                         shuffle=True, drop_last=True,num_workers=workers)



def train_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    wali = WALI(E, G, C).to(device)

    optimizerEG = Adam(list(wali.get_encoder_parameters()) + list(wali.get_generator_parameters()),
    lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizerC = Adam(wali.get_critic_parameters(),
    lr=LEARNING_RATE, betas=(BETA1, BETA2))
    saved_data_lst = []

# dataloader

    fix_noise = torch.randn(500, 42, device=device)

    EG_losses, C_losses = [], []
    saved_score = []
    curr_iter = C_iter = EG_iter = 0
    C_update, EG_update = True, False
    print('Training starts...')
    j=0
    for epoch in range(num_epochs):
        for batch_idx, x in enumerate(dataloader, 0):
        #for i, data in enumerate(dataloader, 0):

            x = x[0].float().to(device)

            if curr_iter == 0:
                init_x = x
                curr_iter += 1

            z = torch.randn(x.size(0), 42).to(device)
            C_loss, EG_loss = wali(x, z, lamb=LAMBDA)

            if C_update:
                optimizerC.zero_grad()
                C_loss.backward()
                C_losses.append(C_loss.item())
                optimizerC.step()
                C_iter += 1

            if C_iter == C_ITERS:
                C_iter = 0
                C_update, EG_update = False, True
                continue

            if EG_update:
                optimizerEG.zero_grad()
                EG_loss.backward()
                EG_losses.append(EG_loss.item())
                optimizerEG.step()
                EG_iter += 1

            if EG_iter == EG_ITERS:
                EG_iter = 0
                C_update, EG_update = True, False
                curr_iter += 1
            else:
                continue
          # print training statistics
            if curr_iter % 100 == 0:
                print('[%d/%d%d]\tW-distance: %.4f\tC-loss: %.4f'
                  % (curr_iter, ITER,epoch, EG_loss.item(), C_loss.item()))

        j+=1
            # plot reconstructed images and samples

        real_x, rect_x = init_x[:32].detach().cpu(), wali.reconstruct(init_x[:32]).detach().cpu()
        saved_data = wali.generate(fix_noise).permute([0,2,1]).cpu()
        score_batch = get_batch_r_score(saved_data)
        saved_score.append(score_batch)
        pickle_out = open("../pickle_f/"+name_f+"/val_score.pickle","wb")
        pickle.dump(saved_score, pickle_out)
        saved_data_lst.append([saved_data,real_x, rect_x])
        pickle_out = open("../pickle_f/"+name_f+"/fixed_data_real_data_rect_data.pickle","wb")
        pickle.dump(saved_data_lst, pickle_out)
        pickle_out.close()
        pickle_out_G = open("../pickle_f/"+name_f+"/EG_losses.pickle","wb")
        pickle.dump(EG_losses,pickle_out_G)
        pickle_out_G.close()
        pickle_out_D = open("../pickle_f/"+name_f+"/C_losses.pickle","wb")
        pickle.dump(C_losses,pickle_out_D)
        pickle_out_D.close()
        torch.save(wali.state_dict(), '../saved_models/'+name_f+'/%d.ckpt' %j)


train_model()
