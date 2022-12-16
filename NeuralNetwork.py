import math
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
import os
import ast
import numpy as np
import pywt
import matplotlib.pyplot as plt
from PIL import Image as PImage
device="cuda"
def weights_init(m):
    if isinstance(m, nn.Linear):
      torch.nn.init.kaiming_uniform_(m.weight)
class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()

        # Encoder
        self.encode=nn.Sequential(
        nn.Linear(22,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720)
        )
        self.fc_mu = nn.Linear(720, z_dim)
        self.fc_logvar = nn.Linear(720, z_dim)
        # Decoder
        self.decode=nn.Sequential(
        nn.Linear(z_dim, 720),
        nn.ReLU(),
        nn.Linear(720, 720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,22)
        )
        nn.init.kaiming_uniform_(self.fc_mu.weight)
        nn.init.kaiming_uniform_(self.fc_logvar.weight)
        self.encode.apply(weights_init)
        self.decode.apply(weights_init)

        
    def encoder(self, x):
        a=self.encode(x)
        return self.fc_mu(a),self.fc_logvar(a)     

    def sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample

    def decoder(self, z):
        return self.decode(z)
       
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.sampling(mu, logvar)
        return self.decoder(z), mu, logvar
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(20, 1200)
        nn.init.kaiming_uniform_(self.layer_1.weight)
        self.layer_2 = nn.Linear(1200, 1200)
        nn.init.kaiming_uniform_(self.layer_2.weight)
        self.layer_3 = nn.Linear(1200, 1200)
        nn.init.kaiming_uniform_(self.layer_3.weight)
        self.layer_4 = nn.Linear(1200, 1200)
        nn.init.kaiming_uniform_(self.layer_4.weight)
        self.layer_5 = nn.Linear(1200, 1200)
        nn.init.kaiming_uniform_(self.layer_5.weight)
        self.layer_6 = nn.Linear(1200, 1200)
        nn.init.kaiming_uniform_(self.layer_6.weight)
        self.layer_7 = nn.Linear(1200, 1200)
        nn.init.kaiming_uniform_(self.layer_7.weight)
        self.layer_8 = nn.Linear(1200, 1200)
        nn.init.kaiming_uniform_(self.layer_8.weight)
        self.layer_9 = nn.Linear(1200, 1200)
        nn.init.kaiming_uniform_(self.layer_9.weight)
        self.layer_10 = nn.Linear(1200, 6)
        nn.init.kaiming_uniform_(self.layer_10.weight)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):

        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.relu(self.layer_4(x))
        x = self.relu(self.layer_5(x))
        x = self.relu(self.layer_6(x))
        x = self.relu(self.layer_7(x))
        x = self.relu(self.layer_8(x))
        x = self.relu(self.layer_9(x))
        x = self.tanh(self.layer_10(x))

        return x
def coords(m):
    for i in range(2, len(m) - 3):
        if m[i] == ',':
            return m[2:i],m[(i+2):(len(m)-2)]
trainloss=0
testloss=0
vae = VAE(z_dim=20).double().to(device)
myfinaltrainloss=[]
myfinaltestloss=[]
vae.load_state_dict(torch.load("AbhayBhaskarFourierVAEWeights", map_location=torch.device('cpu')))
loss_function2=nn.MSELoss()
model=NeuralNetwork().double().to(device)   
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
batch_size=128
with open('x_y.txt', 'r') as f: 
    lines = f.readlines()
for epoch in range (200):
    print("Epoch number " + str(epoch))
    count=0
    trainloss=0
    testloss=0
    itertrain=0
    itertest=0
    runningnum=0
    for line in lines:
        count+=1
        x, y = line.split('=')[0], line.split('=')[1]
        a=line.split('=')
        joint1=a[2]
        joint2=a[3]
        joint3=a[4]
        joint4=a[5]
        joint5=a[6]
        x1,y1=coords(joint1)
        x2,y2=coords(joint2)
        x3,y3=coords(joint3)
        x4,y4=coords(joint4)
        x5,y5=coords(joint5)
        x, y = x.split(' '), y.split(' ')
        x = [i for i in x if i]
        y = [i for i in y if i]
        x[0] = x[0][1:]
        y[0] = y[0][1:]
        x[-1] = x[-1][:-1]
        y[-1] = y[-1][:-1]

        x = [float(i) for i in x if i]
        y = [float(i) for i in y if i]
        if(math.isnan(x[30])):
                continue
        S=np.zeros(360, dtype='complex_')
        i=0
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=359
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=1
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=358
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=2
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=357
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=3
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=356
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=4
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=355
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=5
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        input_list=[float(np.real(S[355])),float(np.real(S[356])),float(np.real(S[357])),float(np.real(S[358])), float(np.real(S[359])), float(np.real(S[0])), float(np.real(S[1])),float(np.real(S[2])),float(np.real(S[3])),float(np.real(S[4])),float(np.real(S[5])), float(np.imag(S[355])),float(np.imag(S[356])),float(np.imag(S[357])), float(np.imag(S[358])), float(np.imag(S[359])), float(np.imag(S[0])), float(np.imag(S[1])), float(np.imag(S[2])), float(np.imag(S[3])),float(np.imag(S[4])),float(np.imag(S[5]))]
        input_tensor=torch.tensor(input_list)
        latent_vector=vae.encode(input_tensor)
        myvector2=torch.tensor(latent_vector[0])
        prediction=model(myvector2)     
        output_list=[float(x2),float(x4),float(x5),float(y2),float(y4),float(y5)]
        output_tensor=torch.tensor(output_list)
        loss_function=nn.MSELoss()
        loss=loss_function(prediction,output_tensor)
        if(count==batch_size and runningnum<39168):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count=0
            itertrain+=1
            trainloss+=loss
            myloss=loss.item()
            print(myloss)
        elif(count==batch_size and runningnum>=39168):
            count=0
            testloss+=loss
            itertest+=1
    myfinaltrainloss.append(trainloss/itertrain)
    myfinaltestloss.append(testloss/itertrain)
print(myfinaltrainloss)
print(myfinaltestloss)
            
