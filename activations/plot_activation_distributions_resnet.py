import torch
import cv2
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import ImagenetLoaderValidation
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.models.resnet as ResNet
from torch.hub import load_state_dict_from_url

data_path='./imagenet/imagenet_tmp/raw_data/validation'

dataset_val= ImagenetLoaderValidation.ImageNetDataset(data_path, is_train = False)

BATCH_SIZE =16

data_loader_val = DataLoader(dataset_val, BATCH_SIZE, num_workers=8)



#resnet18 = models.resnet18(pretrained=True)
#alexnet = models.alexnet(pretrained=True)
#squeezenet = models.squeezenet1_0(pretrained=True)
"""
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        image_modules = list(models.resnet18().children())[:-1] #all layer expect last layer
        self.modelA = nn.Sequential(*image_modules)
        
    def forward(self, image):
        a = self.modelA(image)
        x = F.sigmoid(x)
        return x
"""

respones=[]

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = ResNet.conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = ResNet.conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = ResNet.conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        respones.append(out.clone())
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        respones.append(out.clone())
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        respones.append(out.clone())

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = ResNet.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ResNet.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        respones.append(out.clone())
        
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        respones.append(out.clone())
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class MyModel(models.ResNet):
    def __init__(self,L):
        #super(MyModel, self).__init__( BasicBlock, [2,2,2,2])
        super(MyModel, self).__init__( Bottleneck, [3,4,23,3])#,groups=32,width_per_group=8)
        self.L=L
        
  
         
    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
       
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

LayerLists=[]

LayerLists=[1]
allresponses=[]
minval=[]
quant1=[]
quant25=[]
quant50=[]
quant75=[]
quant99=[]
maxval=[]
 
FirstRun=True
for L in LayerLists:
    model = MyModel(L).cuda()
    #state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-5c106cde.pth', progress=True)
    state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', progress=True) 
    model.load_state_dict(state_dict,strict=False)
    model.eval()
    print("NEw Layer")
    with torch.no_grad():
        Ind=0
        FirstRun=True
        for batch_id, batch in enumerate(data_loader_val):
                data=batch[0].cuda()
                label=batch[1].cuda()
                
                model(data)
                for ind,r in enumerate(respones):
                    if FirstRun: 
                        allresponses.append(r.cpu().detach().numpy())
                    else:
                        allresponses[ind]= np.concatenate((allresponses[ind] ,r.cpu().detach().numpy()),axis=0) 

                if FirstRun:
                        FirstRun=False
                respones=[]
                Ind+=1
                
                    
for a in allresponses: 
    a=np.transpose(a,(1,0,2,3))
    a=np.reshape(a,[a.shape[0],-1])  
    a=a-np.tile(np.expand_dims(np.mean(a,1),-1),[1,a.shape[1]])
    a=a/np.tile(np.expand_dims(np.std(a,1)+1e-12,-1),[1,a.shape[1]])
                               
    allr=np.reshape(a,[-1])
    length=allr.shape[-1]
    allr=np.sort(allr)
    minval.append(allr[0])
    quant1.append(allr[int(length*0.01)])
    quant25.append(allr[int(length*0.25)])
    quant50.append(allr[int(length*0.50)])
    quant75.append(allr[int(length*0.75)])
    quant99.append(allr[int(length*0.99)])
    maxval.append(allr[-1])                    
steps=range(len(minval))                    
plt.plot(minval,'r--')
plt.plot(quant1,'y')
plt.plot(quant25,'r')
plt.plot(quant50,'r')
plt.plot(quant75,'r')
plt.plot(quant99,'y')
plt.plot(maxval,'r--')
plt.grid()
#plt.vlines(12.5 ,colors='k', linestyles='solid')
plt.fill_between(steps, minval, maxval,
                alpha=0.1, color='r', label='All values')
plt.fill_between(steps, quant1, quant99,
                alpha=0.2, color='y', label='P1-P99')
plt.fill_between(steps, quant25, quant75,
                alpha=1.0, color='r', label='Q1-Q3')               
plt.legend(fontsize=20)
plt.title('Activation distributions on ResNext-101', fontsize=26)
plt.ylabel('Activations', fontsize=20)
plt.xlabel('Layers', fontsize=20)
#plt.savefig(os.path.join(figures_path, 'loss_landscape.png'), dpi=500, quality=100)
plt.show()
