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


class MyModel(nn.Module):
    def __init__(self,L):
        super(MyModel, self).__init__()
        image_modules = list(models.vgg19(pretrained=True).features)[:L] #first 22 layers
        self.modelA = nn.Sequential(*image_modules)
        
    def forward(self, image):
        x = self.modelA(image)
        return x

LayerLists=[]
image_modules=list(models.vgg19(pretrained=True).features)
for ind,m in enumerate(image_modules):
        print(m)
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                print(m)
                LayerLists.append(ind+1)
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                print(m)
                LayerLists.append(ind+1)
print(LayerLists)
allresponses=[]
minval=[]
quant1=[]
quant25=[]
quant50=[]
quant75=[]
quant99=[]
maxval=[]
 
for L in LayerLists:
    model = MyModel(L).cuda()
    model.eval()
    print("NEw Layer")
    with torch.no_grad():
        Ind=0
        FirstRun=True
        for batch_id, batch in enumerate(data_loader_val):
                data=batch[0].cuda()
                label=batch[1].cuda()
                
                respones=model(data)
                respones=respones.permute(1,0,2,3)
                #respones=respones.reshape(int(respones.shape[0]),-1)
                respones=respones.cpu().detach().numpy()
                if FirstRun:
                    allresponses=respones
                    FirstRun=False
                else:
                    allresponses=np.concatenate((allresponses, respones), axis=1)
                #respones=respones-np.tile(np.expand_dims(np.mean(respones,1),-1),[1,respones.shape[1]])
                #respones=respones/np.tile(np.expand_dims(np.std(respones,1),-1),[1,respones.shape[1]])
                Ind+=1
        allresponses=np.reshape(allresponses,[-1])
        print(allresponses.shape)
        length=allresponses.shape[-1]
        allresponses=np.sort(allresponses)
        minval.append(allresponses[0])
        quant1.append(allresponses[int(length*0.01)])
        quant25.append(allresponses[int(length*0.25)])
        quant50.append(allresponses[int(length*0.50)])
        quant75.append(allresponses[int(length*0.75)])
        quant99.append(allresponses[int(length*0.99)])
        maxval.append(allresponses[-1])
                
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
plt.title('Activation distributions VGG19', fontsize=26)
plt.ylabel('Activations', fontsize=20)
plt.xlabel('Layers', fontsize=20)
#plt.savefig(os.path.join(figures_path, 'loss_landscape.png'), dpi=500, quality=100)
plt.show()
