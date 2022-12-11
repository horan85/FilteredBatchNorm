import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import batchnormlib

seed=42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

# download and transform train dataset
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', download=True, train=True, transform=transforms.Compose([
transforms.ToTensor(), # first, convert image to PyTorch tensor
transforms.Normalize((0.1307,), (0.3081,)) ])), 
batch_size=128, shuffle=True)

# download and transform test dataset
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', download=True, train=False,transform=transforms.Compose([
transforms.ToTensor(), # first, convert image to PyTorch tensor
transforms.Normalize((0.1307,), (0.3081,)) ])), 
 batch_size=128, shuffle=True)
 
  
   

class CNNClassifier(nn.Module):
    def __init__(self,BatchnormType="Builtin"):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, bias=False)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, bias=False)
        
        self.fc1 = nn.Linear(16*4*4, 84, bias=False)
        
        self.fc2 = nn.Linear(84, 10)
        
        if BatchnormType=="Builtin":
                self.bn_conv1 = nn.BatchNorm2d(6)
                self.bn_conv2 = nn.BatchNorm2d(16)
                self.bn_fc1 = nn.BatchNorm1d(84)
        elif BatchnormType=="Filtered":
                self.bn_fc1 = batchnormlib.BN1dFitleredMoments(84)
        elif BatchnormType=="Reimplemented":
                self.bn_conv1 = batchnormlib.BN2dReimplemented(6)
                self.bn_conv2 = batchnormlib.BN2dReimplemented(16)
                self.bn_fc1 = batchnormlib.BN1dReimplemented(84)
                
    def forward(self, x):
        x = F.relu(F.max_pool2d( self.bn_conv1(self.conv1(x)), 2))
        
        x = F.relu(F.max_pool2d( self.bn_conv2(self.conv2(x)), 2))

        x = x.view(-1, 16*4*4)
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        
        x = self.fc2(x)
        
        return F.log_softmax(x,dim=-1)

def train(epoch):
    
    for batch_id, (data, label) in enumerate(train_loader):
        clf.train()
        data=data.cuda()
        label=label.cuda()
        
        opt.zero_grad()
        preds = clf(data)
        loss = F.nll_loss(preds, label)
        loss.backward()
        opt.step()
        
        train_loss_history[-1].append(loss.item())
        predind = preds.data.max(1)[1] 
        acc = predind.eq(label.data).cpu().float().mean() 
        train_acc_history[-1].append(acc)
        
        if batch_id % 100 == 0:
            print("Train Loss: "+str(loss.item())+" Acc: "+str(acc.item()))

            #run independent test
            clf.eval() # set model in inference mode (need this because of dropout)
            test_loss = 0
            correct = 0
        
            for data, target in test_loader: 
                data=data.cuda()
                target=target.cuda()  
                with torch.no_grad():    
                   output = clf(data)
                   test_loss += F.nll_loss(output, target).item()
                   pred = output.data.max(1)[1] 
                   correct += pred.eq(target.data).cpu().sum()

            test_loss = test_loss
            test_loss /= len(test_loader) # loss function already averages over batch size
            accuracy =  correct.item() / len(test_loader.dataset)
            #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            #    test_loss, correct, len(test_loader.dataset),
            #    accuracy))
            test_acc_history[-1].append(accuracy)
            test_loss_history[-1].append(test_loss)
            print("Test Loss: "+str(test_loss)+" Acc: "+str(accuracy))

train_loss_history = []
train_acc_history = []
test_loss_history = []
test_acc_history = []


BatchnormType="Builtin"
#BatchnormType="Filtered"
BatchnormType="Reimplemented"

for repeat in range(0, 10):
    clf = CNNClassifier(BatchnormType)
    clf.cuda()
    opt = optim.SGD(clf.parameters(), lr=0.01, momentum=0.5)
    train_loss_history.append([])
    train_acc_history.append([])
    test_loss_history.append([])
    test_acc_history.append([])
    for epoch in range(0, 5):
        print("Epoch %d" % epoch)
        train(epoch)
    

np.save("res/"+BatchnormType+"_1.0_train_loss.npy",np.array(train_loss_history))
np.save("res/"+BatchnormType+"_1.0_train_acc.npy",np.array(train_acc_history))
np.save("res/"+BatchnormType+"_1.0_test_loss.npy",np.array(test_loss_history))
np.save("res/"+BatchnormType+"_1.0_test_acc.npy",np.array(test_acc_history))

