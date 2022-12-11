import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose(
     [transforms.Resize((224,224)) ,
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

# download and transform train dataset
train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../cifar_data', download=True, train=True, transform=transform), 
batch_size=128, shuffle=True,num_workers=4)

# download and transform test dataset
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../cifar_data', download=True, train=False, transform=transform),
 batch_size=128, shuffle=True,num_workers=4)
 
 
def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

class BN2double(nn.Module):
    def __init__(self, Channels, Thres=4.0  ):
        super(BN2double , self).__init__()
        self.ChannelNum=Channels
        self.beta=nn.Parameter(torch.tensor([0.0]*Channels), requires_grad=True).cuda()
        self.gamma=nn.Parameter(torch.tensor([1.0]*Channels), requires_grad=True).cuda()
        self.beta=self.beta.reshape(1,Channels,1,1)
        self.gamma=self.gamma.reshape(1,Channels,1,1)
        self.Thres=Thres
        
    def forward(self, xorig):
        x=xorig.permute([1,0,2,3])
        x=x.reshape((self.ChannelNum,-1))
        
        Mean=torch.mean(x, dim=-1).reshape((self.ChannelNum,1))
        Var=torch.var(x, dim=-1).reshape((self.ChannelNum,1))
        Mean=Mean.expand((self.ChannelNum,x.shape[1]))
        Var=Var.expand((self.ChannelNum,x.shape[1]))
        
        eps=1e-10
        normalized= (x-Mean)/torch.sqrt(Var+eps)
        
        Selected= ((normalized<self.Thres) * (normalized>-self.Thres)).float()
        #masked mean
        Mean=torch.sum(x*Selected, dim=[-1])/torch.sum(Selected,dim=[-1])
        
        Diff=(x - Mean.reshape((self.ChannelNum,1)).expand(self.ChannelNum,x.shape[1])  )**2
        
        Mean=Mean.reshape((1,self.ChannelNum,1,1))
        Mean=Mean.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
        
        
        Var= torch.sum(Diff*Selected , dim=[-1])/torch.sum(Selected,dim=[-1])
        Var=Var.reshape((1,self.ChannelNum,1,1))
        Var=Var.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
        

        eps=1e-20
        beta=self.beta.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
        gamma=self.gamma.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
        
        bn3= ((self.gamma*(xorig-Mean))/torch.sqrt(Var+eps)      )+self.beta
        #bn3= (((xorig-Mean))/(Var+eps)      )
        
        return bn3


class CNNClassifier(nn.Module):  
    #ALexNet
    def __init__(self, num_classes=10):
        super(CNNClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            BN2double(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            BN2double(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            BN2double(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            BN2double(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            #nn.BatchNorm2d(256),
            BN2double(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x)

# create classifier and optimizer objects
#clf = CNNClassifier()
#clf.cuda()


train_loss_history = []
train_acc_history = []
test_loss_history = []
test_acc_history = []

def train(epoch):
    
    for batch_id, (data, label) in enumerate(train_loader):
        clf.train()
        if data.shape[0]==128:
                data=data.cuda()
                label=label.cuda()
                opt.zero_grad()
                preds = clf(data)
                loss = F.nll_loss(preds, label)
                loss.backward()
                train_loss_history[-1].append(loss.item())
                opt.step()
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
                        if data.shape[0]==128:
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


for repeat in range(0, 10):
    clf = CNNClassifier()
    #clf.apply(init_weights)
    clf.cuda()
    opt = optim.SGD(clf.parameters(), lr=0.01, momentum=0.5)
    train_loss_history.append([])
    train_acc_history.append([])
    test_loss_history.append([])
    test_acc_history.append([])
    for epoch in range(0, 25):
        print("Epoch %d" % epoch)
        train(epoch)
    
#torch.save(clf.state_dict(), "MyNetMnist")
np.save("bndouble4_train_loss.npy",np.array(train_loss_history))
np.save("bndouble4_train_acc.npy",np.array(train_acc_history))
np.save("bndouble4_test_loss.npy",np.array(test_loss_history))
np.save("bndouble4_test_acc.npy",np.array(test_acc_history))
#plt.plot(train_loss_history)
#plt.plot(train_acc_history)
#plt.show()
