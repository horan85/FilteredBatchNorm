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


def RunTest(BNType,Epoch,Repeat,BatchSize):
        transform = transforms.Compose(
             [transforms.Resize((224,224)) ,
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])
        
        if BNType=="Builtin":
                BNFunction=nn.BatchNorm2d   
        elif BNType=="Filtered": 
                BNFunction=batchnormlib.BN2dFitleredMoments
        elif BNType=="Ref": 
                BNFunction=BN2dRef
        elif BNType=="None": 
                BNFunction=[]
                
        
        
        # download and transform train dataset
        train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../cifar_data', download=True, train=True, transform=transform), 
        batch_size=BatchSize, shuffle=True,num_workers=4)

        # download and transform test dataset
        test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../cifar_data', download=True, train=False, transform=transform),
         batch_size=BatchSize, shuffle=True,num_workers=4)

        class CNNClassifier(nn.Module):  
            #ALexNet
            def __init__(self, num_classes=10):
                super(CNNClassifier, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                    BNFunction(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 192, kernel_size=5, padding=2),
                    BNFunction(192),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(192, 384, kernel_size=3, padding=1),
                    BNFunction(384),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(384, 256, kernel_size=3, padding=1),
                    BNFunction(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    BNFunction(256),
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
                return F.log_softmax(x,dim=-1)
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


        for repeat in range(0, Repeat):
            clf = CNNClassifier().cuda()

            #for p in clf.parameters():
            #    print(p.shape)
            opt = optim.SGD(clf.parameters(), lr=0.01, momentum=0.5)
            #opt = optim.Adam(clf.parameters(), lr=0.01)
            
            train_loss_history.append([])
            train_acc_history.append([])
            test_loss_history.append([])
            test_acc_history.append([])
            for epoch in range(0, Epoch):
                print("Epoch %d" % epoch)
                train(epoch)
            
        #torch.save(clf.state_dict(), "MyNetMnist")
        np.save(BNType+"_train_loss.npy",np.array(train_loss_history))
        np.save(BNType+"_train_acc.npy",np.array(train_acc_history))
        np.save(BNType+"_test_loss.npy",np.array(test_loss_history))
        np.save(BNType+"_test_acc.npy",np.array(test_acc_history))


#print("Test Filtered")
#RunTest("Filtered",25,10,128)
print("Test Builtin")
RunTest("Builtin",25,10,128)
