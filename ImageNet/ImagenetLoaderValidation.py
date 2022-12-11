from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from skimage import io
import numpy as np
import time
from PIL import Image


        
        

IMG_SIZE = (224,224)

class ImageNetDataset(Dataset):
    def __init__(self, data_path, is_train, train_split = 0.99, random_seed = 42, target_transform = None, num_classes = None):
        super(ImageNetDataset, self).__init__()
        self.data_path = data_path

        self.is_classes_limited = False

        valdata_path='./imagenet/imagenet_tmp/raw_data/validation'

        traindata_path='./imagenet/imagenet_tmp/raw_data/train'

        ClassList=os.listdir(traindata_path)
        print(ClassList[0])
        
        ImgList=sorted(os.listdir(valdata_path))

        self.ImgClasses=[]
        self.ImgNames=[]
        
        with open('./imagenet/imagenet_tmp/raw_data/imagenet_2012_validation_synset_labels.txt', 'r') as content_file:
            content = content_file.read().split()
            Ind=0
            for c in content:
                print(c)
                print(ClassList.index(c))
                self.ImgClasses.append(ClassList.index(c))
                self.ImgNames.append(valdata_path+"/"+ImgList[Ind])
                Ind+=1

    def __len__(self):
        return len(self.ImgNames)

    def __getitem__(self, index):

        img_name = self.ImgNames[index]
        img_class = self.ImgClasses[index]

        img = Image.open(img_name)

        if img.mode == 'L':
            tr = transforms.Grayscale(num_output_channels=3)
            img = tr(img)

        #tr = transforms.ToTensor()
        #img1 = tr(img)

        #width, height = img.size
        #if min(width, height)>IMG_SIZE[0] * 1.5:
        #    tr = transforms.Resize(int(IMG_SIZE[0] * 1.5))
        #    img = tr(img)

        #width, height = img.size
        #if min(width, height)<IMG_SIZE[0]:
        #    tr = transforms.Resize(IMG_SIZE)
        #    img = tr(img)

        #tr = transforms.RandomCrop(IMG_SIZE)
        #img = tr(img)
        tr = transforms.Resize(IMG_SIZE)
        img = tr(img)

        tr = transforms.ToTensor()
        img = tr(img)
        if (img.shape[0] != 3):
            img = img[0:3]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        img = normalize(img)
        
        return [img, img_class]
        #return dict(image = img, cls = img_info['cls']['class_idx'], class_name = img_info['cls']['class_name'])

    def get_number_of_classes(self):
        return 1000


