import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset,DataLoader


class DataProcess(Dataset):
    def __init__(self, args, root_path, ano_path, Frames, transform = None):
        super(DataProcess,self).__init__()
        self.img_path = self.filelist(args, root_path, Frames)
        self.ano_path = ano_path
        self.transform = transform
        self.tokensize = args.patch_size
        self.imagesize = args.image_size
        self.tokenRawTotal = self.imagesize // self.tokensize

    def filelist(args, root_path, FramesNumber):
        img_path = []  
        # single sequence
        if args.is_single_sequences == True:
            for dirs in os.listdir(root_path):
                datalist = [files for files in os.listdir(os.path.join(root_path, dirs))]
                datalist.sort(key=lambda x: int(x.split('.')[0]))
                datalist = [os.path.join(root_path,dirs,i) for i in datalist]
                datalist = [datalist[i : i+FramesNumber] for i in range(0,len(datalist)-FramesNumber)]
                for item in datalist:
                    img_path.append(item)

        # multiple sequences
        else:
            datalist = [files for files in os.listdir(root_path)]
            datalist.sort(key=lambda x: int(x.split('.')[0]))
            datalist = [os.path.join(root_path,i) for i in datalist]
            datalist = [datalist[i : i+FramesNumber] for i in range(0,len(datalist)-FramesNumber)]
            for item in datalist:
                img_path.append(item)

        return img_path

    def xmllist(self,root_path,file_type=('xml')):
        return [os.path.join(root_path,f) for root,dirs,files in os.walk(root_path) for f in files if f.endswith(file_type)]
        
    def __len__(self):
        return len(self.img_path)                      

    def __getitem__(self, index):
        img_file = self.img_path[index]
        xml_file = os.path.join(self.ano_path,'/'.join(img_file[-1].split('/')[-2:]).replace('.bmp','.xml'))
        assert os.path.exists(xml_file),"{} file not found!".format(xml_file)
        xy_centers = []
        tokenIndex = []
        xyPostionIndex = []
        xy_center = self.get_center(xml_file)
        for i in xy_center:
         xy_centers.append(i)
        tokenIndex.append(self.calTokenIndex(xy_center))
        xy_centers = torch.as_tensor(xy_centers, dtype=torch.float32)
        xyPostionIndex.append(self.calPostionIndex(xy_center))
        tokenIndex = torch.as_tensor(tokenIndex, dtype=torch.int64)
        xyPostionIndex = torch.as_tensor(xyPostionIndex, dtype=torch.int64)
        targets = {}
        targets['label'] = tokenIndex * 256 + xyPostionIndex
        targets['center'] = xy_centers
        targets['tokenIndex'] = tokenIndex
        targets['xyPostionIndex'] = xyPostionIndex
        targets['img_path'] = img_file
        img = torch.cat([self.transform(Image.open(item)) for item in img_file])
        img = self.getMeanStd(img)
        return img,targets
    
    def get_center(self,xml_file):
        assert True
        tree = ET.parse(xml_file)
        root = tree.getroot()
        xy_center_list = []
        for tag in root:
            if tag.tag == "object":
                for b_tag in tag:
                    if b_tag.tag == "centerbox":
                        x_center = int(b_tag.findtext("x"))
                        y_center = int(b_tag.findtext("y"))
                        xy_center_list.append((x_center, y_center))
        return xy_center_list 

    def calPostionIndex(self,location):
        x,y = int(location[0][0]),int(location[0][1])
        xPostionIndex = x % self.tokensize 
        yPostionIndex = y % self.tokensize
        xyPosintionIndex = yPostionIndex * self.tokensize + xPostionIndex
        return xyPosintionIndex
        
    
    def calTokenIndex(self,location):
        x,y = int(location[0][0]),int(location[0][1])
        x_index = x // self.tokensize 
        y_index = y // self.tokensize 
        tokenIndex = self.tokenRawTotal * y_index + x_index
        return tokenIndex

    def getMeanStd(self,image):
        std = 1 if image.std() == 0 else image.std()
        retImage = torch.zeros_like(image)
        for cIndex in range(retImage.shape[0]):
            for y in range(self.tokensize ):
                for x in range(self.tokensize ):
                    x_begin = x * self.tokensize 
                    x_end = (x+1) * self.tokensize 
                    y_begin = y * self.tokensize 
                    y_end = (y+1)* self.tokensize 
                    imagetoken = image[cIndex,x_begin:x_end,y_begin:y_end]
                    std = 1 if imagetoken.std() == 0 else imagetoken.std()

                    imagetoken = (imagetoken - imagetoken.mean()) / std
                    retImage[cIndex,x_begin:x_end,y_begin:y_end] = torch.as_tensor(imagetoken)
        return retImage

def buildDataLoader(args,train_path,Frames,ano_path,transform):
    dataset = DataProcess(args, train_path, ano_path=ano_path, Frames=Frames, transform=transform)
    trainloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    return trainloader