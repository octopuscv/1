import os
import cv2
import torch

def calPosition(x:int, image_size, patch_size):
    patch_num = (image_size // patch_size)**2
    patch_Row = image_size // patch_size
    tokenIndex = x // image_size
    templateIndex = x % image_size
    x_pred = tokenIndex % patch_Row * patch_size + templateIndex % patch_size
    y_pred = tokenIndex // patch_Row * patch_size + templateIndex // patch_size
    return tokenIndex, templateIndex, x_pred, y_pred

def getMeanStd(image, tokensize):
        std = 1 if image.std() == 0 else image.std()
        retImage = torch.zeros_like(image)
        for cIndex in range(retImage.shape[0]):
            for y in range(tokensize ):
                for x in range(tokensize ):
                    x_begin = x * tokensize 
                    x_end = (x+1) * tokensize 
                    y_begin = y * tokensize 
                    y_end = (y+1)* tokensize 
                    imagetoken = image[cIndex,x_begin:x_end,y_begin:y_end]
                    std = 1 if imagetoken.std() == 0 else imagetoken.std()

                    imagetoken = (imagetoken - imagetoken.mean()) / std
                    retImage[cIndex,x_begin:x_end,y_begin:y_end] = torch.as_tensor(imagetoken)
        return retImage

def pack_by_Frames(list, Frames):
    return [list[i:i+Frames] for i in range(0, len(list)-Frames)]

def filelist(args, root_path, FramesNumber):
        img_path = []  
        # single sequence
        if args.is_single_sequences == True:    
            datalist = [files for files in os.listdir(root_path)]
            datalist.sort(key=lambda x: int(x.split('.')[0]))
            datalist = [os.path.join(root_path,i) for i in datalist]
            datalist = [datalist[i : i+FramesNumber] for i in range(0,len(datalist)-FramesNumber)]
            for item in datalist:
                img_path.append(item)
        # multiple sequences
        else:
           for dirs in os.listdir(root_path):
                datalist = [files for files in os.listdir(os.path.join(root_path, dirs))]
                datalist.sort(key=lambda x: int(x.split('.')[0]))
                datalist = [os.path.join(root_path,dirs,i) for i in datalist]
                datalist = [datalist[i : i+FramesNumber] for i in range(0,len(datalist)-FramesNumber)]
                for item in datalist:
                    img_path.append(item)
        return img_path

def save_image(image_path, x_pred, y_pred):
        dirname,pngname = image_path.split('/')[-2:]
        dirpath = "anno_img/{}".format(dirname)
        os.makedirs(dirpath,exist_ok=True)
        src_img  = cv2.imread(image_path)
        cv2.circle(src_img,(x_pred,y_pred),2,(0,255,0),-1) 
        cv2.imwrite('{}/{}'.format(dirpath,pngname),src_img)  
