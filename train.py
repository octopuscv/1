import torch
import argparse
from numpy import *
from DataProcesser import buildDataLoader
from torchvision import transforms
from models.model import SiamVIT
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


parse = argparse.ArgumentParser()
parse.add_argument('--train_path' , type = str, default = './data/data12', help = "trainset path")
parse.add_argument('--ano_path', type=str, default='./Annotations',help = "xml files path" )
parse.add_argument('--epoch',  type = int,  default = 101,help = "epochs")
parse.add_argument('--image_size' , type = int, default = 256, help = "image_size")
parse.add_argument('--patch_size', type = int, default = 16, help = "set patch_size")
parse.add_argument('--Frames', type = int, default = 3, help = "select the number of frames")
parse.add_argument('--Interval_period', type = int, default = 5, help="model save interval period")
parse.add_argument('--lr', type=float, default = 1e-5, help = "learning rate")
parse.add_argument('--is_single_sequences', type=float, default = 1e-5, help = "single/multiple sequences")
args = parse.parse_args(args=[])

device = 'cuda' if torch.cuda.is_available else 'cpu'
TIMEMARK = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
writer = SummaryWriter("runs/dimOD_{}".format(TIMEMARK))
model = SiamVIT(image_size=256, template_size=args.patch_size, in_dim = 3 * args.patch_size**2, embed_dim= 3 * args.patch_size**2,
                num_heads = 8, deepth=6, in_c = 3 * args.Frames, patch_size = args.patch_size, mlp_radio = 4, Frame = args.Frames
                ).to(device)
transform = transforms.Compose([transforms.ToTensor()])
training_loader = buildDataLoader(args,train_path=args.train_path, Frames = args.Frames, ano_path = args.ano_path, transform = transform)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

model.train()
for epoch in tqdm(range(args.epoch)):  # loop over the dataset multiple times
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    for i,data in enumerate(training_loader, 0):
        total += 1
        inputs, labels = data
        inputs = inputs.cuda()
        labels_list = [labels[t].flatten(0).to(device) for t in labels 
                                if t == "tokenIndex"][0]
        xyPositionIndex_list = [labels[t].flatten(0).to(device) for t in labels if t == "xyPostionIndex"][0]
        label = [labels[i].flatten(0).to(device) for i in labels if i == "label"][0]
        outputs = model(inputs)
        loss = loss_fn(outputs[1].squeeze(), label)
        print("loss:" , loss.item())
        running_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss_step", loss, epoch * len(training_loader) + i) 
        writer.flush()  
        if epoch % args.Interval_period == 0:
            dictmodel = {"m": model.state_dict()}
            torch.save(dictmodel,"model_data_{}.pth".format(epoch))
    writer.add_scalar("loss_echo", running_loss/total, epoch)
    writer.flush()
print('Finished Training')

                   