import torch
import argparse
from PIL import Image
from torchvision import transforms
from models.model import SiamVIT
from tqdm import tqdm
from utils import calPosition, getMeanStd, filelist,save_image

parse = argparse.ArgumentParser()
parse.add_argument('--image_path', type = str, default='./data/test_seq',help = "image path")
parse.add_argument('--image_size' , type = int, default = 256, help = "image_size")
parse.add_argument('--patch_size', type = int, default = 16, help = "set patch_size")
parse.add_argument('--batch_size', type = int, default = 16, help = "batch_patch_size")
parse.add_argument('--num_workers', type = int, default = 8, help = "num_workers")
parse.add_argument('--Frames', type = int, default = 1, help = "select the number of frames")
parse.add_argument('--lr', type=float, default = 1e-5, help = "learning rate")
parse.add_argument('--is_single_sequences', type=bool, default = True, help = "single/multiple sequences")
args = parse.parse_args(args=[])

device = 'cuda' if torch.cuda.is_available else 'cpu'
model = SiamVIT(image_size=256, template_size=args.patch_size, in_dim = 3 * args.patch_size**2, embed_dim= 3 * args.patch_size**2,
                num_heads = 8, deepth=6, in_c = 3 * args.Frames, patch_size = args.patch_size, mlp_radio = 4, Frame = args.Frames
                ).to(device)
transform = transforms.Compose([
    transforms.ToTensor(),])

with torch.no_grad():
    model.eval()      # Don't need to track gradents for validation
    checkpoint = torch.load("best_model_frame{}_num_head8.pth".format(args.Frames), map_location="cuda")
    model.load_state_dict(checkpoint['m'])
    datalist = filelist(args, args.image_path, args.Frames)
    for _ in tqdm(datalist):
        input = getMeanStd(torch.cat([transform(Image.open(item)) for item in _]), args.patch_size)
        res,attn = model(input.unsqueeze(dim = 0).cuda())
        tokenIndex, templateIndex, x_pred, y_pred = calPosition(int(attn.argmax()), args.image_size, args.patch_size)
        save_image(_[0], x_pred, y_pred)           
    print('Finished value')