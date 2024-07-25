import argparse
import evaluate
import torch
import models.unet
def inferece(args):
    model=models.unet.UNet(channels=3,classes=3)
    model.load_state_dict(torch.load(args.model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    dice=evaluate.evaluate(model,args.data_path,device)
    print(f'Test Dice Score: {dice:.4f}')
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='saved_models/07-25-10-13-26.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path',default="dataset\oxford-iiit-pet", type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    inferece(args)
