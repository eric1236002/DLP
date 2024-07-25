import torch
import argparse
import oxford_pet
import utils
from tqdm import tqdm
def evaluate(net, data, device):
    net.eval()
    dataset = oxford_pet.load_dataset(data,mode='test')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # num_samples=2
    # if num_samples is not None:
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    #     dataloader = list(dataloader)[:num_samples]

    dice_scores = []
    for i, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
            image = sample['image'].to(device).float()
            mask = sample['mask'].to(device).long().squeeze(1)
            with torch.no_grad():
                pred_mask = net(image)
            pred_mask = torch.argmax(pred_mask, dim=1)
            dice = utils.dice_score(pred_mask, mask)
            dice_scores.append(dice)
        
    return sum(dice_scores) / len(dice_scores) if dice_scores else 0.0

    

