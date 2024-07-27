import argparse
import evaluate
import torch
import models.unet
import models.resnet34_unet
import matplotlib.pyplot as plt

def inferece(args):
    if args.model_base=='resnet34_unet':
        model=models.resnet34_unet.ResNet34_UNet(channels=3,classes=3)
    else:
        model=models.unet.UNet(channels=3,classes=3)
    model.load_state_dict(torch.load(args.model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    dice,results=evaluate.evaluate(model,args.data_path,device,args.num_samples)
    print(f'Test Dice Score: {dice:.4f}')
    if args.num_samples is not None:
        visualize_results(results,args.num_samples)

def visualize_results(results, num_samples=5):
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    for i, result in enumerate(results[:num_samples]):
        image = result['image'].squeeze().transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())
        true_mask = result['true_mask'].squeeze()
        pred_mask = result['pred_mask'].squeeze()

        axes[i, 0].imshow(image)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(image)
        axes[i, 1].imshow(true_mask, alpha=0.5, cmap='jet')
        axes[i, 1].set_title('True Mask')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(image)
        axes[i, 2].imshow(pred_mask, alpha=0.5, cmap='jet')
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('segmentation_results.png')
    plt.show()

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='D:\Cloud\DLP\Lab_3\saved_models\88re.pth', help='path to the stored model weoght')
    parser.add_argument('--model_base', default='resnet34_unet', help='base of the model')
    parser.add_argument('--data_path',default="Lab_3\dataset\oxford-iiit-pet", type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--num_samples', '-n', type=int, default=None, help='number of samples to visualize')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    inferece(args)
