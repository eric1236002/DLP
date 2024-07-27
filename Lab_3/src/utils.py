import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def dice_score(pred, target):
    '''
    2 * (number of common pixels) / (predicted img size + groud truth img size)
    pred 形狀: [batch_size, num_classes, height, width]
    target 形狀: [batch_size, height, width]
    但是輸入是[batch_size,1, height, width]所以可以直接做flatten
    '''
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    common_pixels=(pred_flat * target_flat).sum()
    total = pred_flat.sum() + target_flat.sum()
    
    dice = (2. * common_pixels) / (total)
    return dice


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        '''
        pred 形狀: [batch_size, num_classes, height, width]
        target 形狀: [batch_size, height, width]        
        '''

        # 將 pred 轉換為概率
        pred = F.softmax(pred, dim=1)
        
        # 將 target 轉換為 one-hot 編碼
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        # 計算 Dice coefficient
        intersection = (pred * target_one_hot).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()