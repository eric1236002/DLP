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

def dice_score(pred_mask, gt_mask):
    '''
    2 * (number of common pixels) / (predicted img size + groud truth img size)
    '''
    pred_flat = pred_mask.view(-1)
    target_flat = gt_mask.view(-1)
    
    common_pixels=(pred_flat * target_flat).sum()
    total = pred_flat.sum() + target_flat.sum()
    
    dice = (2. * common_pixels) / (total)
    return dice

def dice_loss(pred_mask, gt_mask):
    return 1 - dice_score(pred_mask, gt_mask)
