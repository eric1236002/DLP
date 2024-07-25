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
