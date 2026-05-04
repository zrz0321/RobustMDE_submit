from external.Marigold.src.util.metric import threshold_percentage

def delta1_acc_times_minus1(pred, gt, valid_mask):
    return -1 * threshold_percentage(pred, gt, 1.25, valid_mask)