from .focal_loss import FocalLoss

def make_loss():
    focal = FocalLoss()  # triplet loss
    def loss_func(dist, assign):
        focal(dist, assign)
    return loss_func