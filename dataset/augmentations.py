import torchvision.transforms as T

def get_train_transforms(cfg):
    return T.Compose([
        T.ToTensor(),
        T.Resize((cfg.img_size, cfg.img_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(30)
    ])
    
def get_valid_transforms(cfg):
    return T.Compose([
        T.ToTensor(),
        T.Resize((cfg.img_size, cfg.img_size))
    ])
    
def get_test_transforms(cfg):
    return T.Compose([
        T.ToTensor(),
        T.Resize((cfg.img_size, cfg.img_size))
    ])