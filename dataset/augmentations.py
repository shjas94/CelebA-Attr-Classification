import torchvision.transforms as T

def get_transforms(img_size, mode='train'):
    if mode == 'train':
        return T.Compose([
            T.ToTensor(),
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30)
        ])
    elif mode == 'valid':
        return T.Compose([
        T.ToTensor(),
        T.Resize((img_size, img_size))
    ])
    elif mode == 'test':
        return T.Compose([
        T.ToTensor(),
        T.Resize((img_size, img_size))
    ])