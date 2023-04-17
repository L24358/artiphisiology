import deeplake
from torchvision import transforms

ds = deeplake.load("hub://activeloop/tiny-imagenet-train")

tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

deeplake_loader = ds.pytorch(num_workers=0, batch_size=4, transform={
                        'images': tform, 'labels': None}, shuffle=True)

for i, data in enumerate(deeplake_loader):
    images, labels = data['images'], data['labels']
    import pdb; pdb.set_trace()