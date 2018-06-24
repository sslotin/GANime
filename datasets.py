import glob

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# maybe wrap default ImageFolder instead

class ImageDataset(Dataset):
    def __init__(self, root):
        self.files = glob.glob(root + '/*/*.jpg')
        # transforms?

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        img = img.resize((64, 64))
        img = transforms.ToTensor()(img)
        return img
    
    def __len__(self):
        return len(self.files)

FacesDataset = ImageDataset('faces')
AnimeDataset = ImageDataset('anime')
