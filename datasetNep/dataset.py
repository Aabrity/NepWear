import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class NepVTONDataset(Dataset):
    def __init__(self, data_root, pair_file='pair.txt', transform=None):
        self.data_root = data_root
        self.pair_path = os.path.join(data_root, pair_file)
        self.transform = transform

        with open(self.pair_path, 'r') as f:
            self.pairs = [line.strip().split() for line in f.readlines()]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        person_name, cloth_name = self.pairs[idx]
        person_path = os.path.join(self.data_root, 'image', person_name)
        cloth_path = os.path.join(self.data_root, 'cloth', cloth_name)

        person = Image.open(person_path).convert('RGB')
        cloth = Image.open(cloth_path).convert('RGB')

        if self.transform:
            person = self.transform(person)
            cloth = self.transform(cloth)

        return {
            'person': person,
            'cloth': cloth,
            'person_name': person_name,
            'cloth_name': cloth_name,
        }
