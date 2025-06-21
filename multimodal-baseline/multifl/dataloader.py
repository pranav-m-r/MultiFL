import torch
from torch.utils.data import Dataset
import PIL.Image
import os

class HatefulMemesDataset(Dataset):
    def __init__(self, dataset, tokenizer, transform=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Process image
        image = PIL.Image.open(os.path.join('../', item['img'])).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Process text
        text = item['text']
        text_encoding = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=100, 
            return_tensors='pt'
        )
        text_input = text_encoding['input_ids'].squeeze()
        
        # Label
        label = item['label']
        
        return {
            'image': image,
            'text': text_input,
            'label': torch.tensor(label, dtype=torch.long)
        }