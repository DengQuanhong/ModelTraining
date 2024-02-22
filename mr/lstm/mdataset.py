import os
import torchtext
from tqdm import tqdm
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from datasets import load_from_disk
from torch.utils.data import DataLoader,Dataset
import pdb
from config import my_config
from torch.nn.utils.rnn import pad_sequence
import torch

class MyDataset(Dataset):
    def __init__(self, dataset,config):
        self.dataset = dataset
        self.tokenizer = get_tokenizer('basic_english')
        self.config = config
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        # pdb.set_trace()
        d = self.dataset[index]
        text,label = d['text'],d['label']
        text = self.tokenizer(text)
        x = [self.config.glove.stoi[w] for w in text if w in self.config.glove.stoi]
        x = torch.tensor(x)
        # x = GloVe.get_vecs_by_tokens(text)
        return x,label



def get_dataloader(config):
    def collate_fn(batch):
        text,label = zip(*batch)
        x_pad = pad_sequence(text,batch_first=True)
        label = torch.tensor(label)
        return x_pad,label
    dataset_hf = load_from_disk("/data/xuzhi/datasets/rotten_tomatoes_dataset")
    train_set = MyDataset(dataset_hf['train'],config) # 数据类实例化
    val_set = MyDataset(dataset_hf['validation'],config)
    test_set = MyDataset(dataset_hf['test'],config)
    
    train_loader = DataLoader(train_set,batch_size=config.batch_size,shuffle=True,collate_fn=collate_fn,num_workers=20)    
    val_loader = DataLoader(val_set,batch_size=config.batch_size,shuffle=True,collate_fn=collate_fn,num_workers=20)
    test_loader = DataLoader(test_set,batch_size=config.batch_size,shuffle=True,collate_fn=collate_fn,num_workers=20)
    
    return train_loader,val_loader,test_loader
if __name__=='__main__':
    ds = load_from_disk("/data/xuzhi/datasets/rotten_tomatoes_dataset")
    dataset = MyDataset(ds['train'])
    print(dataset[0])