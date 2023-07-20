from TerraByte.model.model import DilatedMegabyte

import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
PRIME_LEN = 100
SEQ_LEN = 50000

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

#vision dataset
transform = ToTensor()
cifar_train_data = CIFAR10(root="./data", train=True, download=True,transform=transform)
cifar_val_data = CIFAR10(root='./data', train=False, download=True, transform=transform)


class MultiModalDataset(Dataset):
    def __init__(self, text_data, image_data, seq_len):
        super().__init__()
        self.text_data = text_data
        self.image_data = image_data
        self.seq_len = seq_len

    def __getitem__(self, index):
        #randomly select a modality (0 for text and 1 for image)
        modality = random.choice([0, 1])

        if modality == 0:
            rand_start = torch.randint(0, self.text_data.size(0) - self.seq_len, (1,))
            full_seq = self.text_data[rand_start: rand_start + self.seq_len].long()
            return full_seq.cuda(), modality
        else:
            #image
            image = self.image_data[index]
            #image preprocessing code
            return image.cuda(), modality
    
    def __len__(self):
        return min(len(self.text_data) // self.seq_len, len(self.image_data))

# instantiate GPT-like decoder model

model = DilatedMegabyte(
    num_tokens = 256,
    dim = (768, 512, 256),
    depth = (6, 4, 2),
    max_seq_len = (512, 4, 4),
    flash_attn = True
).cuda()

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    x = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    train_x, valid_x = np.split(x, [int(90e6)])
    data_train, data_val = map(torch.from_numpy, (train_x, valid_x))

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = MultiModalDataset(data_train, cifar_train_data, SEQ_LEN)
val_dataset   = MultiModalDataset(data_val, cifar_val_data, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        inputs, modalities = next(train_loader)
        loss = model(inputs, modalities, return_loss=True)
        loss.backward()

    print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            inputs, modalities = next(val_loader)
            loss = model(inputs, modalities, return_loss=True)
            print(f'validation loss: {loss.item()}')

    if i != 0 and i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime_inp = inp[:PRIME_LEN]
        prime = decode_tokens(prime_inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        sample = model.generate(prime_inp[None, :])
        sample = sample.flatten(1)

        output_str = decode_tokens(sample[0][PRIME_LEN:])
        print(output_str)
