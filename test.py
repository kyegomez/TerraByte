import random
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch 
from TerraByte import MEGABYTE

transform = ToTensor()
cifar_train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)


def random_image_input():
    index = random.randint(0, len(cifar_train_data) - 1)
    image, _ = cifar_train_data[index]
    image = image.view(-1, 3, 32, 32) #reshaping the image tensor to (batch_size, 3, 32, 32)
    return image.unsqueeze(0)


model = MEGABYTE(
    num_tokens=16000, #number of tokens
    dim = (512, 512), #transform model dimension (512 for coarsest, 256 for fine in this example)
    max_seq_len= (1024, 4), #sequence length for global and then local this can be more than 2
    depth = (6, 4), #number of layers for global and then local. this can be more than 2, but length must match the max seq len
    heads = 8, #number of attention heads
    flash_attn=True #use flash attention
)


x = torch.randint(0, 16000, (1, 1024, 4))

loss = model(x, modality=0, return_loss = True)
# loss.backward()


# x_text = torch.randint(0, 16000, (1, 1024, 4))
# x_image = random_image_input()

# # select the first element along the batch dimension
# x_text = x_text[0]

# loss_text = model(x_text, modality=0, return_loss=True)
# loss_text.backward()

# loss_image = model(x_image, modality=1, return_loss=True)
# loss_image.backward()

# modality = 0
# loss = model(x_text, modality, return_loss=True)
back = loss.backward()
print(f"back: {back}g")
#then after training


# 
# #provide the modality argument(0 for text, 1 for image)
# modality = 0

# loss = model(x_text, modality, return_loss=True)
# back = loss.backward()
# print(f"back: {back}")

#then after training
# logits = model(x)


#and sample from logits
#or use the generate function
# sampled = model.generate(temperature=0.9, filter_thres=0.9)#(1, 1024, 4)



