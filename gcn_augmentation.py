import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms

np.random.seed(0)


class Padding(torch.nn.Module):
    """blur a single image on CPU"""
    def __init__(self, ):
        super(Padding, self).__init__()

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()
        self.v, self.c, self.t = 50, 3, 64
    def forward(self, img):
        img = self.pil_to_tensor(img).squeeze()
        container = torch.zeros(self.t, self.v,self.c)
        #print(img.size())
        c,v,t = img.size()
        t_limit = min(64, t)
        v_limit = min(50, v)
        #print(img.size())
        container[:t_limit, :v_limit, :] = img.permute(2,1,0)[:t_limit, :v_limit, :]
        container = container.permute(2,1,0)
        #img = self.tensor_to_pil(container)
        return container