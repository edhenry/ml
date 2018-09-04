import torch
import torchvision
from torchvision import transforms
import logging


class ColorDigit(object):
    def __init__(self, opt):

        self.opt = opt
        self.step_counter = 1

        dataset = torchvision.datasets.MNIST(root='./data',
                                             train=False,
                                             transform=transforms.ToTensor(),
                                             download=True)

    def load_digit(self):
    
    def reset(self):
    
    def get_action_range(self):
    
    def get_comm_limited(self):

    def get_reward(self):
    
    def step(self):
    
    def get_state(self):
