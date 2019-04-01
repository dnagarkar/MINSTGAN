import torch
import torch.nn.functional as F
import torch.nn as nn


class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        # self.fc1 = nn.Linear(94*94, 64*64)
        self.conv_1 = nn.Conv2d(3, 128, 4, 2, padding = 1)
        # self.conv1_bn = nn.BatchNorm2d(128)

        self.conv_2 = nn.Conv2d(128, 256, 4, 2, padding = 1)
        self.conv2_bn = nn.BatchNorm2d(256)

        self.conv_3 = nn.Conv2d(256, 512, 4, 2, padding = 1)
        self.conv3_bn = nn.BatchNorm2d(512)

        self.conv_4 = nn.Conv2d(512, 1024, 4, 2, padding = 1)
        self.conv4_bn = nn.BatchNorm2d(1024)

        self.conv_5 = nn.Conv2d(1024, 1, 4, 1, padding = 0)
        # self.conv5_bn = nn.BatchNorm2d(1)

        self.leaky = torch.nn.LeakyReLU(0.2)

    def forward(self, x):

        m = nn.ReLU()

        x = self.conv_1(x)
        x = self.leaky(x)

        x = self.conv_2(x)
        x = self.leaky(x)
        x = self.conv2_bn(x)

        x = self.conv_3(x)
        x = self.leaky(x)
        x = self.conv3_bn(x)


        x = self.conv_4(x)
        x = self.leaky(x)
        x = self.conv4_bn(x)

        x = self.conv_5(x)
        # x = self.conv5_bn(x)
        # x = m(x)

        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim

        self.conv_1 = nn.ConvTranspose2d(noise_dim, 1024, 4, 1, padding = 0)
        self.conv1_bn = nn.BatchNorm2d(1024)

        self.conv_2 = nn.ConvTranspose2d(1024, 512, 4, 2, padding = 1)
        self.conv2_bn = nn.BatchNorm2d(512)

        self.conv_3 = nn.ConvTranspose2d(512, 256, 4, 2, padding = 1)
        self.conv3_bn = nn.BatchNorm2d(256)

        self.conv_4 = nn.ConvTranspose2d(256, 128, 4, 2, padding = 1)
        self.conv4_bn = nn.BatchNorm2d(128)

        self.conv_5 = nn.ConvTranspose2d(128, 3, 4, 2, padding = 1)
        # self.conv5_bn = nn.BatchNorm2d(3)

        self.relu = torch.nn.ReLU(0.2)

    def forward(self, x):

        x = x.view(-1, self.noise_dim, 1, 1)

        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv1_bn(x)

        x = self.conv_2(x)
        x = self.relu(x)
        x = self.conv2_bn(x)

        x = self.conv_3(x)
        x = self.relu(x)
        x = self.conv3_bn(x)

        x = self.conv_4(x)
        x = self.relu(x)
        x = self.conv4_bn(x)

        x = self.conv_5(x)

        x = torch.tanh(x)

        # print(x.size())
        # x = x.view(128, 3, 64, 64)
        return x
