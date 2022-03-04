import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import math


class SRCNN(nn.Module):
    # https://keras.io/examples/vision/super_resolution_sub_pixel/
    # https://mfarahmand.medium.com/cnn-based-single-image-super-resolution-6ffcd39ec993
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
        
class SRCNN_extended(nn.Module):
    # https://keras.io/examples/vision/super_resolution_sub_pixel/
    # https://mfarahmand.medium.com/cnn-based-single-image-super-resolution-6ffcd39ec993
    def __init__(self, num_channels=3):
        super(SRCNN_extended, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=9, padding=9 // 2)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv5 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x
        
        
class DRRN(nn.Module):
    # https://github.com/jt827859032/DRRN-pytorch/blob/master/drrn.py
    # TODO: fix channels
	def __init__(self):
		super(DRRN, self).__init__()
		self.input = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False) # multiply channels by3?
		self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
		self.output = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
		self.relu = nn.ReLU(inplace=True)

		# weights initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))

	def forward(self, x):
		residual = x
		inputs = self.input(self.relu(x))
		out = inputs
		for _ in range(25):
			out = self.conv2(self.relu(self.conv1(self.relu(out))))
			out = torch.add(out, inputs)

		out = self.output(self.relu(out))
		out = torch.add(out, residual)
		return out
		
class DRRN_adapted(nn.Module):
    # https://github.com/jt827859032/DRRN-pytorch/blob/master/drrn.py
    # TODO: fix channels
	def __init__(self):
		super(DRRN_adapted, self).__init__()
		self.input = nn.Conv2d(in_channels=3, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False)# multiplied channels by3!
		self.conv1 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False)# multiplied channels by3!
		self.conv2 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False)# multiplied channels by3!
		self.output = nn.Conv2d(in_channels=384, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)# multiplied channels by3!
		self.relu = nn.ReLU(inplace=True)

		# weights initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))

	def forward(self, x):
		residual = x
		inputs = self.input(self.relu(x))
		out = inputs
		for _ in range(25):
			out = self.conv2(self.relu(self.conv1(self.relu(out))))
			out = torch.add(out, inputs)

		out = self.output(self.relu(out))
		out = torch.add(out, residual)
		return out
