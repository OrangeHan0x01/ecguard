#神经网络
import torch
import torch.nn as nn
import torch.nn.functional as F


class Ecg2dnet(nn.Module):
	def __init__(self):
		super(Ecg2dnet,self).__init__()
		self.conv1 = nn.Conv2d(in_channels=2,out_channels=6,kernel_size=(3,5),stride=1)
		self.bn1=nn.BatchNorm2d(6)
		self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
		self.conv2 = nn.Conv2d(in_channels=6,out_channels=24,kernel_size=(3,5),stride=1)
		self.bn2=nn.BatchNorm2d(24)
		self.pool2 = nn.MaxPool2d(kernel_size=(1,3),stride=2)#[16, 1, 19]
		self.conv3 = nn.Conv2d(in_channels=24,out_channels=4,kernel_size=(1,5),stride=1)#[4, 1, 15]
		self.bn3=nn.BatchNorm2d(4)
		self.pool3 = nn.MaxPool2d(kernel_size=(1,3),stride=2)#4, 1, 7
		self.fc1 = nn.Linear(4*7,2)
	def forward(self,x):
		x = self.pool1(F.relu(self.bn1(self.conv1(x))))#使用bn层归一化和缓解梯度爆炸问题
		x = self.pool2(F.relu(self.bn2(self.conv2(x))))
		x = self.pool3(F.relu(self.bn3(self.conv3(x))))
		x=x.view(-1,4*7)
		x=F.relu(self.fc1(x))
		#x=F.relu(self.fc2(x))
		return x

