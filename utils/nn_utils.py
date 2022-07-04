#数据集模块：

#创建自己的数据集：
import pickle
import numpy as np
import torch
import torch.nn as nn
from model import Ecg2dnet
from torch.utils.data import DataLoader

class MyDataset(torch.utils.data.Dataset): #创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self,datapath): #初始化一些需要传入的参数
        super(MyDataset,self).__init__()
        fx = open(datapath+'x.pcdat', 'rb')
        self.data_x = pickle.load(fx,encoding='iso-8859-1')#3d,要取出一个成2d
        fx.close()
        fy = open(datapath+'y.pcdat', 'rb')
        self.data_y = pickle.load(fy,encoding='iso-8859-1')
        fy.close()
 
    def __getitem__(self, index):
        fftx=np.array(np.abs(np.fft.fft(self.data_x[index])))[np.newaxis,:]
        fx1=np.array(self.data_x[index])[np.newaxis,:]
        fx=np.concatenate((fftx, fx1), axis=0)
        label = torch.tensor(self.data_y[index],dtype=torch.long)
        fx_t = torch.from_numpy(fx)#.unsqueeze(dim=0)
        fx_t = fx_t.type(torch.FloatTensor)
        return fx_t,label
 
    def __len__(self):
        return len(self.data_y)
 

class MyDataset2d(torch.utils.data.Dataset):#没有频域通道的数据集读取
    def __init__(self,datapath): #初始化一些需要传入的参数
        super(MyDataset2d,self).__init__()
        fx = open(datapath+'x.pcdat', 'rb')
        self.data_x = pickle.load(fx,encoding='iso-8859-1')#3d,要取出一个成2d
        fx.close()
        fy = open(datapath+'y.pcdat', 'rb')
        self.data_y =pickle.load(fy,encoding='iso-8859-1')
        fy.close()

    def __getitem__(self, index):
        fx=np.array(self.data_x[index])
        label = torch.tensor(self.data_y[index],dtype=torch.long)
        fx_t = torch.from_numpy(fx).unsqueeze(dim=0)
        fx_t = fx_t.type(torch.FloatTensor)
        return fx_t,label
 
    def __len__(self):
        return len(self.data_y)
#根据自己定义的那个MyDataset来创建数据集！注意是数据集！而不是loader迭代器
#train_data=MyDataset('../data/save_data/myset')
#然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
#train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)


#异常数据量 4934
#正常数据量 5290


#训练模块：
#1、需要分测试和训练集 2、在一定epoch后每隔几个就保存一次模型  3、输入数据时进行随机增强
from visdom import Visdom
import torch.optim as optim

class Ecg_trainer:
	def __init__(self,max_epoch=30,save=0,save_epoch=70):
		print('[*]start to init model.')
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		net = Ecg2dnet()
		net=net.to(device)
		save_flag=0
		nn.Sequential(*list(net.children())[:4])
		for m in net.modules():
			if isinstance(m, (nn.Conv2d, nn.Linear)):#Xavier初始化的基本思想是保持输入和输出的方差一致，这样就避免了所有输出值都趋向于0。这是通用的方法，适用于任何激活函数。这里对卷积和全连接层初始化
				nn.init.xavier_uniform_(m.weight)
		print('[+]model init complete,start to init train modules.')
		viz = Visdom()
		viz.line([[0.,0.,0.]], [0], win='train', opts=dict(title='loss&acc', legend=['loss', 'acc','valid']))
		train_data=MyDataset('../data/save_data/myset_train')
		test_data=MyDataset('../data/save_data/myset_test')
		print('[*]data_len:',train_data.__len__)
		train_loader = DataLoader(dataset=train_data, batch_size=24, shuffle=True)
		test_loader = DataLoader(dataset=test_data, batch_size=24, shuffle=True)
		criterion = nn.CrossEntropyLoss()#声明损失(交叉熵)
		optimizer = optim.SGD(net.parameters(),lr=0.0015,momentum=0.8,weight_decay=0.001)#声明优化器，只需要参数，第一个固定为网络的参数集合
		print('[+]train modules init complete,start to train')
		for epoch in range(max_epoch):#30次循环
			print('[*]start epoch:',epoch)
			save_flag=0
			running_corrects = 0.0
			running_loss = 0.0#每次循环开始时初始化总损失为0
			for i,data in enumerate(train_loader,0):#迭代数据加载器
				inputs,labels = data#数据加载器每次返回的是一个批次的数据，这个批次不是由torch.utils.data.Dataset定义而是由Dataloader定义的
				inputs,labels=inputs.to(device),labels.to(device)#放入cuda
				optimizer.zero_grad()#设置当前梯度为0
				outputs=net(inputs)#前向推理
				_,preds = torch.max(outputs,1)
				loss = criterion(outputs,labels)#计算一批数据的交叉熵损失
				loss.backward()#反向推理
				optimizer.step()#优化
				running_corrects += torch.sum(preds == labels.data)
				running_loss += loss.item()#*inputs.size(0)总损失增加,按照一个batch的输入量增加
				epoch_acc = running_corrects.double()
				if i%100 == 99:#1600条数据进行一次显示，注意viz.line特别花时间...
					net.eval()
					test_corrects=0
					for test_i,test_data in enumerate(test_loader,0):
						test_inputs,test_labels=test_data
						test_inputs,test_labels=test_inputs.to(device),test_labels.to(device)
						test_outputs=net(test_inputs)
						_,test_preds = torch.max(test_outputs,1)
						test_corrects += torch.sum(test_preds == test_labels.data)
					if i%200 == 199:
						print('[+]correct_number:',float(running_corrects))#最大应该是epoch*100
						print('[%d,%5d] test_acc: %.3f' %(epoch+1,i+1,float(test_corrects)/1000))
						print('[%d,%5d] acc: %.3f' %(epoch+1,i+1,epoch_acc/(100*24)))#显示的是每个批次的acc与loss
						print('[%d,%5d] loss: %.3f' %(epoch+1,i+1,running_loss/100))#总损失变成平均损失，每2000个序号计算一次
					net.train()
					if((test_corrects/1000)>=0.923):
						save_flag=1
					viz.line([[float(running_loss/100), float(epoch_acc/(100*24)),float(test_corrects/1000)]], [epoch+1], win='train', update='append')
					epoch_acc_out=int(epoch_acc*1000/(100*24))
					running_corrects = 0.0
					running_loss = 0.0
			if((save==1)&(save_flag==1)&(epoch>=save_epoch)):
				torch.save(net,'../data/save_model/net_'+str(int(test_corrects))+'_'+str(epoch_acc_out)+'.pt')


Ecg_trainer(max_epoch=300,save=1,save_epoch=150)
#python -m visdom.server 启动可视化训练服务器
'''
#保存、加载整个模型：
torch.save(model,path)
torch.load(path)#可选参数：map_location={'cuda:1':'cuda:0'}，没有保存时用的设备会用不了时用这个参数
#model.train()和model.eval():
#所有Batch Normalization的训练和测试时的操作不同(对batch和单图片)。dropout在训练中，每个隐层的神经元先乘以概率P，然后再进行激活。在测试中，所有的神经元先进行激活，然后每个隐层神经元的输出乘P。
'''
