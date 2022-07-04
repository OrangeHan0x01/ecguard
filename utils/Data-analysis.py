import pickle
import numpy as np
import torch
import torch.nn as nn
from model import Ecg2dnet
from torch.utils.data import DataLoader
#方法1：统计原数据、处理过数据的正常异常标签总和
def stat_label():
#1、读取原数据，正常预处理到删除保留数据前一步
#2、读取标签列表，获取：总长度、正常数据量、异常数据量
#3、从测试集、训练集读取总长度、正常数据量、异常数据量
	#原数据：共15542条，异常数据量 4934，正常数据量 10608
	#train&test_read：
	fy = open('../data/save_data/myset_'+'y.pcdat', 'rb')
	data_y = pickle.load(fy,encoding='iso-8859-1')
	y_1=0
	y_0=0
	for i in data_y:
		if(i==0):
			y_0+=1
		elif(i==1):
			y_1+=1
	print('标签为0：',y_0)
	print('标签为1：',y_1)
	print('总共：',y_0+y_1)




#方法2：4种判断类别分析（随意，可以分测试集训练集）
def model_test():
	datapath='../data/save_data/myset_'
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	cnet2c=torch.load('../data/save_model/net2_922_981.pt').to(device)
	#前面两个分别代表tp和fn
	conf2c_tt=0
	conf2c_tf=0
	conf2c_ft=0
	conf2c_ff=0
	fx = open(datapath+'x.pcdat', 'rb')
	data_x = pickle.load(fx,encoding='iso-8859-1')
	fx.close()
	fy = open(datapath+'y.pcdat', 'rb')
	data_y = pickle.load(fy,encoding='iso-8859-1')
	fy.close()
	print('[*]init over.')
	print('[*]data_len:',len(data_x))
	print('[*]label_len:',len(data_y))
	for i in range(len(data_x)):
		#预处理
		fx=np.array(data_x[i])
		label=data_y[i]
		fftx=np.array(np.abs(np.fft.fft(data_x[i])))[np.newaxis,:]
		fx1=np.array(data_x[i])[np.newaxis,:]
		f2c_t=np.concatenate((fftx, fx1), axis=0)
		f2c = torch.from_numpy(f2c_t).type(torch.FloatTensor).unsqueeze(dim=0).to(device)
		#识别
		output2c=cnet2c(f2c)
		_,pred2c = torch.max(output2c,1)
		#统计
		if(label==1):#实际为1，即异常,t
			if(pred2c==1):
				conf2c_tt+=1
			elif(pred2c==0):
				conf2c_tf+=1
		elif(label==0):
			if(pred2c==1):
				conf2c_ft+=1
			elif(pred2c==0):
				conf2c_ff+=1
	print('[*]run over.')
	print('[+]conf2c_tt:',conf2c_tt)
	print('[+]conf2c_tf:',conf2c_tf)
	print('[+]conf2c_ft:',conf2c_ft)
	print('[+]conf2c_ff:',conf2c_ff)

model_test()