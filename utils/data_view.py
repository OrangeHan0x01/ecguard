#测试
import pickle
import matplotlib.pyplot as plt
import numpy as np


def showdata(datapath):
	fx=open(datapath+'x.pcdat', 'rb')
	fy=open(datapath+'y.pcdat', 'rb')
	data_x = pickle.load(fx,encoding='iso-8859-1')
	data_y = pickle.load(fy,encoding='iso-8859-1')
	fx.close()
	fy.close()
	print('数据量',len(data_y))
	y_1=0
	y_0=0
	for di in data_y:
		if(di==1):
			y_1+=1
		elif(di==0):
			y_0+=1
		else:
			print(di)
	print('正常数据量：',y_0)
	print('异常数据量：',y_1)
	index=0
	fftx=np.array(np.abs(np.fft.fft(data_x[index])))[np.newaxis,:]
	fx1=np.array(data_x[index])[np.newaxis,:]
	dx=np.concatenate((fx1,fftx), axis=0).tolist()
	plt.rcParams['figure.figsize'] = (8.0, 8.0)
	plt.subplots_adjust(wspace =0, hspace =0)
	q=0
	for i in range(8):
			ax=plt.subplot(421+i)
			plt.plot(range(90),dx[0][i])
	plt.show()

showdata('../data/save_data/myset_train')
