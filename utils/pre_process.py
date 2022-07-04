import numpy as np
from scipy import signal
import wfdb
import random
#注：QRS_util可能需要numpy格式数据

#def2:输入整个数据和周期定位点列表，修改采样率为100（一般都是超出的，输入参数删值即可）
def sample_100(sig,qrs_list,fs):#每fs个数据中要删除fs-100个数据,其实可以先清数据，然后qrs用自带算法
	if fs <= 100:
		print('采样率过低，无法再删减')
		return sig
	new_sig=[]
	new_qrs=[]
	id_list=[0]
	#print('new_len:',100*len(sig)/fs)#新数据长度
	#print('rt:',100/fs)#新数据占原数据大小比例
	dt=1/(100/fs)#计算间隔
	#print('dt',dt)
	max_id=0
	while(round(max_id+dt)<=len(sig)-1):
		if(round(max_id)!=round(max_id+dt)):
			max_id+=dt
			id_list.append(round(max_id))
		else:
			max_id+=dt
	#print('length of id_list:',len(id_list))
	for id in id_list:#序号与id_list对比，一样的写入新数组，注意qrs也要相应削减！
	#qrs削减：-round(100*qrs/fs)
		new_sig.append(sig[id])
	for qrs_d in qrs_list:
		qrs_d=round(100*qrs_d/fs)
		new_qrs.append(qrs_d)
	return new_sig,new_qrs

def sample_200(sig,qrs_list,fs):#每fs个数据中要删除fs-200个数据,其实可以先清数据，然后qrs用自带算法
	if fs <= 200:
		print('采样率过低，无法再删减')
		return sig
	new_sig=[]
	new_qrs=[]
	id_list=[0]

	dt=1/(200/fs)#计算间隔

	max_id=0
	while(round(max_id+dt)<=len(sig)-1):
		if(round(max_id)!=round(max_id+dt)):
			max_id+=dt
			id_list.append(round(max_id))
		else:
			max_id+=dt
	#print('length of id_list:',len(id_list))
	for id in id_list:#序号与id_list对比，一样的写入新数组，注意qrs也要相应削减！
	#qrs削减：-round(100*qrs/fs)
		new_sig.append(sig[id])
	for qrs_d in qrs_list:
		qrs_d=round(200*qrs_d/fs)
		new_qrs.append(qrs_d)
	return new_sig,new_qrs

def upsample_200(sig,qrs_list,fs):#每fs个数据中要增加200-fs个数据,其实可以先清数据，然后qrs用自带算法
	if fs <= 200:
		print('采样率过低，无法再删减')
		return sig
	new_sig=[]
	new_qrs=[]
	id_list=[0]
	#print('new_len:',100*len(sig)/fs)#新数据长度
	#print('rt:',100/fs)#新数据占原数据大小比例
	dt=1/(200/fs)#计算间隔
	#print('dt',dt)
	max_id=0
	while(round(max_id+dt)<=len(sig)-1):
		if(round(max_id)!=round(max_id+dt)):
			max_id+=dt
			id_list.append(round(max_id))
		else:
			max_id+=dt
	#print('length of id_list:',len(id_list))
	for id in id_list:#序号与id_list对比，一样的写入新数组，注意qrs也要相应削减！
	#qrs削减：-round(100*qrs/fs)
		new_sig.append(sig[id])
	for qrs_d in qrs_list:
		qrs_d=round(200*qrs_d/fs)
		new_qrs.append(qrs_d)
	return new_sig,new_qrs

def n_getbeat(sig,qrs_list,o_annlist):#整合流程,减少错误和不对齐
	sig_list=[]
	class_VEBaF=['A','a','J','S','V','E','F']
	ann_list=[]
	del_list=[]
	for point in qrs_list:
		if point<=41:
			del_list.append(point)
		elif point>=(len(sig)-50):
			del_list.append(point)
	for point in del_list:
		qrs_list.remove(point)
	print('[+]get_beat_siglen:',len(sig))

	for i in range(len(qrs_list)):
		sig_list.append(sig[int(qrs_list[i]-40):int(qrs_list[i]+50)])
		if(o_annlist[i] in class_VEBaF):
			ann_list.append(1)
		else:
			ann_list.append(0)

	print('[+]annlist_len:',len(ann_list))
	print('[+]get_beat_outlen:',len(sig_list))
	ann_labels=[]
	a2d_array=[]
	array_len=int(len(sig_list)/8)
	for i in range(array_len):#0
		a2d_elem=[]
		j_sum=0
		for j in range(8):#0-7
			a2d_elem.append(sig_list[i*8+j])
			j_sum+=ann_list[i*8+j]

		if(j_sum>=1):#这是为了数据集作处理的部分，可以改掉
			ann_labels.append(1)
			a2d_array.append(a2d_elem)
		else:
			if(random.choice([1,-1,-2,0])>=0):
				ann_labels.append(0)#0是正常数据！
				a2d_array.append(a2d_elem)
	return a2d_array,ann_labels


def datain(record_addr):
	head = wfdb.rdheader(record_addr)
	try:
		annotation=wfdb.rdann(record_addr,'atr').symbol#symbol即标签
		ann_pos=wfdb.rdann(record_addr,'atr').sample#即标签对应的位置
	except:
		annotation=[]
	siglen=head.__dict__['sig_len']
	fs=head.__dict__['fs']
	if 'MLII' in head.__dict__['sig_name']:
		dst_channel=head.__dict__['sig_name'].index('MLII')####normal的只有ECG1!
	elif 'II' in head.__dict__['sig_name']:
		dst_channel=head.__dict__['sig_name'].index('II')
	else:
		return [],[],[],0,0
	record = wfdb.rdrecord(record_addr,channels=[dst_channel]).p_signal[:, 0]
	return record,annotation,ann_pos,siglen,fs#返回数据：数据集，标签，长度，频率



#def9:滤波器，0.5~100Hz，用于模拟ad8232的情况，还要滤除50Hz工频干扰（或者干脆0.5~40Hz带通滤波）,感觉要提高到200采样进行滤波再改回来（采样定理，2倍于最高。。。）
def bandstop(data,fs=200):#fs应该为200，滤除48-52Hz频率成分。360Hz
	nfs = fs/2
	low = 48/nfs
	high = 52/nfs
	b, a = signal.butter(2, [low,high], 'bandstop')
	filtedData = signal.lfilter(b, a, data)
	return filtedData
#用之前检验一下数据的频率成分用fft!（已经检验）


def nor(data):#数组归一化，输入可以是list或者np.array，这个函数可以在数据输入时使用，避免训练数据与正式处理数据时的不同
	_range = np.max(data) - np.min(data)
	return (data - np.min(data)) / _range



#其它：ad8232用的应该是I导联
#maixpy需要tuple(h,w,c)格式输入，tuple(list)即可转换
#qrs算法需要numpy格式，可以尝试：#github上的算法
#使用2号导联（因为数据集如此）mlii或者ii



