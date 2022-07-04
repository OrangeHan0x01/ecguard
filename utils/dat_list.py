#用于获取训练集和测试集数据
#原则：1、不需要使用到全部数据   2、导联1（ECG1、I）和导联MLII、II都可以有，但导联1的不应当多于II的。  3、ptb_xl数据集没有ann
#4、ptb-xl的频率是100，所以最好还是降低到100，且ptb-xl每个文件应该最多获取一个2d_array
#
#

import os
from pre_process import sample_100,n_getbeat,datain,nor
import random
import numpy
import pickle

def get_filelist(dataset_name):#输入数据集名称，返回所有该数据集下的文件为[[文件名,文件路径],[]]的格式
	basepath='../data/'
	path=[]
	if dataset_name=='mit-normal':
		basepath='../data/mit-normal'
		path=['']
	elif dataset_name=='mit-arr':
		basepath='../data/mit-arr'
		path=['']
	elif dataset_name=='x-mitdb':
		basepath='../data/mit-arr/x_mitdb'
		path=['']
	#这里枚举dir,找到所有dat文件后去除后缀
	files=[]
	for dirname in path:
		ls=os.listdir(basepath+dirname)
		for file in ls:#枚举每一个文件
			if('.dat' in file):#只对dat后缀数据操作
				file0=file.strip('.dat')
				if(dirname==''):
					filepath=basepath+'/'+file0
				else:
					filepath=basepath+'/'+dirname+'/'+file0
				files.append(filepath)
	return files

#输入模型进行识别
#数据集分别读取（生成新混合数据集）：优先从心律失常数据集中读取数据，输入一部分数据后统计标签分别为1（异常）和0（正常）的数据量。

def data_save(store_path,d2arrays,labels):#存储文件的函数，需要化为字典,暂时不用了
	file_x = open(store_path+'_x.pcdat', 'wb')
	file_y = open(store_path+'_y.pcdat', 'wb')
	pickle.dump(d2arrays, file_x)
	pickle.dump(labels, file_y)
	file_x.close()
	file_y.close()
#读取：pickle.load(file)

def dataset_gen(record_f,label_1,label_0):#生成一个文件的组数据，需要读取ann列表
	record,annotation,ann_pos,siglen,fs=datain(record_f)
	if(len(record)<=2):
		return numpy.array([]),numpy.array([]),label_1,label_0
	R_peaks=[]
	#R_peaks同样可以从算法中获得
	for ann_i in ann_pos:
		ann_i+=random.choice([1,-1,0])#添加随机值，增强泛化能力
		R_peaks.append(ann_i)#R_peaks会在get_beat中减少
	sig,qrslist=sample_100(record,R_peaks,fs)#qrslist其实就是从ann_pos中获得的，并且添加了随机数
	d2arrays,ann_list=n_getbeat(sig,qrslist,annotation)
	if(len(annotation)<=2):#异常annotation,即那些没有数据,默认为全正常，注意
		print('[+]异常annotation:长度过短，请注意是否要去除')
		ann_list=[]
		for i in range(len(d2arrays)):
			ann_list.append(0)
			label_0+=1
	else:
		for i in ann_list:
			if i==1:
				label_1+=1
			else:
				label_0+=1
	print('[+]矩阵长度：',len(d2arrays))
	print('[+]标签长度：',len(ann_list))
	return d2arrays,ann_list,label_1,label_0


def dataset_storage(store_path,flist_arr):#整合label为0和1的数据集，且最终保存生成新数据集文件
    label_1=0
    label_0=0
    d2array_list=[]
    ann_llist=[]
    for f in flist_arr:
        print('[+]正在处理，文件名：'+f)
        d2arrays,labels,label_1,label_0=dataset_gen(f,label_1,label_0)
        try:#先全化为list对象
            d2arrays=d2arrays.tolist()
        except:
            pass
        try:
            labels=labels.tolist()
        except:
            pass
        for sin_dele in d2arrays:
            d2array_list.append(sin_dele)
        for sin_lele in labels:
            ann_llist.append(sin_lele)
    data_save(store_path,d2array_list,ann_llist)
    print('datas_store_shape:',numpy.array(d2array_list).shape)
    print('labels_store_shape:',numpy.array(ann_llist).shape)
    print('异常数据量',label_1)
    print('正常数据量',label_0)
    return d2array_list,ann_llist#形状分别为3d和1d