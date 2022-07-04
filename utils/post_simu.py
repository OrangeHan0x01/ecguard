import time
from QRS_util import EKG_QRS_detect
from pre_process import n_getbeat
import numpy as np
from sqlmethods import d_transform
import requests



def data_process(data):
	qrs_list, _, _=EKG_QRS_detect(np.array(data), 100, False, False)
	a2d_elem,_=n_getbeat(data,qrs_list.tolist(),range(len(qrs_list)))
	print(len(a2d_elem[0]))
	a2d_elem=a2d_elem[0]
	#n_getbeat的第三个参数是标签列表，处理数据集用的，采集时不需要故不使用
	fft_elem=np.abs(np.fft.fft(a2d_elem))
	n_data=[a2d_elem,fft_elem]
	return d_transform(n_data)


def data_post(n_data,uid,url):#n_data是已经base64编码的数据
	data = {'ed': n_data,'uid': uid}
	requests.post(url, data=data)
	return

def post_test():
	with open('post_sample.txt','r') as f:
		data=[float(i) for i in f.readlines()]
	data_p=data_process(data)
	data_post(data_p,'7','http://127.0.0.1:5000/data_post')
	print('Data posted, try looking for your data in the database.')

post_test()