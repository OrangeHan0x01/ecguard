#除sql以外的功能函数,以及测试输入数据等
import pickle
from flask_sqlalchemy import SQLAlchemy
from sqlmethods import d_transform,d_decode
import random
from flask import Flask
from datetime import datetime
import numpy as np
import torch
import time
app = Flask(__name__)
db=SQLAlchemy(app)

app.config['SQLALCHEMY_DATABASE_URI']='postgresql://school:school@localhost:5432/schoolwork'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

class Ecg_data(db.Model):#ecg sql表
	__tablename__='ecg'
	id=db.Column(db.Integer,primary_key=True,index=True)
	uid=db.Column(db.Text)
	data=db.Column(db.Text)#确认一下，这里长度应该要很长
	result=db.Column(db.Text,default='None')#识别结果是“异常”与“正常”，但是可以手动修改
	timestamp=db.Column(db.DateTime,default=datetime.utcnow)

def input_data(datapath):#向数据库输入示例数据！
	fx = open(datapath+'x.pcdat', 'rb')
	data_x = pickle.load(fx,encoding='iso-8859-1')
	fx.close()
	fy = open(datapath+'y.pcdat', 'rb')
	data_y = pickle.load(fy,encoding='iso-8859-1')
	fy.close()
	for index in range(80):#插入80条数据
		fftx=np.array(np.abs(np.fft.fft(data_x[index])))[np.newaxis,:]
		fx1=np.array(data_x[index])[np.newaxis,:]
		dx=np.concatenate((fx1,fftx), axis=0).tolist()
		dy=''
		if (data_y[index]==0):
			dy='正常'
		else:
			dy='异常'
		dx=d_transform(dx)
		userid=random.choice([1,2,3,4,5])
		n_record=Ecg_data(data=dx,uid=userid,result=dy)
		db.session.add(n_record)
	db.session.commit()
	print('success')

#input_data('../data/save_data/myset_train')

#2、调用模型进行识别
def rec_test():
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	cnet=torch.load('../data/save_model/net2_922_981.pt')
	cnet=cnet.to(device)
	dst_object=Ecg_data.query.filter_by(id=81).first()
	
	start_time=time.time()
	dst_data=d_decode(dst_object.data)
	npdata=np.array(dst_data)
	fx_t = torch.from_numpy(npdata)
	fx_t = fx_t.type(torch.FloatTensor).unsqueeze(dim=0).to(device)
	outputs=cnet(fx_t)
	_,preds = torch.max(outputs,1)
	if preds==1:
		n_result='正常'
	else:
		n_result='异常'
	cost=time.time()-start_time
	print(cost)
	print(n_result)

rec_test()

