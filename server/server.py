#flask+bootstrap服务器

from flask import Flask,request
from flask import render_template,flash,redirect,url_for
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm,CSRFProtect
from wtforms import Form
from wtforms import StringField,SubmitField
from wtforms.validators import DataRequired
from sqlmethods import s2l,sspl,d_decode
from datetime import datetime
from flask_moment import Moment
import matplotlib.pyplot as plt
import io
import base64
import torch
import numpy as np
#新建表：ecg，包括ecg数据和用户id、识别结果
app = Flask(__name__)
#csrf = CSRFProtect(app)
bootstrap = Bootstrap(app)
app.config['SQLALCHEMY_DATABASE_URI']='postgresql://用户名:密码@localhost:端口/数据库名'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
#app.config.setdefault('BOOTSTRAP_SERVE_LOCAL',True)
app.config["SECRET_KEY"] = 'xHmi9uLWrbyVr5au2v92DXZw1'
#app.config['WTF_CSRF_CHECK_DEFAULT'] = False
#app.config.from_object(__name__)
moment = Moment(app)
db=SQLAlchemy(app)
#加载pytorch模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cnet=torch.load('../data/save_model/net2_922_981.pt').to(device)


class Ecg_data(db.Model):#ecg sql表
	__tablename__='ecg'
	id=db.Column(db.Integer,primary_key=True,index=True)
	uid=db.Column(db.Text)
	data=db.Column(db.Text)#确认一下，这里长度应该要很长
	result=db.Column(db.Text,default='None')#识别结果是“异常”与“正常”，但是可以手动修改
	timestamp=db.Column(db.DateTime,default=datetime.utcnow)


class ResultForm(FlaskForm):
	n_result = StringField('结果',validators=[DataRequired()])#怎么才能在这里有默认值为所读取的原结果？
	submit = SubmitField('提交修改')

@app.route('/index',methods=['GET', 'POST'])
def index():
	#需要：列表、按钮（识别、详情（详情中可以修改结果）），识别按钮是跳转返回(或者异步)然后给一个flash
	page=request.args.get('page',1,type=int)
	pagination=Ecg_data.query.order_by(Ecg_data.id.desc()).paginate(page,per_page=20,error_out=False)
	posts=pagination.items
	#剩余：按钮
	return render_template('list.html',pagination=pagination,posts=posts)


@app.route('/data',methods=['GET', 'POST'])
def data_page():
	#详情页面,左侧是绘制图像，右侧包括：1、修改结果的文本框和对应按钮。2、下载数据按钮（默认view）（下载按钮只需要一个路径）
	did=request.args.get("id")#图像x,y反了
	dst_object=Ecg_data.query.filter_by(id=did).first()
	#form = ResultForm(dst_object.result,csrf_enabled=False)
	form = ResultForm()
	#form.csrf_token=form.csrf_token
	#form.n_result.default = dst_object.result
	#form.process()
	d3=d_decode(dst_object.data)
	#绘制图像部分
	img = io.BytesIO()
	plt.rcParams['figure.figsize'] = (8.0, 8.0)#横，纵
	plt.subplots_adjust(wspace =0, hspace =0)
	#只绘制d3[0]
	for i in range(8):
		ax=plt.subplot(421+i)
		plt.plot(range(90),d3[0][i])
		#ax.axes.xaxis.set_ticklabels()
	plt.savefig(img, format='png',bbox_inches='tight',pad_inches = 0)
	img.seek(0)
	plot_url = base64.b64encode(img.getvalue()).decode()
	#图像绘制结束，目前还需要一个表单、2个按钮
	if form.validate_on_submit():
		try:
			n_result=form.n_result.data
			Ecg_data.query.filter_by(id=did).update({'result': n_result})
			db.session.commit()
			flash('诊断结果已更新')
		except:
			flash('诊断结果更新失败')
	return render_template('data.html', plot_url=plot_url,dst_object=dst_object,form=form)



@app.route('/data_post',methods=['POST'])
def data_post():#post数据api
	try:
		ecg_data=request.form['ed']
		userid=request.form['uid']
		n_record=Ecg_data(data=ecg_data,uid=userid)
		db.session.add(n_record)
		db.session.commit()
		return 'success'
	except:
		return 'error'

@app.route('/download',methods=['GET', 'POST'])
def download():
	did=request.args.get("id")
	dst_object=Ecg_data.query.filter_by(id=did).first()
	dst_data=str(d_decode(dst_object.data))
	dst_result=dst_object.result
#根据参数决定原样输出还是通道分开成数组模式输出
	if(request.args.get("view")):#输出分行、分通道，且最后输出结果
		mid0,mid1=sspl(dst_data)
		mid2,mid3=s2l(mid0),s2l(mid1)
		out_str=''
		for i in mid2:
			out_str=out_str+str(i)+'\n'
		out_str+='\n\n'
		for i in mid3:
			out_str=out_str+str(i)+'\n'
		out_str+='\n\n'
		out_str+=dst_result
		dst_data=out_str
	return dst_data

@app.route('/rec',methods=['GET'])
def rec():
	global cnet
	did=request.args.get("id")
	dst_object=Ecg_data.query.filter_by(id=did).first()
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
	Ecg_data.query.filter_by(id=did).update({'result': n_result})
	db.session.commit()
	flash('诊断结果已更新')
	return redirect(url_for('index'))
