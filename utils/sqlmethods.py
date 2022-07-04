#sql相关函数
#
import struct
import base64

def s2l(s):#2d数组转化
	lm=s.split('], [')#形如[[1,2,3], [4,5,6]]
	lm0=[i.strip('[').strip(']') for i in lm]#转化为1d,其中每个元素为一个list--已经去掉了边界
	lr=[]
	for elem in lm0:
		ele_m=elem.split(', ')
		ele_m1=[float(i) for i in ele_m]
		lr.append(ele_m1)
	return lr

def sspl(s):#3d转2个2d,形如[[[1,2,3], [4,5,6]], [[3,2,1], [6,5,4]]]
	ll=s.split(']], [[')#[[[1,2,3], [4,5,6        3,2,1], [6,5,4]]]
	l0='[['+ll[0].strip('[[[')+']]'#1,2,3], [4,5,6
	l1='[['+ll[1].strip(']]]')+']]'
	return l0,l1#l1是fft

def d_transform(array):#数据转字节码再转字符串，base64,array为一个3d矩阵
#每次取出一行元素，共计8*2=16行
	strb64=''
	struct_id=''
	for i in range(90):
			struct_id+='f'
	for i in array:
		for j in i:
			bt=struct.pack(struct_id,* j)#直接每个字符转一次然后拼接吧
			strb64=strb64+','+base64.b64encode(bt).decode('gbk')#b64编码方便传输，拆数组也方便，因为都是用字母和==存储的，使用gbk编码
	strb64=strb64[1:]#去除首个字符‘,’
	return strb64#长度：7695，最小长度为4*8*90*2=5760字节，这里冗余不大

def d_decode(strb64):#字符串转3d数据
	result=[]
	data_array=[]
	strarray=strb64.split(',')#应该有16条
	struct_id=''
	for i in range(90):
			struct_id+='f'
	if(len(strarray)!=16):
		return 'error'
	for strd in strarray:
		bt=strd.encode('gbk')
		bt=base64.b64decode(bt)
		array=list(struct.unpack(struct_id,bt))
		data_array.append(array)#验证为16*90
	d_t=data_array[0:8]
	d_f=data_array[8:16]
	result.append(d_t)
	result.append(d_f)
	return result
