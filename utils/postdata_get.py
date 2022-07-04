import numpy as np
from scipy import signal
import wfdb
import random
from pre_process import sample_100
import matplotlib.pyplot as plt
#将数据写至文本文件、绘制

datafrom='../data/mit-arr/100'
head = wfdb.rdheader(datafrom)
fs=head.__dict__['fs']
record = wfdb.rdrecord(datafrom,channels=[head.__dict__['sig_name'].index('MLII')]).p_signal[:, 0][0:5000]

nrecord,_=sample_100(record,[],fs)

'''
x=range(len(nrecord))
print('len:',len(nrecord))
plt.plot(x,nrecord)
plt.show()
'''



with open('post_sample.txt','w') as f:
	for data in nrecord:
		f.write(str(data))
		f.write('\n')


