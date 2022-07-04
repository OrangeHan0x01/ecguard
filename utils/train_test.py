#用于从数据分割训练测试集

import pickle

fx = open('../data/save_data/myset_x.pcdat', 'rb')
data_x = pickle.load(fx,encoding='iso-8859-1')#3d,要取出一个成2d
fx.close()
fy = open('../data/save_data/myset_y.pcdat', 'rb')
data_y = pickle.load(fy,encoding='iso-8859-1')
fy.close()

fx_train=data_x[:-1000]#1到倒数1000条，共14500+条
fx_test=data_x[-1000:]#1000条
fy_train=data_y[:-1000]#1到倒数1000条，共14500+条
fy_test=data_y[-1000:]#1000条
file_xtrain = open('../data/save_data/myset_trainx.pcdat', 'wb')
file_ytrain = open('../data/save_data/myset_trainy.pcdat', 'wb')
file_xtest = open('../data/save_data/myset_testx.pcdat', 'wb')
file_ytest = open('../data/save_data/myset_testy.pcdat', 'wb')
print('start to dump')
pickle.dump(fx_train, file_xtrain)
pickle.dump(fy_train, file_ytrain)
print('train_set dump over')
pickle.dump(fx_test, file_xtest)
pickle.dump(fy_test, file_ytest)
file_xtrain.close()
file_ytrain.close()
file_xtest.close()
file_ytest.close()


#len为15000+，这里抽取1000条数据作测试集！
