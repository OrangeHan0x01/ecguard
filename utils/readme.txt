文件说明：
dataset_create.py:（需要执行）从mit-bih数据集进行预处理并提取所需数据以pickle格式保存
train_test.py:（需要执行）在创建数据文件后分割训练测试集
nn_utils.py:训练保存模型
data_view.py:（测试用）绘制dataset_create.py创建的数据
post_simu.py:模拟上传数据
dat_list.py:用于封装获取、保存数据的函数
Data-analysis.py:（测试用），用于分析结果
model.py：模型结构文件
postdata_get.py:（测试用）将一些mit-bih中的数据写到一个txt文件中，用于上传和绘图
pre_process.py:数据预处理相关函数
sqlmethods.py:sql、base64相关函数，这个文件和服务器目录中的相同




mit-bih数据集的路径：将官网文件解压到../data/mit-arr/下