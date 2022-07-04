from dat_list import *
from pre_process import *




flist_arr=get_filelist('mit-arr')
flist_arr+=get_filelist('x-mitdb')
flist_bih=get_filelist('mit-normal')
xs,ys=dataset_storage('../data/save_data/origset',flist_arr)#x,y就是可以直接输入训练的数据！for x in xs和y in ys:x是一个(8,90)的2维数组，y是0或1
