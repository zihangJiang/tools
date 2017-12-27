import sys
import os
#to rename all the file in the parent_dir
path = sys.path[0]
parent_dir='/parent'
#choose a name for every image
child_name='child'
list=os.listdir(path+parent_dir)
num=0
for file in list:
	num=num+1
	newname=child_name+str(num)+'.jpg'
	os.rename(path+parent_dir+'/'+file,path+parent_dir+'/'+newname)
	
print('finished')