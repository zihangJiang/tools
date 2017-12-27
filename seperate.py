import os
import shutil
dir="data/test1"
names = ["beijing","tianjin","shanghai","chongqing","hebei","shanxi","liaoning","jilin","heilongjiang"\
    ,"jiangsu","zhejiang","anhui","fujian","jiangxi","shandong","henan","hunan","hubei","guangdong",\
    "guangxi","hainan","sichuan","guizhou","yunnan","shaanxi","gansu","qinghai","taiwan","neimenggu",\
    "xizang","ningxia","xinjiang","xianggang","aomen","hefei","hangzhou","taiyuan","shijiazhuang","wuhan"\
    ,"nanjing"]
os.mkdir('data_set')
for c in range(40):
    count=c
    province=names[count]
    print('making '+province)
    save_dir='data_set/'+province
    le=len(province)
    os.mkdir(save_dir)
    i=0
    for root,dirs,filename in os.walk(dir):
        for fi in filename:
            if str(fi[0:le])==province:
                i=i+1
                shutil.copy(os.path.join(root,fi),save_dir)
                os.rename(save_dir+'/'+province+'.jpg',save_dir+'/'+province+str(i)+'.jpg')
