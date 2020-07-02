本存储库构建于论文《Woodblock-printing Mongolian Words Recognition by Bi-LSTM with Attention Mechanism》 论文地址：https://ieeexplore.ieee.org/document/8978006

##运行环境：
###1.推荐在gpu环境下运行
需安装cuda，cudnn，cudatoolkit
###2.具体环境在conda.txt中列出
其中必要包为theano==0.9.0, numpy==1.13.3, six==1.11.0, tqdm==4.15.0, h5py==2.7.0, pygpu==0.6.9，推荐使用pip安装指定版本


##训练模型的具体步骤：
###1.预处理图像
pool_x.py利用图像目录生成模型所需数据，fl*为半帧像素高（如：fl2为半帧为2像素，fl3为半帧为3像素），*.lst为图像路径的列表文件
运行样例（按相邻帧重叠、半帧高8像素的方式对列表文件中的图像进行预处理）：
python pool_x_overlap.py fl8 testing_set_data.lst 

运行样例（按相邻帧不重叠、半帧高8像素的方式对列表文件中的图像进行预处理）：

python pool_x_nonoverlap fl8 testing_set_data.lst 

###2. 预处理图像标签
pool_y.py利用图像标签生成模型所需数据，.lst文件中为图像列表文件中各个图像的标签

运行样例：

python pool_y.py testing_set_label.lst

###3.生成词典，用于后处理
build_tree.py利用图像所有标签的列表生成词典，用于test的后处理

运行样例（第一个参数为输入的标签列表文件，第二个参数为输出的词典文件）：

python build_tree.py label.lst tree

###4.训练模型
使用train.py训练模型，输入输出须在文件内指定

运行样例（fl8指半帧高度为8，200指训练200轮）：

THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32 python train.py fl8 200 

###5.测试模型
使用test.py测试模型，输入输出须在文件内指定

运行样例（fl2指半帧高度为2，100指测试经过100轮训练的模型）：

THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32 python test.py fl2 100