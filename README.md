���洢�⹹�������ġ�Woodblock-printing Mongolian Words Recognition by Bi-LSTM with Attention Mechanism�� ���ĵ�ַ��https://ieeexplore.ieee.org/document/8978006
\<br>
#���л�����\<br>
1.�Ƽ���gpu���������У��谲װcuda��cudnn��cudatoolkit\<br>
2.���廷����conda.txt���г������б�Ҫ��Ϊtheano==0.9.0, numpy==1.13.3, six==1.11.0, tqdm==4.15.0, h5py==2.7.0, pygpu==0.6.9���Ƽ�ʹ��pip��װָ���汾\<br>
\<br>
#ѵ��ģ�͵ľ��岽�裺\<br>
##1.Ԥ����ͼ��\<br>
pool_x.py����ͼ��Ŀ¼����ģ���������ݣ�fl*Ϊ��֡���ظߣ��磺fl2Ϊ��֡Ϊ2���أ�fl3Ϊ��֡Ϊ3���أ���*.lstΪͼ��·�����б��ļ�\<br>
����������������֡�ص�����֡��8���صķ�ʽ���б��ļ��е�ͼ�����Ԥ������\<br>
python pool_x_overlap.py fl8 testing_set_data.lst \<br>
����������������֡���ص�����֡��8���صķ�ʽ���б��ļ��е�ͼ�����Ԥ������\<br>
python pool_x_nonoverlap fl8 testing_set_data.lst \<br>
\<br>
##2. Ԥ����ͼ���ǩ\<br>
pool_y.py����ͼ���ǩ����ģ���������ݣ�.lst�ļ���Ϊͼ���б��ļ��и���ͼ��ı�ǩ\<br>
����������\<br>
python pool_y.py testing_set_label.lst\<br>
\<br>
##3.���ɴʵ䣬���ں���\<br>
build_tree.py����ͼ�����б�ǩ���б����ɴʵ䣬����test�ĺ���\<br>
������������һ������Ϊ����ı�ǩ�б��ļ����ڶ�������Ϊ����Ĵʵ��ļ�����\<br>
python build_tree.py label.lst tree\<br>
\<br>
##4.ѵ��ģ��\<br>
����������fl8ָ��֡�߶�Ϊ8��200ָѵ��200�֣���\<br>
THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32 python train.py fl8 200 \<br>
\<br>
##5.����ģ��\<br>
����������fl2ָ��֡�߶�Ϊ2��100ָ���Ծ���100��ѵ����ģ�ͣ���\<br>
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32 python test.py fl2 100 \<br>