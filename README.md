# cnn-tensorflow
卷积神经网络示例
cnn-split-2.py:20%的数据作为测试集。80%作为训练集。
NC_3.csv
tumour_3.csv
经过matlab整理之后的脑影像数据。文件中的每一行代表一张灰度图片的所有像素点。


卷积神经网络的结构
输入数据为61*73的二维矩阵
层数  名称     尺寸      卷积核大小
0	    输入层	 61*73*1	
1	   卷积层	 57*69*16	   5*5
2	   最大化池	 28*34*16	  2*2
3	   卷积层	  24*30*32	 5*5
4	  最大化池	12*15*32	  2*2
5	  卷积层    10*13*16	   3*3
6	  最大化池	5*6*16	    2*2
7	  softmax层	2（类别概率输出）	
