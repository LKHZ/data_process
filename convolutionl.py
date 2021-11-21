import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img = Image.open('图片1.png')
grey_img = np.array(img.convert('L'),dtype=np.float32)

plt.figure(figsize=(6,6))
plt.imshow(grey_img,cmap=plt.cm.gray)
plt.axis('on') #开启坐标轴
plt.show()

#卷积前的操作,将图片变为四维数组
imgh,imgw = grey_img.shape
new_grayimg = torch.from_numpy(grey_img.reshape((1,1,imgh,imgw)))
print('卷积前的尺寸:',new_grayimg.shape)

#设置卷积核（图像轮廓提取卷积核）
kersize = 5
ker = torch.ones(kersize,kersize,dtype=torch.float32)*-1
ker[2,2] = 24
ker = ker.reshape((1,1,kersize,kersize))        #维度1*1*5*5
print('图像轮廓提取卷积核:',ker)


#卷积操作
conv2d = nn.Conv2d(1,2,(kersize,kersize),bias = False)
print('卷积参数设置:',conv2d)
conv2d.weight.data[0] = ker                     #权重0是轮廓卷积核，卷重1是随机卷积核
imconv2dout = conv2d(new_grayimg)               #用两个卷积核对图片进行卷积
print('卷积后的尺寸:',imconv2dout.shape)
imconv2dout_im = imconv2dout.data.squeeze()     #将第一个batch_size维度去除
print('卷积后的尺寸(去除batch_size):',imconv2dout_im.shape)

#画图
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(imconv2dout_im[0],cmap=plt.cm.gray)
plt.axis('on')

plt.subplot(1,2,2)
plt.imshow(imconv2dout_im[1],cmap=plt.cm.gray)
plt.axis('on')
plt.show()




# plt.figure(figsize=(6,6))
# plt.imshow(grey_img,cmap=plt.cm.gray)
# plt.axis('on') #关闭坐标轴
# plt.show()


"""
输出结果
卷积前的尺寸: torch.Size([1, 1, 512, 511])
图像轮廓提取卷积核: tensor([[[[-1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1.],
          [-1., -1., 24., -1., -1.],
          [-1., -1., -1., -1., -1.],
          [-1., -1., -1., -1., -1.]]]])
卷积参数设置: Conv2d(1, 2, kernel_size=(5, 5), stride=(1, 1), bias=False)
卷积后的尺寸: torch.Size([1, 2, 508, 507])
卷积后的尺寸(去除batch_size): torch.Size([2, 508, 507])
"""
