# Frangi-filter-based-Hessian
基于Hessian矩阵的Frangi滤波 python版本
包括：Hessian、 eig2image and Frangi

![test image](https://github.com/yimingstyle/Frangi-filter-based-Hessian-/blob/master/Screenshots/test.tif)


Frangi filter:


![test image](https://github.com/yimingstyle/Frangi-filter-based-Hessian-/blob/master/Screenshots/result.png)


可能有的朋友误会了，我贴的两张图片中，滤波后的结果背景处那些密密麻麻的东西，其实并不是毛细血管。
而是图片的背景噪声也被当作血管处理了。
我只是想展示一下frangi对不同管径的血管对滤波效果，所以没有改卷积核的尺度范围。
fringe滤波只是在形态上，对管状结构进行平滑。它没有办法对图像中的内容识别，如果在它的卷积尺度内，它会把一切东西都平滑成管状。
所以，在frangi之前也应该对图像进行一些处理，尽量减少图像中的背景噪声，再选择合适的卷积尺度。
