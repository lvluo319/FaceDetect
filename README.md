# FaceDetect

This project is used to detect faces from a group photo, where some people in the photo have been registed and others not.

To run this project, you need to make sure that you have Caffe, Dlib, Opencv successfully installed. By the way, I use Pyhon3.5 in this project.

References for implementation:

[1] To install Caffe: https://github.com/BVLC/caffe

[2] Models use the "Axpy" layer which is a combination of two consecutive operations channel-wise scale and element-wise summation, you have to add this layer in Caffe and build. Reference: https://blog.csdn.net/qq_38451119/article/details/82256095

[3] Models is too large to commit, you can download here: https://pan.baidu.com/s/1chEyDifZt0PtdnwHI3HiMQ and the code is 9s69, and set up a new file folder named "models" and put them in it.
