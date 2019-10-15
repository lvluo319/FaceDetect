# FaceDetect

This project is used to detect faces from a group photo, where some people in the photo have been registed and others not.

To run this project, you need to make sure that you have Caffe, Dlib, Opencv successfully installed. By the way, I use Pyhon3.5 in this project.

References for implementation:

[1] To install Caffe: https://github.com/BVLC/caffe

[2] You can use Pip to install Dlib and Opencv.

[3] The face detection pretrained model I choose is the SE-ResNet-50 model, which is a pretrained model trained on VGGFace2, here is the reference: https://github.com/ox-vgg/vgg_face2. This model use the "Axpy" layer which is a combination of two consecutive operations channel-wise scale and element-wise summation, you have to add this layer in Caffe and rebuild it. If you dont't know how to add new layer, here is the reference: https://blog.csdn.net/qq_38451119/article/details/82256095

[4] Models is too large to commit, you can download here: https://pan.baidu.com/s/1chEyDifZt0PtdnwHI3HiMQ and the code is 9s69, and set up a new file folder named "models" and put them in it.

[5] You can just run the project in a default mode without any parameters. You can also use your own photos to run the project.
