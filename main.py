import os
import cv2
import dlib
import caffe
import argparse
import numpy as np
import sklearn.metrics.pairwise as pw


def parse_args():
    parser = argparse.ArgumentParser(description='Face Detection.')
    parser.add_argument('--SoloPicDir', type=str, default='./solopic')
    parser.add_argument('--GroupPicDir', type=str, default='./grouppic')
    parser.add_argument('--ResultPicDir', type=str, default='./resultpic')
    args = parser.parse_args()
    return args

thershold = 0.5
imgsize = 224
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
net = caffe.Classifier('./models/senet50_256.prototxt', './models/senet50_256.caffemodel')

def regist_face(root_dir):
    fts = np.zeros((1, 256))
    labels = []
    for file in os.listdir(root_dir):
        img = cv2.imdecode(np.fromfile(root_dir + '/' + file, dtype=np.uint8), -1)
        dets = detector(img, 1)
        num_faces = len(dets)
        if num_faces == 0:
            continue
        faces = dlib.full_object_detections()
        for det in dets:
            faces.append(sp(img, det))
        images = dlib.get_face_chips(img, faces, size=224)
        labelName = file.split('.')[0]
        image = images[0]
        X = np.empty((1, 3, imgsize, imgsize))
        # BGR
        averageImg = [131.0912, 103.8827, 91.4953]
        X[0, 0, :, :] = image[:, :, 0] - averageImg[0]
        X[0, 1, :, :] = image[:, :, 1] - averageImg[1]
        X[0, 2, :, :] = image[:, :, 2] - averageImg[2]
        out = net.forward_all(data=X)
        feature = np.float64(out['feat_extract'])
        feature = np.reshape(feature, (1, 256))
        fts = np.concatenate((fts, feature))
        labels.append(labelName)

        cv2.waitKey(2)

    cv2.destroyAllWindows()

    fts = fts[1:, :]

    return labels, fts

def compar_pic(features1, features2, predicts):
    predicts.append(pw.cosine_similarity(features1, features2))

def compare_face(root_dir, lbs, fts, result_dir):
    for file in os.listdir(root_dir):
        img = cv2.imdecode(np.fromfile(root_dir + '/' + file, dtype=np.uint8), -1)
        dets = detector(img, 1)
        num_faces = len(dets)
        if num_faces == 0:
            print("Sorry, there were no faces found in current frame")
            continue
        top_allfaces = []
        bottom_allfaces = []
        right_allfaces = []
        left_allfaces = []
        for i in range(num_faces):
            face = [dets[i]]
            for k, d in enumerate(face):
                top_allfaces.append(d.top())
                bottom_allfaces.append(d.bottom())
                right_allfaces.append(d.right())
                left_allfaces.append(d.left())
        faces = dlib.full_object_detections()
        for det in dets:
            faces.append(sp(img, det))
        images = dlib.get_face_chips(img, faces, size=224)
        X = np.empty((num_faces, 3, imgsize, imgsize))
        # BGR
        averageImg = [131.0912, 103.8827, 91.4953]
        for i in range(len(images)):
            image = images[i]
            X[i, 0, :, :] = image[:, :, 0] - averageImg[0]
            X[i, 1, :, :] = image[:, :, 1] - averageImg[1]
            X[i, 2, :, :] = image[:, :, 2] - averageImg[2]
        out = net.forward_all(data=X)
        features = np.float64(out['feat_extract'])
        features = np.reshape(features, (num_faces, 256))
        all_scores = []
        compar_pic(features, fts, all_scores)
        for j in range(num_faces):
            scores = all_scores[0][j].tolist()
            bestscore = max(scores)
            bestscore_index = scores.index(bestscore)
            likelyuser = lbs[bestscore_index]
            if bestscore > thershold:
                cv2.putText(img, likelyuser, (left_allfaces[j], top_allfaces[j]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(img, (left_allfaces[j], top_allfaces[j]),
                                  (right_allfaces[j], bottom_allfaces[j]), (0, 255, 0), 2)
            else:
                ROI_temp = img[top_allfaces[j]:bottom_allfaces[j], left_allfaces[j]:right_allfaces[j], :]
                ROI = cv2.resize(ROI_temp, (imgsize, imgsize), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(result_dir + '/' + str(j) + '.jpg', ROI)

        cv2.imwrite(result_dir + '/' + 'result.jpg', img)

    cv2.destroyAllWindows()

def main(solo_pic_dir, group_pic_dir, result_pic_dir):
    labels, fts = regist_face(solo_pic_dir)
    compare_face(group_pic_dir, labels, fts, result_pic_dir)

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.ResultPicDir):
        os.mkdir(args.ResultPicDir)
    main(args.SoloPicDir, args.GroupPicDir, args.ResultPicDir)




