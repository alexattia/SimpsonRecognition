import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from faster_rcnn import config, data_generators
import faster_rcnn.resnet as nn
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from faster_rcnn import roi_helpers
overlap_thresh = 0.2
bbox_threshold = 0.5

def format_img(img, C):
    """
    Normalize and resize image to have the smallest side equal to 600
    """
    img_min_side = float(C.im_size)
    (height,width,_) = img.shape
    (resized_width, resized_height, ratio) = data_generators.get_new_img_size(width, height, C.im_size)
    img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
    img = data_generators.normalize_img(img, C)
    return img, ratio

def get_real_coordinates(ratio, x1, y1, x2, y2):
    """
    Method to transform the coordinates of the bounding box to its original size
    """
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)

def get_models(C):
    """
    Create models : rpn, classifier and classifier only
    :param C: config object
    :return: models
    """
    img_input = Input(shape=(None, None, 3))
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=(None, None, 1024))

    # define the base network (resnet here)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    # define the classifer, built on the feature map
    classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping), trainable=True)

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')
    return model_rpn, model_classifier, model_classifier_only

def detect_predict(pic, C, model_rpn, model_classifier, model_classifier_only, class_mapping, class_to_color, print_dets=False, export=False):
    """
    Detect and predict object in the picture
    :param pic: picture numpy array
    :param C: config object
    :params model_*: models from get_models function
    :params class_*: mapping and colors, need to be loaded to keep the same colors/classes 
    :return: picture with bounding boxes 
    """
    img = pic 
    X, ratio = format_img(img, C)

    img_scaled = np.transpose(X.copy()[0, (2, 1, 0), :, :], (1, 2, 0)).copy()
    img_scaled[:, :, 0] += 123.68
    img_scaled[:, :, 1] += 116.779
    img_scaled[:, :, 2] += 103.939
    img_scaled = img_scaled.astype(np.uint8)

    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))

    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)
    

    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}
    # print(class_mapping)
    for jk in range(R.shape[0]//C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0]//C.num_rois:
            #pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        for ii in range(P_cls.shape[1]):

            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []
    boxes_export = {}
    for key in bboxes:
        bbox = np.array(bboxes[key])
        # Eliminating redundant object detection windows
        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=overlap_thresh)
        
        # Keep only the best prediction per character
        jk = np.argmax(new_probs)
        
        # Threshold for best prediction
        if new_probs[jk] > 0.55:
            (x1, y1, x2, y2) = new_boxes[jk,:]

            # Convert predicted picture box coordinates to real-size picture coordinates
            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
            
            # Exporting box coordinates instead of draw on the picture
            if export:
                boxes_export[key] = [(real_x1, real_y1, real_x2, real_y2), int(100*new_probs[jk])]

            else:
                cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

                textLabel = '{}: {}%'.format(key,int(100*new_probs[jk]))
                all_dets.append((key,100*new_probs[jk]))

                (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)

                # To avoid putting text outside the frame
                # replace the legende if the box is outside the image
                if real_y1 < 20 and real_y2 < img.shape[0]:
                    textOrg = (real_x1, real_y2+5)
                    
                elif real_y1 < 20 and real_y2 > img.shape[0]:
                    textOrg = (real_x1, img.shape[0]-10)
                else:
                    textOrg = (real_x1, real_y1+5)

                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
                cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)


    if print_dets:
        print(all_dets)
    if export:
        return boxes_export
    else:
        return img

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
    (options, args) = parser.parse_args()
    if not options.test_path:   # if filename is not given
        parser.error('Error: path to test data must be specified. Pass --path to command line')

    ## Load config object
    with open('./config.pickle', 'rb') as f_in:
        C = pickle.load(f_in)

    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False
    model_rpn, model_classifier, model_classifier_only = get_models(C)
    class_mapping = C.class_mapping
    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)
    class_mapping = {v: k for k, v in class_mapping.items()}
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

    for idx, img_name in enumerate(sorted(os.listdir(options.test_path))):
        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue
        print(img_name)
        filepath = os.path.join(options.test_path,img_name)
        img = cv2.imread(filepath)
        st = time.time()
        img = detect_predict(img, C, model_rpn, model_classifier, model_classifier_only, class_mapping, class_to_color, True)
        print('Elapsed time = {}'.format(time.time() - st))
        cv2.imwrite('./results_test/result_{}.png'.format(img_name.replace('.png', '')),img)
