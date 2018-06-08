from ctypes import *
import math
import random
import numpy as np
import cv2
import os
import time


def original_nms(dets, thresh=0.8):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def filter_boxes_between_class_v1(r):
    if r['customer'].shape[0] != 0 and r['saler'].shape[0] != 0:
        all_boxes = np.concatenate((r['customer'], r['saler']))
        keeps = original_nms(all_boxes)
        num_customer = r['customer'].shape[0]
        r['customer'] = all_boxes[keeps[np.where(keeps < num_customer)[0]]]
        r['saler'] = all_boxes[keeps[np.where(keeps >= num_customer)[0]]]
    else:
        pass
    return r


def filter_boxes_between_class_v2(r):
    if r['customer'].shape[0] != 0 and r['saler'].shape[0] != 0:
        all_boxes = np.concatenate((r['customer'], r['saler']))
        keeps = remove_boxes_in_class(all_boxes)
        num_customer = r['customer'].shape[0]
        r['customer'] = all_boxes[keeps[np.where(keeps < num_customer)[0]]]
        r['saler'] = all_boxes[keeps[np.where(keeps >= num_customer)[0]]]
    else:
        pass
    return r


def remove_boxes_in_class(dets, thresh=0.8):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = areas.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / areas[order[1:]]

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return np.array(keep)


def filter_boxes_in_class(r):
    for key in r.keys():
        if r[key].shape[0] != 0:
            r[key] = r[key][remove_boxes_in_class(r[key])]
    return r


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/zzdxfei/Desktop/rcnn/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

float_to_image = lib.float_to_image
float_to_image.argtypes = [c_int, c_int, c_int, POINTER(c_float)]
float_to_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

transform_image = lib.transform_image
transform_image.argtypes = [IMAGE]

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, im, thresh=.2, hier_thresh=.5, nms=.45):
    # im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    # free_image(im)
    free_detections(dets, num)
    return res


def get_image_names(path):
    imgs = os.listdir(path)
    filted = []
    for item in imgs:
        if item[-3:] == "jpg":
            filted.append(os.path.join(path, item))

    return filted


def convert_r_to_dict(r):
    result = {'customer': [], 'saler': []}
    for item in r:
        label, score, box = item
        x_min = box[0] - box[2] / 2.0
        y_min = box[1] - box[3] / 2.0
        x_max = box[0] + box[2] / 2.0
        y_max = box[1] + box[3] / 2.0
        result[label].append([x_min, y_min, x_max, y_max, score])

    result['customer'] = np.array(result['customer'])
    result['saler'] = np.array(result['saler'])
    return result


def transform_xy(x_min, y_min, x_max, y_max, img_w, img_h, offset=0):
    x_min = max(x_min, 0) + offset
    y_min = max(y_min, 0) + offset
    x_max = min(x_max, img_w - 1) + offset
    y_max = min(y_max, img_h - 1) + offset
    return x_min, y_min, x_max, y_max


def sort_boxes(r):
    if r['customer'].shape[0] != 0:
        scores = r['customer'][:, 4]
        order = scores.argsort()[::-1]
        r['customer'] = r['customer'][order]
    if r['saler'].shape[0] != 0:
        scores = r['saler'][:, 4]
        order = scores.argsort()[::-1]
        r['saler'] = r['saler'][order]
    return r


if __name__ == "__main__":
    data = np.ones((1000, 1000, 3), dtype=np.float32) * 11
    data = data.ravel()
    time_b = time.time()
    a1 = (c_float * len(data)).from_buffer(data)
    # print data.ravel()
    # print len(data.ravel())
    # a1[:] = data
    test_img = float_to_image(1000, 1000, 3, a1)
    time_e = time.time()
    print 1000 * (time_e - time_b), " ms."

    # print data
    # test_img = float_to_image(2, 2, 2, pointer(c_float(a1[0])))

    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]

    # net = load_net("cfg/yolov3-boe.cfg", "boe/yolov3-boe.weights", 0)
    # meta = load_meta("cfg/boe.data")

    net = load_net("cfg/yolov3-shangqi.cfg", "yolov3-shangqi_final.weights", 0)
    meta = load_meta("cfg/shangqi.data")

    # img_lists = get_image_names('/home/zzdxfei/Desktop/rcnn/darknet/test')
    img_lists = get_image_names('/home/zzdxfei/Desktop/rcnn/darknet/shangqi')

    """
    customer_lines = []
    saler_lines = []

    IMAGE_ROOT = './shangqi'
    with open("test.txt", 'r') as fp:
        images = fp.readlines()
        images = [item.strip() for item in images]
        for img in images:
            img_path = os.path.join(IMAGE_ROOT, img + ".jpg")
            img_h, img_w, img_c = cv2.imread(img_path).shape

            r = detect(net, meta, img_path)
            r = convert_r_to_dict(r)
            r = filter_boxes_in_class(r)
            r = filter_boxes_between_class_v1(r)
            r = filter_boxes_between_class_v2(r)
            r = sort_boxes(r)

            for item in r['customer']:
                x_min, y_min, x_max, y_max, score = item
                x_min, y_min, x_max, y_max = transform_xy(
                    x_min, y_min, x_max, y_max, img_w, img_h, 1)
                x_min, y_min, x_max, y_max, score = (
                    str(x_min), str(y_min), str(x_max), str(y_max), str(score))
                
                current = ' '.join([img, score, x_min, y_min, x_max, y_max])
                current += '\n'
                customer_lines.append(current)

            for item in r['saler']:
                x_min, y_min, x_max, y_max, score = item
                x_min, y_min, x_max, y_max = transform_xy(
                    x_min, y_min, x_max, y_max, img_w, img_h, 1)
                x_min, y_min, x_max, y_max, score = (
                    str(x_min), str(y_min), str(x_max), str(y_max), str(score))
                
                current = ' '.join([img, score, x_min, y_min, x_max, y_max])
                current += '\n'
                saler_lines.append(current)

    CUSTOMER_NAME = "customer.txt"
    SALER_NAME = "saler.txt"
    with open(CUSTOMER_NAME, 'w') as fp:
        fp.writelines(customer_lines)
    with open(SALER_NAME, 'w') as fp:
        fp.writelines(saler_lines)
"""

    for img_name in img_lists:
        print "handing ...  ", img_name, "   ",

        # import ipdb
        # ipdb.set_trace(context=30)

        original_img = cv2.imread(img_name)

        time_begin = time.time()
        img = original_img.astype(np.float32)
        h, w, c = img.shape
        img = img[:, :, (2, 1, 0)]
        img = img.swapaxes(1, 2).swapaxes(0, 1)
        img = img.ravel()
        img /= 255.0

        data = (c_float * len(img)).from_buffer(img)
        # print data.ravel()
        # print len(data.ravel())
        # a1[:] = data
        test_img = float_to_image(w, h, c, data)

        r = detect(net, meta, test_img)

        r = convert_r_to_dict(r)
        r = filter_boxes_in_class(r)
        r = filter_boxes_between_class_v1(r)
        r = filter_boxes_between_class_v2(r)
        time_end = time.time()
        print 1000 * (time_end - time_begin), "ms"

        img = original_img
        img_h, img_w, img_c = img.shape

        for item in r['customer']:
            x_min, y_min, x_max, y_max, score = item
            x_min, y_min, x_max, y_max = transform_xy(
                x_min, y_min, x_max, y_max, img_w, img_h)
            x_min, y_min, x_max, y_max, score = (
                int(x_min), int(y_min), int(x_max), int(y_max), str(score)[2:4])
            label = ' C'
            cv2.putText(img, label, (x_min, y_min + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, score, (x_min, y_min + 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        for item in r['saler']:
            x_min, y_min, x_max, y_max, score = item
            x_min, y_min, x_max, y_max = transform_xy(
                x_min, y_min, x_max, y_max, img_w, img_h)
            x_min, y_min, x_max, y_max, score = (
                int(x_min), int(y_min), int(x_max), int(y_max), str(score)[2:4])
            label = ' S'
            cv2.putText(img, label, (x_min, y_min + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, score, (x_min, y_min + 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        output_name = img_name.replace('shangqi', 'shangqi_result')
        cv2.imwrite(output_name, img)
