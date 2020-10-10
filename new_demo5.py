import os
import torch
from torchvision import transforms

from models.fast_scnn import get_fast_scnn
from utils.visualize import get_color_pallete

import numpy as np
import cv2
import gc
import time

import pycuda
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.elementwise import ElementwiseKernel
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import math
from PIL import Image
from keras import backend as K

from keras.models import load_model, Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import json


def ReadJson(JsonFile):
    with open(JsonFile, 'r') as f1:
        content = json.load(f1)
        return content


# 分类模型初始化函数
class Model(object):
    def __init__(self):
        self.model = None

    # 建立一个CNN模型，一层卷积、一层池化、一层卷积、一层池化、抹平之后进行全链接、最后进行分类      其中flatten是将多维输入一维化的函数 dense是全连接层
    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(filters=4, kernel_size=(3, 3), padding='same', dim_ordering='tf',
                              input_shape=(28, 28, 3)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        self.model.add(Dense(32))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        self.model.add(Dense(3))
        self.model.add(BatchNormalization())
        self.model.add(Activation('softmax'))
        self.model.summary()

    def predict(self, img, num):
        # img = cv2.resize(img,(28,28))
        # img = img.reshape((1,  28,28,3))
        img = img.astype('float32')
        img = img / 255.0
        probs = self.model.predict_proba(img, batch_size=num)  # 测算一下该img属于某个label的概率
        classes = self.model.predict_classes(img, batch_size=num)
        return classes, probs  # ,result[0][max_index] #第一个参数为概率最高的label的index,第二个参数为对应概率

    def load_weights(self, path):
        self.model.load_weights(path)


# 一些初始化
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1  # 这个模型很小，不让这个tensoeflow模型占用太多gpu
sess = tf.compat.v1.Session(config=config)
# class_model=model_load()
class_model = Model()
class_model.build_model()
class_model.load_weights('3class_dataset3-ep016-loss0.023-val_acc0.990_2.h5')
print('Model Loaded.')

# 海陆分割模型初始化
dataset = 'citys'
weights_folder = 'yasuoweight'
cpu = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 海陆分割输入图片需要的一些处理，转tensor和归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 海陆分割模型加载
start4 = time.time()
land_sea_model = get_fast_scnn(dataset, pretrained=True, root=weights_folder, map_cpu=cpu).to(device)
land_sea_model.eval()
end4 = time.time()
print('time_load:', end4 - start4)

# pycuda之后要用到的一个逐元素操作函数
gpu_thresh = ElementwiseKernel(
    "float *in, float *out",
    "out[i] = (in[i]>=10)*255;",
    "gpu_thresh"
)

gpu_absthresh = ElementwiseKernel(
    "float *in, float *out",
    "out[i] = (((in[i]>0)*in[i])>=10)*255;",
    "gpu_absthresh"
)

gpu_sub = ElementwiseKernel(
    "float *in1, float *in2,float *out",
    "out[i] = in1[i]-in2[i];",
    "gpu_sub"
)


def land_sea(image, land_sea_model, name, debug=False):
    with torch.no_grad():
        torch.cuda.empty_cache()
        outputs = land_sea_model(image)

    pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()

    mask = pred.astype(np.uint8)  # 不加会报错
    mask[(mask > 1)] = 0

    if debug == True:
        mask2 = get_color_pallete(pred, dataset)  # 244, 35, 232,sea 70, 70, 70,land
        # 这个函数其实是把pred矩阵上了个色
        outname = os.path.splitext(os.path.split(name)[-1])[0] + '.png'
        # outname='1.png'
        mask_path = 'test_out/' + 'mask_' + outname
        mask2.save(mask_path)
    return mask


# 腐蚀
def erode_demo(mask, kernel, h, w, debug=False):
    mask = cv2.erode(mask, kernel)
    mask = cv2.resize(mask, (h, w), interpolation=cv2.INTER_NEAREST)

    if debug == True:
        outname = '1.png'  # os.path.splitext(os.path.split(input1)[-1])[0] + '.png'
        mask_path = 'test_out/' + 'erode_' + outname
        cv2.imwrite(mask_path, mask)
    return mask


# 从大图中截取出小图进行船只云雾二分类
def get_part_pic(im, points):
    p1 = np.min(points, 0).squeeze()
    p2 = np.max(points, 0).squeeze()
    xc = int((p1[0] + p2[0]) / 2)
    yc = int((p1[1] + p2[1]) / 2)
    rect = cv2.minAreaRect(points)

    w = rect[1][0]  # 斜框的w，h 不是正框的
    h = rect[1][1]

    if w > 0 and h > 0:
        if w > h:
            w_h = 1.0 * w / h
        else:
            w_h = 1.0 * h / w
        if w_h > 1.3 and w * h > 10 and w * h < 200:
            crop_im = im[yc - 14:yc + 14, xc - 14:xc + 14]
            return crop_im, np.array([xc, yc, w, h])
    return np.array([1]), np.array([1])


# 船只云雾二分类预测
def get_class(all_im, model):
    num_pic = len(all_im)
    picType, prob = model.predict(all_im, num_pic)
    return picType, prob


def get_class2(all_im, model):
    all_im = all_im.reshape([1, 28, 28, 3])
    picType, prob = model.predict(all_im, 1)
    prob = prob[0]
    # print(prob)
    prob = prob[np.argmax(prob)]
    return picType, prob


# 画图，在图上绘制边界框
def drawObject(im, points, color):
    p1 = np.min(points, 0).squeeze() - 8
    p2 = np.max(points, 0).squeeze() + 8
    cv2.line(im, (p1[0], p1[1]), (p1[0], p2[1]), color)
    cv2.line(im, (p1[0], p1[1]), (p2[0], p1[1]), color)
    cv2.line(im, (p2[0], p2[1]), (p1[0], p2[1]), color)
    cv2.line(im, (p2[0], p2[1]), (p2[0], p1[1]), color)
    return im


def drawObject_xie(im, points, color):
    rect = cv2.minAreaRect(points)
    box = cv2.cv.BoxPoints(rect)
    w = rect[1][0]
    h = rect[1][1]
    if w > 0 and h > 0:
        if w > h:
            w_h = 1.0 * w / h
        else:
            w_h = 1.0 * h / w
        if w_h > 1.3 and w * h > 10 and w * h < 100:
            cv2.line(im, (int(box[0][0]), int(box[0][1])), (int(box[1][0]), int(box[1][1])), color)
            cv2.line(im, (int(box[1][0]), int(box[1][1])), (int(box[2][0]), int(box[2][1])), color)
            cv2.line(im, (int(box[2][0]), int(box[2][1])), (int(box[3][0]), int(box[3][1])), color)
            cv2.line(im, (int(box[3][0]), int(box[3][1])), (int(box[0][0]), int(box[0][1])), color)
    return im


def drawObject_withclass(im, points, color):
    p1 = np.min(points, 0).squeeze() - 8
    p2 = np.max(points, 0).squeeze() + 8
    font = cv2.FONT_HERSHEY_SIMPLEX

    rect = cv2.minAreaRect(points)
    w = rect[1][0]
    h = rect[1][1]
    rect2 = list(rect)
    rect3 = list(rect2[0])
    rect4 = list(rect2[1])

    rect4[0] = w * 1.5
    rect4[1] = h * 1.5
    rect5 = list()
    rect5.append(rect3)
    rect5.append(rect4)
    rect5.append(rect2[2])

    rect5 = tuple(rect5)
    box = cv2.boxPoints(rect5)  # cv2.cv.BoxPoints(rect)

    crop_im, boxs = get_part_pic(im, points)
    if len(crop_im) > 10:
        if crop_im.shape[1] > 27 and crop_im.shape[0] > 27:

            label, prob = get_class2(crop_im, class_model)

            if label == 2:
                color2 = (0, 0, 255)
                cv2.putText(im, str(prob * 100)[:2], (int(box[0][0] - 20), int(box[0][1] - 20)), font, 0.7,
                            (0, 255, 255), 1)
                cv2.line(im, (p1[0], p1[1]), (p1[0], p2[1]), color)
                cv2.line(im, (p1[0], p1[1]), (p2[0], p1[1]), color)
                cv2.line(im, (p2[0], p2[1]), (p1[0], p2[1]), color)
                cv2.line(im, (p2[0], p2[1]), (p2[0], p1[1]), color)

            elif label == 0 or label == 1:
                color2 = (0, 0, 255)
                cv2.line(im, (p1[0], p1[1]), (p1[0], p2[1]), color2)
                cv2.line(im, (p1[0], p1[1]), (p2[0], p1[1]), color2)
                cv2.line(im, (p2[0], p2[1]), (p1[0], p2[1]), color2)
                cv2.line(im, (p2[0], p2[1]), (p2[0], p1[1]), color2)

                # cv2.ocl.setUseOpenCL(False)
                cv2.putText(im, str(prob * 100)[:2], (int(box[0][0] - 20), int(box[0][1] - 20)), font, 0.7,
                            (0, 255, 255), 1)
    return im


def find_object(name, im_pad, kernel, ss, ss2, logname, debug=False, log_debug=False):
    t0 = time.time()
    # transform
    im_resize2 = cv2.resize(im_pad, (ss2, ss2), interpolation=cv2.INTER_NEAREST)
    im_resize = cv2.resize(im_pad, (ss, ss), interpolation=cv2.INTER_NEAREST)
    t1 = time.time()

    if log_debug == True:
        f = open(logname, 'a')
        f.write(time.strftime("%H:%M:%S"))
        f.write(' 缩放成功!\n')
        f.close()

    image = Image.fromarray(cv2.cvtColor(im_resize, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)
    t2 = time.time()

    if log_debug == True:
        f = open(logname, 'a')
        f.write(time.strftime("%H:%M:%S"))
        f.write(' 预处理成功!\n')
        f.close()

    # 海陆分割
    mask = land_sea(image, land_sea_model, name)

    if log_debug == True:
        f = open(logname, 'a')
        f.write(time.strftime("%H:%M:%S"))
        f.write(' 海陆分割成功!\n')
        f.close()

    t3 = time.time()
    # 腐蚀并将mask重新转成原图大小
    mask = erode_demo(mask, kernel, ss2, ss2)

    if debug == True:
        f = open(logname, 'a')
        f.write(time.strftime("%H:%M:%S"))
        f.write(' 腐蚀成功!\n')
        f.close()

    t4 = time.time()
    # 转灰度进行后处理
    im = cv2.cvtColor(im_resize2, cv2.COLOR_BGR2GRAY)

    if log_debug == True:
        f = open(logname, 'a')
        f.write(time.strftime("%H:%M:%S"))
        f.write(' 转灰度成功!\n')
        f.close()

    t44 = time.time()
    im = cv2.GaussianBlur(im, (5, 5), 2)
    tg = time.time()
    img1 = cv2.GaussianBlur(im, (3, 3), 2)
    tg2 = time.time()
    img2 = cv2.blur(im, (50, 50))

    if log_debug == True:
        f = open(logname, 'a')
        f.write(time.strftime("%H:%M:%S"))
        f.write(' blur success!\n')
        f.close()

    tg3 = time.time()
    img1 = gpuarray.to_gpu(img1)
    img2 = gpuarray.to_gpu(img2)
    mask = gpuarray.to_gpu(mask)

    if log_debug == True:
        f = open(logname, 'a')
        f.write(time.strftime("%H:%M:%S"))
        f.write(' to gpu array success!\n')
        f.close()

    t5 = time.time()

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    if log_debug == True:
        f = open(logname, 'a')
        f.write(time.strftime("%H:%M:%S"))
        f.write(' array astype success!\n')
        f.close()

    # im3 =  img1- img2
    gpu_sub(img1, img2, img2)
    im3 = img2

    if log_debug == True:
        f = open(logname, 'a')
        f.write(time.strftime("%H:%M:%S"))
        f.write(' subtraction success!\n')
        f.close()

    tq = time.time()

    gpu_absthresh(im3, im3)

    if log_debug == True:
        f = open(logname, 'a')
        f.write(time.strftime("%H:%M:%S"))
        f.write(' absthresh success!\n')
        f.close()

    t6 = time.time()

    t66 = time.time()
    im3 = im3.astype(np.uint8)  # 加上这个总时间更快，不然乘法会比较慢
    mask = mask.astype(np.uint8)

    im3 = im3 * mask

    if log_debug == True:
        f = open(logname, 'a')
        f.write(time.strftime("%H:%M:%S"))
        f.write(' multiplication success!\n')
        f.close()

    im3 = im3.get()

    if log_debug == True:
        f = open(logname, 'a')
        f.write(time.strftime("%H:%M:%S"))
        f.write(' gpuarray to cpu success!\n')
        f.close()

    # cv版本不一样，输出参数个数不一样
    # binary,contours,hierarchy = cv2.findContours(im3.astype(np.uint8), cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(im3.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if log_debug == True:
        f = open(logname, 'a')
        f.write(time.strftime("%H:%M:%S"))
        f.write(' find contour success!\n')
        f.close()

    t7 = time.time()
    # del im
    # del im3
    # gc.collect()
    # 二分类
    im = im_resize2
    # del im_pad
    all_crop_pic = list()
    all_box = list()
    for p in contours:
        crop_im, box = get_part_pic(im, p)
        if len(crop_im) > 10:
            if crop_im.shape[1] > 27 and crop_im.shape[0] > 27:
                all_crop_pic.append(crop_im)
                all_box.append(box)

    if log_debug == True:
        f = open(logname, 'a')
        f.write(time.strftime("%H:%M:%S"))
        f.write(' crop image success!\n')
        f.close()

    num_pic = len(all_crop_pic)

    ship_box = list()

    if num_pic > 0:
        all_crop_pic = np.array([all_crop_pic]).reshape((num_pic, 28, 28, 3))
        classes, probs = get_class(all_crop_pic, class_model)
        for i in range(len(classes)):
            if classes[i] == 0:
                ship_box.append(all_box[i])
    if log_debug == True:
        f = open(logname, 'a')
        f.write(time.strftime("%H:%M:%S"))
        f.write(' get class success!\n')
        f.close()
    # print(label)
    t8 = time.time()

    if debug == True:
        # 绘制带框的图
        print('resize:', t1 - t0)
        print('transform:', t2 - t1)
        print('fenge:', t3 - t2)
        print('erode:', t4 - t3)
        print('to gray:', t44 - t4)
        print('gassin55blur:', tg - t44)
        print('gassin33blur:', tg2 - tg)
        print('blur5050:', tg3 - tg2)
        print('to gpu:', t5 - tg3)
        print('abs:', t6 - tq)
        # print('astype:',t666-t5)
        print('jianfa:', tq - t5)
        print('thresh:', t66 - t6)
        print('find contour:', t7 - t66)
        # print('chuli:',t7-t4)
        print(num_pic)
        # if num_pic>0:
        # print('get class:',t8-aa)
        print('classify:', t8 - t7)
        print('total:', t8 - t0)

        for p in contours:
            if ((len(p) <= 60) & (len(p) >= 4)):
                img = drawObject_withclass(im, p, (0, 255, 0))
        img_str = cv2.imencode('.jpg', img)[1].tostring()
        print('write')
    # return ship_box
    return img_str


def demo(input_value, k, s, ss, ss2, debug=True, log_debug=True):
    # 腐蚀的一些参数，预先设定好

    if log_debug == True:
        logname = time.strftime("%Y_%m_%d") + '.txt'
        f = open(logname, 'w')
        f.write(time.strftime("%H:%M:%S"))
        f.write(' start success!\n')
        f.close()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    # 图片读取
    input_value = np.fromstring(input_value, np.uint8)
    image_origin = cv2.imdecode(input_value, cv2.IMREAD_UNCHANGED)  # Image.open(input1).convert('RGB')
    h, w, c = image_origin.shape
    im_pad = image_origin[0:s, 0:s]
    img_stream = find_object('test.jpg', im_pad, kernel, ss, ss2, logname, debug, log_debug)
    # print(ship_box)
    return img_stream


if __name__ == "__main__":
    s = 6912  # +3200  #测试的图片尺寸，32的倍数
    ss = 640  # ladn_sea fenge size
    ss2 = s  # findcontour size
    demo('test_jpg', 89, s, ss, ss2)  # 89 is erode kernel size
'''
JsonPara = ReadJson('low_res_parm.json')
s = JsonPara['origin_size']  #测试的图片尺寸，32的倍数   
kernel_size = JsonPara['erode_kernel_size']
ss = JsonPara['land_sea_segsize']  #ladn_sea fenge size
ss2 = JsonPara['find_object_size']  #findcontour size
'''
