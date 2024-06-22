import os
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from PIL import Image


# 从文件夹中读取图片路径
def getPath(filePath):
    paths = []
    for root, dirs, files in os.walk(filePath):
        if root != filePath:
            break
        for file in files:
            path = os.path.join(root, file)
            paths.append(path)
            # print(path)
            # 检查路径无误
    for i in paths:
        if 'txt' in i:
            index = paths.index(i)
    paths.remove(paths[index])
    # print(paths, "ok")
    # 检查验证没有问题
    return paths


def getImages(filePath):
    paths = getPath(filePath)
    img = [cv2.imread(x) for x in paths]
    return img


# 从txt文件中读出曝光时间
def getTimes(path):
    times = []
    with open(path, 'r') as fp:
        line = [x.strip() for x in fp]
    for i in line:
        # times.append(eval(i[6:]))
        times.append(eval(i[0:]))
        # 给出的exposures是直接上数据
    times = np.array(times).astype(np.float32)
    return times


images = []
times = []
lists = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
for i in range(len(lists)):
    list = './Dataset/Test/EXTRA/' + lists[i] + '/'
    # print(list)
    images.append(getImages(list))
    times.append(getTimes(list + "exposure.txt"))

# print(images)
times = np.resize(times, (10, 3))
Ldr_images = []
Hdr_images = []
j = 0
while j < 10:
    i = 0
    while i < 4:
        if i < 3:
            Ldr_images.append(images[j][i])
        else:
            Hdr_images.append(images[j][i])
        i = i + 1
    j = j + 1


# print(Ldr_images[ 0 ])
# print(Ldr_images[ 0 ].shape)
# print(Hdr_images[ 0 ])
# print(Hdr_images[ 0 ].shape)


# print(times.shape)
# print(times[0][0])
# plt.imshow(images[1][0])
# plt.show()
def transform_function(images, jamx, imax):
    transform = transforms.Compose([
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    j = 0
    while j < jamx:
        i = 0
        while i < imax:
            pic_1 = Image.fromarray(images[j * imax + i])
            pic_2 = transform(pic_1)
            pic_3 = transforms.ToPILImage()(pic_2)
            # plt.imshow(pic_3)
            # plt.show()
            # print("第  个文件的第  张图片imshow出来", j + 1, i + 1)
            i = i + 1
        j = j + 1


textnum = 1
transform_function(Ldr_images, 10, 3)
transform_function(Hdr_images, 10, 1)
