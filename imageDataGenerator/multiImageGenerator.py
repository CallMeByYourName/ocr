from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import cv2
from PIL import Image

datagen = ImageDataGenerator(rotation_range=5,    # 随机旋转的度数
                             width_shift_range=0.05,    # 随机水平平移
                             height_shift_range=0.05,   # 随机垂直平移
                             rescale=1/255,            # 数据归一化
                             shear_range=0.5,           # 随机错切变换
                             zoom_range=0.05,           # 随机放大
                             horizontal_flip=False,     # 水平翻转
                             fill_mode='nearest',      # 填充方式
                             )

path = 'D:/imgae classification/imageDataGenerator/image'
j = 0
num = 4    #要生成的图片数量
while j < num:
    for file in os.listdir(path):
        image = load_img(os.path.join(path, file))
        image = image.resize((28,28),Image.ANTIALIAS)
        # img = load_img(os.path.join(path,file))
        x = img_to_array(image)
        # x = np.expand_dims(img, 0)
        x = np.expand_dims(x, 0)
        for batch in datagen.flow(x, batch_size=1, save_to_dir='tmp', save_prefix=file.split('.png')[0], save_format='png'):
            j = j + 1
            break
        if j == num:
            break

print("done")


