import gzip
import os
import pickle
import sys
import wget
import numpy as np
import cv2

from PIL import Image
from network import NeuralNetwork
import matplotlib.pyplot as plt

#读取训练数据，验证数据，测试数据
def load_mnist():
    if not os.path.exists(os.path.join(os.curdir, "data")):
        os.mkdir(os.path.join(os.curdir, "data"))
        wget.download("http://deeplearning.net/data/mnist/mnist.pkl.gz", out="data")

    path = "D:/imgae classification/9.20/digit-classifier-master/data3"

    data_file = gzip.open(os.path.join(os.curdir, "data", "mnist.pkl.gz"), "rb")
    train_data, val_data, test_data = pickle.load(data_file, encoding="latin1")
    data_file.close()

    #原本的训练集
    train_inputs = [np.reshape(x, (784, 1)) for x in train_data[0]]
    train_results = [vectorized_result(y) for y in train_data[1]]
    train_data = list(zip(train_inputs, train_results))

    #新训练集
    train_path = os.path.join(path,"train")
    train_input_1 = []
    train_lable = []
    for i in os.listdir(train_path):
        img = cv2.resize(np.array(Image.open(os.path.join(train_path,i)).convert('L'), 'f')/255,(28,28))
        # if i == "R_155.png":
        #     plt.imshow(img)
        #     plt.show()
        img = np.reshape(img, (784,1))
        train_input_1.append(img)
        if i.split('_')[0] == "0":
            train_lable.append(0) #标签
        elif i.split('_')[0] == "5":
            train_lable.append(1) #标签
        elif i.split('_')[0] == "10":
            train_lable.append(2) #标签
        elif i.split('_')[0] == "-10":
            train_lable.append(3) #标签
        elif i.split('_')[0] == "20":
            train_lable.append(4) #标签
        elif i.split('_')[0] == "-20":
            train_lable.append(5) #标签
        elif i.split('_')[0] == "30":
            train_lable.append(6) #标签
        elif i.split('_')[0] == "-30":
            train_lable.append(7) #标签
        elif i.split('_')[0] == "35":
            train_lable.append(8) #标签
        elif i.split('_')[0] == "40":
            train_lable.append(9) #标签
        elif i.split('_')[0] == "-40":
            train_lable.append(10) #标签
        elif i.split('_')[0] == "50":
            train_lable.append(11) #标签
        elif i.split('_')[0] == "-50":
            train_lable.append(12) #标签
        elif i.split('_')[0] == "55":
            train_lable.append(13) #标签
        elif i.split('_')[0] == "60":
            train_lable.append(14) #标签
        elif i.split('_')[0] == "70":
            train_lable.append(15) #标签
        elif i.split('_')[0] == "80":
            train_lable.append(16) #标签
        elif i.split('_')[0] == "90":
            train_lable.append(17) #标签
        elif i.split('_')[0] == "100":
            train_lable.append(18) #标签
        elif i.split('_')[0] == "110":
            train_lable.append(19) #标签
        elif i.split('_')[0] == "120":
            train_lable.append(20) #标签
        elif i.split('_')[0] == "130":
            train_lable.append(21) #标签
        elif i.split('_')[0] == "135":
            train_lable.append(22) #标签
        elif i.split('_')[0] == "140":
            train_lable.append(23) #标签
        elif i.split('_')[0] == "150":
            train_lable.append(24) #标签
        elif i.split('_')[0] == "160":
            train_lable.append(25) #标签
        elif i.split('_')[0] == "170":
            train_lable.append(26) #标签
        elif i.split('_')[0] == "180":
            train_lable.append(27) #标签
    train_lable = [vectorized_result(y) for y in train_lable]
    train_data_1 = list(zip(train_input_1, train_lable))

    #原本的验证集
    val_inputs = [np.reshape(x, (784, 1)) for x in val_data[0]]
    val_results = val_data[1]
    val_data = list(zip(val_inputs, val_results))

    #新的验证集
    val_path = os.path.join(path,"val")
    val_input_1 = []
    val_lable = []
    for i in os.listdir(val_path):
        img = cv2.resize(np.array(Image.open(os.path.join(val_path,i)).convert('L'), 'f')/255,(28,28))
        # if i == "R_155.png":
        #     plt.imshow(img)
        #     plt.show()
        img = np.reshape(img, (784,1))
        val_input_1.append(img)
        if i.split('_')[0] == "0":
            val_lable.append(0) #标签
        elif i.split('_')[0] == "5":
            val_lable.append(1) #标签
        elif i.split('_')[0] == "10":
            val_lable.append(2) #标签
        elif i.split('_')[0] == "-10":
            val_lable.append(3) #标签
        elif i.split('_')[0] == "20":
            val_lable.append(4) #标签
        elif i.split('_')[0] == "-20":
            val_lable.append(5) #标签
        elif i.split('_')[0] == "30":
            val_lable.append(6) #标签
        elif i.split('_')[0] == "-30":
            val_lable.append(7) #标签
        elif i.split('_')[0] == "35":
            val_lable.append(8) #标签
        elif i.split('_')[0] == "40":
            val_lable.append(9) #标签
        elif i.split('_')[0] == "-40":
            val_lable.append(10) #标签
        elif i.split('_')[0] == "50":
            val_lable.append(11) #标签
        elif i.split('_')[0] == "-50":
            val_lable.append(12) #标签
        elif i.split('_')[0] == "55":
            val_lable.append(13) #标签
        elif i.split('_')[0] == "60":
            val_lable.append(14) #标签
        elif i.split('_')[0] == "70":
            val_lable.append(15) #标签
        elif i.split('_')[0] == "80":
            val_lable.append(16) #标签
        elif i.split('_')[0] == "90":
            val_lable.append(17) #标签
        elif i.split('_')[0] == "100":
            val_lable.append(18) #标签
        elif i.split('_')[0] == "110":
            val_lable.append(19) #标签
        elif i.split('_')[0] == "120":
            val_lable.append(20) #标签
        elif i.split('_')[0] == "130":
            val_lable.append(21) #标签
        elif i.split('_')[0] == "135":
            val_lable.append(22) #标签
        elif i.split('_')[0] == "140":
            val_lable.append(23) #标签
        elif i.split('_')[0] == "150":
            val_lable.append(24) #标签
        elif i.split('_')[0] == "160":
            val_lable.append(25) #标签
        elif i.split('_')[0] == "170":
            val_lable.append(26) #标签
        elif i.split('_')[0] == "180":
            val_lable.append(27) #标签
    val_lable = np.array(val_lable)
    val_data_1 = list(zip(val_input_1, val_lable))

    # for i in range(0,9):
    #     img = np.array(np.reshape(test_data[0][i],(28,28)))
    #     plt.imshow(img)
    #     plt.show()
    #     print(test_data[1][i])

    #原本的测试集
    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))
    # return train_data, val_data, test_data

    #新的测试集
    test_path = os.path.join(path,"test")
    test_input_1 = []
    test_lable = []
    for i in os.listdir(test_path):
        img = cv2.resize(np.array(Image.open(os.path.join(test_path,i)).convert('L'), 'f')/255,(28,28))
        # if i == "R_155.png":
        #     plt.imshow(img)
        #     plt.show()
        img = np.reshape(img, (784,1))
        test_input_1.append(img)
        if i.split('_')[0] == "0":
            test_lable.append(0) #标签
        elif i.split('_')[0] == "5":
            test_lable.append(1) #标签
        elif i.split('_')[0] == "10":
            test_lable.append(2) #标签
        elif i.split('_')[0] == "-10":
            test_lable.append(3) #标签
        elif i.split('_')[0] == "20":
            test_lable.append(4) #标签
        elif i.split('_')[0] == "-20":
            test_lable.append(5) #标签
        elif i.split('_')[0] == "30":
            test_lable.append(6) #标签
        elif i.split('_')[0] == "-30":
            test_lable.append(7) #标签
        elif i.split('_')[0] == "35":
            test_lable.append(8) #标签
        elif i.split('_')[0] == "40":
            test_lable.append(9) #标签
        elif i.split('_')[0] == "-40":
            test_lable.append(10) #标签
        elif i.split('_')[0] == "50":
            test_lable.append(11) #标签
        elif i.split('_')[0] == "-50":
            test_lable.append(12) #标签
        elif i.split('_')[0] == "55":
            test_lable.append(13) #标签
        elif i.split('_')[0] == "60":
            test_lable.append(14) #标签
        elif i.split('_')[0] == "70":
            test_lable.append(15) #标签
        elif i.split('_')[0] == "80":
            test_lable.append(16) #标签
        elif i.split('_')[0] == "90":
            test_lable.append(17) #标签
        elif i.split('_')[0] == "100":
            test_lable.append(18) #标签
        elif i.split('_')[0] == "110":
            test_lable.append(19) #标签
        elif i.split('_')[0] == "120":
            test_lable.append(20) #标签
        elif i.split('_')[0] == "130":
            test_lable.append(21) #标签
        elif i.split('_')[0] == "135":
            test_lable.append(22) #标签
        elif i.split('_')[0] == "140":
            test_lable.append(23) #标签
        elif i.split('_')[0] == "150":
            test_lable.append(24) #标签
        elif i.split('_')[0] == "160":
            test_lable.append(25) #标签
        elif i.split('_')[0] == "170":
            test_lable.append(26) #标签
        elif i.split('_')[0] == "180":
            test_lable.append(27) #标签
    test_lable = np.array(test_lable)
    # print(test_lable)
    test_data_1 = list(zip(test_input_1, test_lable))

    return train_data_1, val_data_1, test_data_1



#验证集的标签
def vectorized_result(y):
    e = np.zeros((28, 1))
    e[y] = 1.0
    return e


if __name__ == "__main__":
    np.random.seed(42)

    layers = [784, 30, 28]
    learning_rate = 0.01
    mini_batch_size = 16
    epochs = 200

    # Initialize train, val and test data
    train_data, val_data, test_data = load_mnist()


    # print(len(test_data))
    nn = NeuralNetwork(layers, learning_rate, mini_batch_size, "relu")
    nn.fit(train_data, val_data, epochs)
    # nn.load()

    accuracy = nn.validate(test_data,sign=1)*100 / len(test_data)
    print(f"Test Accuracy: {accuracy}%.")

    nn.save()
