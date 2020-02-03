import torch
import torch.nn as nn
import gzip
import os
import argparse
import random
import numpy as np
from model import LeNet
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from matplotlib import pyplot as plt
from MNIST import MnistData
import matplotlib.image as mpimg
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(30, 10, 681, 541))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox = QtWidgets.QGroupBox(self.horizontalLayoutWidget)
        self.groupBox.setObjectName("groupBox")
        self.ShowImage = QtWidgets.QPushButton(self.groupBox)
        self.ShowImage.setGeometry(QtCore.QRect(10, 30, 221, 71))
        self.ShowImage.setObjectName("ShowImage")
        self.ShowImage.clicked.connect(self.button1)
        self.Parameter = QtWidgets.QPushButton(self.groupBox)
        self.Parameter.setGeometry(QtCore.QRect(10, 120, 221, 81))
        self.Parameter.setObjectName("Parameter")
        self.Parameter.clicked.connect(self.button2)
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 210, 221, 91))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.button3)
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QtCore.QRect(10, 310, 221, 91))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.clicked.connect(self.button4)
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_5.setGeometry(QtCore.QRect(10, 410, 221, 91))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.clicked.connect(self.button5)
        self.textEdit = QtWidgets.QLineEdit(self.groupBox)
        self.textEdit.setGeometry(QtCore.QRect(450, 420, 201, 70))
        self.textEdit.setObjectName("textEdit")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(250, 430, 171, 51))
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.groupBox)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "GroupBox"))
        self.ShowImage.setText(_translate("MainWindow", "5.1 Show Image"))
        self.Parameter.setText(_translate("MainWindow", "5.2 Show Hyperparameters"))
        self.pushButton_3.setText(_translate("MainWindow", "5.3 Train one epoch"))
        self.pushButton_4.setText(_translate("MainWindow", "5.4 Show Trainning Result"))
        self.pushButton_5.setText(_translate("MainWindow", "5.5 Inference"))
        self.label.setText(_translate("MainWindow", "Test Image index:0~9999"))
    def button1(self):
        showImage()
    def button2(self):
        printParameter()
    def button3(self):
        trainModel(EPOCH_NUM=1,show=True)
    def button4(self):
        plt.figure()
        img = mpimg.imread('accuracy_loss.jpg')
        plt.imshow(img)
        plt.show()
    def button5(self):
        index =  self.textEdit.text()
        Inference(int(index))
    

def get_data(image_url,label_url):
    images = gzip.GzipFile(image_url,'rb').read()
    labels = gzip.GzipFile(label_url,'rb').read()
    img_num = int.from_bytes(images[4:8],byteorder='big')
    label_num = int.from_bytes(labels[4:8], byteorder='big')
    assert (img_num == label_num)
    row = int.from_bytes(images[8:12],byteorder ='big')
    col = int.from_bytes(images[12:16],byteorder='big')
    img_size = row * col
    x,y=[],[]
    for i in range(img_num):
        img_offset = 16 + img_size *i
        lbl_offest = 8 + i
        img = torch.Tensor(list(images[img_offset:img_offset+img_size])).float()
        img = img.view(1,row,col)
        lbl = int(labels[lbl_offest])
        x.append(img)
        y.append(lbl)
    return x,y

def trainModel(EPOCH_NUM=50,save=False,show=False):
    print("Train Start:")
    net = LeNet().to(devices)
    criterion = nn.CrossEntropyLoss()
    optimizer =torch.optim.RMSprop(net.parameters(),lr=LR,alpha=0.9,eps=1e-08, weight_decay=0, momentum=0, centered=False)
    #x,trainloss,trainacc,testacc = [],[],[],[]
    batch,batchloss =[],[]
    for epoch in range(EPOCH_NUM):
        sum_loss = 0.0
        acc = 0
        iter = 0
        for i, (inputs, labels) in enumerate(trainLoader):
            inputs, labels = inputs.to(devices), labels.to(devices)
            # forward and backward
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            _ ,pred = torch.max(outputs.data,1)
            acc += (pred == labels).sum()
            iter = iter +1
            batch.append(i)
            batchloss.append(loss.item())
        if show == True :
            plt.figure()
            plt.plot(batch, batchloss, 'b')
            plt.title('one epoch')
            plt.xlabel('iteration')
            plt.ylabel('loss')
            plt.show()
       # trainloss.append(sum_loss/iter)
       # trainacc.append(100*acc/len(trainData))
       # x.append(epoch)
        print('Epoch [%d] : loss [%f]'%(epoch+1,sum_loss/iter))
        print('train accuracy = %f%%'%(100*acc/len(trainData)))
        #with torch.no_grad():
        #    correct = 0
        #    total = 0
        #    for data in testLoader:
        #        images, labels = data
        #        images, labels = images.to(devices), labels.to(devices)
        #        outputs = net(images)
        #        _, predicted = torch.max(outputs.data, 1)
        #        total += labels.size(0)
        #        correct += (predicted == labels).sum()
        #print('test accuracy = %f%%'%(100*correct/total))
        #testacc.append(100*correct/total)
    if save == True:
        torch.save(net.state_dict(), 'MNIST_Model.pth')
    #return x,trainloss,trainacc,testacc

root = os.getcwd()
# CPU or GPU
devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#parameter
BATCH_SIZE = 32
LR = 0.001

# get training data

trainImage,trainLabel = get_data(os.path.join(root,"dataset/train-images-idx3-ubyte.gz"),os.path.join(root,"dataset/train-labels-idx1-ubyte.gz"))
trainData = MnistData(trainImage,trainLabel)
trainLoader = DataLoader(dataset=trainData,
                         batch_size=BATCH_SIZE,
                         shuffle=True )
# get testing data
testImage,testLabel = get_data(os.path.join(root,"dataset/t10k-images-idx3-ubyte.gz"),os.path.join(root,"dataset/t10k-labels-idx1-ubyte.gz"))
testData = MnistData(testImage,testLabel)
testLoader = DataLoader(dataset=testData,
                         batch_size=BATCH_SIZE,
                         shuffle=True )

def showImage(number = 10) :
    fig,ax = plt.subplots(nrows=1,ncols=10,sharex='all',sharey='all')
    ax = ax.flatten()
    len =  trainData.__len__()
    for i in range(number):
        x = random.randint(0,len-1)
        img,label = trainData.__getitem__(x)
        ax[i].imshow(img.reshape(28,28),cmap='Greys',interpolation='nearest')
        ax[i].set_title(label)
    plt.show()

def printParameter():
    print('hyperparameters:')
    print('batch size:',BATCH_SIZE)
    print('learning rate:',LR)
    print ('optimizer:SGD')

def plot(x,loss,trainacc,testacc):
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(x,loss,'-r',linewidth=1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss function')
    plt.subplot(3,1,3)
    plt.xlabel('epoch')
    plt.ylim(0,100)
    plt.ylabel('%')
    p1,=plt.plot(x,trainacc,'-b',label='train')
    p2,=plt.plot(x,testacc,'-r',label='test')
    plt.legend(handles=[p1,p2],labels=['train','test'],loc='lower right')
    plt.title('accuracy')
    plt.savefig("accuracy_loss.jpg")
    plt.show()
def Inference(index):
    model = LeNet()
    model.load_state_dict(torch.load('MNIST_Model.pth'))
    img,label= testData.__getitem__(index)
    img = img.unsqueeze(0)
    output = model(img)
    print(output)
    output = output.tolist()
    output = [i for i in output[0]]
    x = [0,1,2,3,4,5,6,7,8,9] 
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(img.reshape(28,28))
    plt.subplot(2,1,2)
    plt.ylim(0,1)
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.xlim(0,9)
    plt.bar(x,output)
    plt.show()

if(__name__ == '__main__'):
   import sys
   app = QtWidgets.QApplication(sys.argv)
   MainWindow = QtWidgets.QMainWindow()
   ui = Ui_MainWindow()
   ui.setupUi(MainWindow)
   MainWindow.show()
   sys.exit(app.exec_())
   #trainModel(EPOCH_NUM=50,save=True)
