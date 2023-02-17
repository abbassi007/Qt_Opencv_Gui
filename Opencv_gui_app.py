# -*- coding: utf-8 -*-
"""
Computer Vision & Image Processing Gui App
                Qt

Created on Fri Feburary 17, 2023
Time: 10:46:22
@author: Shahid Abbas

"""


import random
import numba as nb
import numpy as np

import cv2, imutils
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
from PyQt5.QtGui import (QPixmap, QImage)
from PyQt5.QtWidgets import(

    QFileDialog,
    QGraphicsScene,
    QGraphicsView,
  
)
from PyQt5.QtCore import(Qt)
import sys


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super(Window,self).__init__()
        loadUi("Opencv_gui_app.ui",self)
        
        #----------Window parameters----------
        self.factor = 1.15
        self.load = False
        self.temp_spinZoom_value = 0
        self.spinBox_KernelSize.setValue(1)
        #self.temp_spinBrigthness_value = 0
        

        #----------File dropdown buttons----------
        self.actionOpen.triggered.connect(self.LoadImage)
        self.actionNew.triggered.connect(self.LoadImage)
        self.actionSave_Ctrl_S.triggered.connect(self.SaveImage)
        self.actionClose_Ctrl_F4.triggered.connect(self.CloseWindow)
        self.actionQuit.triggered.connect(self.CloseWindow)
        self.actionPrint.triggered.connect(self.PrintImage)

        #----------ZoomIn and Zoomout Buttoons----------
        self.actionZoom_In.triggered.connect(self.ZoomIn)
        self.actionZoom_Out.triggered.connect(self.ZoomOut)
        self.actionFit.triggered.connect(self.FitWindow)

        #----------Controls spinbox---------
        self.spinBox_Zoom.valueChanged.connect(self.SpinBoxZoom)
        self.spinBox_Brightness.valueChanged.connect(self.SpinBoxBrightness)
        self.spinBox_Contrast.valueChanged.connect(self.SpinBoxContrast)

        #----------Kernel Size Spin------
        self.spinBox_KernelSize.valueChanged.connect(self.Filters)

        #-----------Filrers----------
        self.gau_checker = self.checkBox_Gaussian
        self.med_checker = self.checkBox_Median
        self.ave_checker = self.checkBox_Average
        self.lap_checker = self.checkBox_Laplacian
        self.sob_checker = self.checkBox_Sobel
        self.sch_checker = self.checkBox_Scharr
        self.mor_checker = self.checkBox_Morphological
        self.checkBox_Gaussian.stateChanged.connect(self.GaussianFilter)
        self.checkBox_Median.stateChanged.connect(self.MedianFilter)
        self.checkBox_Average.stateChanged.connect(self.AverageFilter)
        self.checkBox_Laplacian.stateChanged.connect(self.LaplacianFilter)
        self.checkBox_Sobel.stateChanged.connect(self.SobelFilter)
        self.checkBox_Scharr.stateChanged.connect(self.ScharrFilter)
        self.checkBox_Morphological.stateChanged.connect(self.MorphologicalFilter)

        #----------Color Space---------
        self.radioButton_RGB.toggled.connect(self.YCbCr2RGB)
        self.radioButton_YCbCr.toggled.connect(self.RGB2YCbCr)
        self.radioButton_Grayscale.toggled.connect(self.GrayScale)

        #----------Noise----------
        self.radioButton_Salt.toggled.connect(self.SaltNoise)
        self.radioButton_Gaussian.toggled.connect(self.GaussianNoise)
        self.radioButton_Pencil.toggled.connect(self.PencilSketch)
        self.radioButton_Cartoon.toggled.connect(self.CartoonSketch)
        self.radioButton_Oilpainting.toggled.connect(self.OilpaintingSketch)
        
        

    def LoadImage(self):
        self.load = True
        self.file_path = QFileDialog.getOpenFileName(self,"Open","C:/", "Files (*.png *.jpg *.bmp )")

        if(self.file_path[0] != ''):
            self.image = cv2.imread(self.file_path[0])
            self.SetImage(self.image)
            
        else:
            print('Please open file')


    def SetImage(self,image):

        self.o_image = image
        #image = imutils.resize(image,width=640)
        frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
        self.image = QImage(frame,frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)

        self.scene = QGraphicsScene()
        self.scene.addItem(self.input_image.setPixmap(QPixmap.fromImage( self.image.scaled(self.input_image.size()))))
        self.scene.addItem(self.output_image.setPixmap(QPixmap.fromImage( self.image.scaled(self.output_image.size()))))
        self.view = QGraphicsView(self.scene,self)


    def  SaveImage(self):
        filename = QFileDialog.getSaveFileName(filter="JPG(*.jpg);;PNG(*.png);;TIFF(*.tiff);;BMP(*.bmp)")

        if(filename[0] !=''):
            cv2.imwrite(filename[0],self.tmp)
            print('Image saved as:', filename)
        else:
            print('Please gave a name to image')

    def FitWindow(self):
        if(self.load):
            self.input_image.setPixmap(QPixmap.fromImage( self.image.scaled(self.input_image.size())))
            self.output_image.setPixmap(QPixmap.fromImage( self.image.scaled(self.output_image.size())))
        else:
            print("Please open image :")

    def ZoomIn(self):
        if(self.load):
            self.input_image.resize(self.input_image.size()*self.factor)
            self.output_image.resize(self.output_image.size()*self.factor)
        else:
            print("Please open image :")

    def ZoomOut(self):
        if(self.load):
            self.input_image.resize(self.input_image.size()/self.factor)
            self.output_image.resize(self.output_image.size()/self.factor)
        else:
            print("Please open image :")

    def SpinBoxZoom(self):
        if(self.load):
        
            self.spinBox_Zoom.setRange(-100,100)
            value = self.spinBox_Zoom.value()
            if(value > self.temp_spinZoom_value):
                self.input_image.resize(self.input_image.size()*self.factor)
                self.output_image.resize(self.output_image.size()*self.factor)
                self.temp_spinZoom_value = value
            else:
                self.input_image.resize(self.input_image.size()/self.factor)
                self.output_image.resize(self.output_image.size()/self.factor)
                self.temp_spinZoom_value = value
        else:
            print("Please open image :")

    def SpinBoxBrightness(self):
        
        self.spinBox_Brightness.setRange(0,100)
        value = self.spinBox_Brightness.value()

        hsv = cv2.cvtColor(self.o_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim ] += value

        final_hsv = cv2.merge((h,s,v))
        frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img=QImage(frame,frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)

        self.output_image.setPixmap(QPixmap.fromImage(img.scaled(self.output_image.size())))

        self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
 

    def SpinBoxContrast(self):

        self.spinBox_Contrast.setRange(0,100)
        value = self.spinBox_Contrast.value() 
        hsv = cv2.cvtColor(self.o_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        lim = 255 - value
        s[s > lim] = 255
        s[s <= lim ] += value

        final_hsv = cv2.merge((h,s,v))
        frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img=QImage(frame,frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)

        self.output_image.setPixmap(QPixmap.fromImage(img.scaled(self.output_image.size())))

        self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))

    
    def CloseWindow(self):
        self.close()

    def PrintImage(self):
        self.print()

    def wheelEvent(self, event: QtGui.QWheelEvent):
        if (self.load):
            if(event.angleDelta().y() > 0):
                self.input_image.resize(self.input_image.size()*self.factor)
                self.output_image.resize(self.output_image.size()*self.factor)
                
            else:
                self.input_image.resize(self.input_image.size()/self.factor)
                self.output_image.resize(self.output_image.size()/self.factor)
        else:
            print("Please open image :")


    def Filters(self):

        if(self.gau_checker.isChecked()):
            self.GaussianFilter()
        elif(self.med_checker.isChecked()):
            self.MedianFilter()
        elif(self.ave_checker.isChecked()):
            self.AverageFilter()
        elif(self.lap_checker.isChecked()):
            self.LaplacianFilter()
        elif(self.sob_checker.isChecked()):
            self.SobelFilter()
        elif(self.sch_checker.isChecked()):
            self.ScharrFilter()
        elif(self.mor_checker.isChecked()):
            self.MorphologicalFilter()

    def GaussianFilter(self):
        if(self.load):
            self.spinBox_KernelSize.setRange(1,100)
            kernel = self.spinBox_KernelSize.value()*2 + 1
            if (self.gau_checker.isChecked()):

                dst = cv2.cvtColor(self.o_image,cv2.COLOR_BGR2RGB)
                dst = cv2.GaussianBlur(dst,(kernel,kernel),cv2.BORDER_REFLECT)
             
                img=QImage(dst,dst.shape[1],dst.shape[0],dst.strides[0],QImage.Format_RGB888)
                self.output_image.setPixmap(QPixmap.fromImage(img.scaled(self.output_image.size())))

                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
            else:
                self.output_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.output_image.size())))
                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
        else:
            print("Please open image :")

    def MedianFilter(self):
        if(self.load):
            self.spinBox_KernelSize.setRange(1,100)
            kernel = self.spinBox_KernelSize.value()*2 + 1
            if (self.med_checker.isChecked()):

                dst = cv2.cvtColor(self.o_image,cv2.COLOR_BGR2RGB)
                dst = cv2.medianBlur(dst,kernel)
             
                img=QImage(dst,dst.shape[1],dst.shape[0],dst.strides[0],QImage.Format_RGB888)
                self.output_image.setPixmap(QPixmap.fromImage(img.scaled(self.output_image.size())))

                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
            else:
                self.output_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.output_image.size())))
                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
        else:
            print("Please open image :")

    def AverageFilter(self):
        if(self.load):
            self.spinBox_KernelSize.setRange(1,100)
            kernel = self.spinBox_KernelSize.value()*2 + 1
            if (self.ave_checker.isChecked()):
                
                dst = cv2.cvtColor(self.o_image,cv2.COLOR_BGR2RGB)
                dst = cv2.blur(dst,(kernel,kernel))
             
                img=QImage(dst,dst.shape[1],dst.shape[0],dst.strides[0],QImage.Format_RGB888)
                self.output_image.setPixmap(QPixmap.fromImage(img.scaled(self.output_image.size())))

                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
            else:
                self.output_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.output_image.size())))
                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
        else:
            print("Please open image :")

    def LaplacianFilter(self):
        if(self.load):
            self.spinBox_KernelSize.setRange(1,50)
            kernel = self.spinBox_KernelSize.value()*2 + 1
            if (self.lap_checker.isChecked()):

                #dst = cv2.GaussianBlur(self.o_image,(kernel,kernel),cv2.BORDER_TRANSPARENT)
                dst = cv2.cvtColor(self.o_image,cv2.COLOR_BGR2GRAY)
                dst = cv2.Laplacian(dst,kernel,cv2.CV_64F)

                img=QImage(dst,dst.shape[1],dst.shape[0],dst.strides[0],QImage.Format_Grayscale8)
                self.output_image.setPixmap(QPixmap.fromImage(img.scaled(self.output_image.size())))

                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
            else:
                self.output_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.output_image.size())))
                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
        else:
            print("Please open image :")

    def SobelFilter(self):
        if(self.load):
            self.spinBox_KernelSize.setRange(1,50)
            kernel = self.spinBox_KernelSize.value()*2 + 1
            if (self.sob_checker.isChecked()):

     
                dst = cv2.Sobel(self.o_image,cv2.CV_64F,1,0,kernel,cv2.BORDER_CONSTANT)

                img=QImage(dst,dst.shape[1],dst.shape[0],dst.strides[0],QImage.Format_Grayscale8)
                self.output_image.setPixmap(QPixmap.fromImage(img.scaled(self.output_image.size())))

                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
            else:
                self.output_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.output_image.size())))
                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
        else:
            print("Please open image :")

    def ScharrFilter(self):
        if(self.load):
            self.spinBox_KernelSize.setRange(1,50)
            kernel = self.spinBox_KernelSize.value()*2 + 1
            if (self.sch_checker.isChecked()):

        
                dst = cv2.Scharr(self.o_image,cv2.CV_64F,1,0,cv2.BORDER_CONSTANT)

                img=QImage(dst,dst.shape[1],dst.shape[0],dst.strides[0],QImage.Format_Grayscale8)
                self.output_image.setPixmap(QPixmap.fromImage(img.scaled(self.output_image.size())))

                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
            else:
                self.output_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.output_image.size())))
                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
        else:
            print("Please open image :")


    def MorphologicalFilter(self):
        if(self.load):
            self.spinBox_KernelSize.setRange(1,50)
            kernel = self.spinBox_KernelSize.value()*2 + 1
            if (self.mor_checker.isChecked()):

                dst = cv2.cvtColor(self.o_image,cv2.COLOR_BGR2RGB)
                M = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel,kernel))
                dst = cv2.morphologyEx(dst,cv2.MORPH_GRADIENT,M)

                img=QImage(dst,dst.shape[1],dst.shape[0],dst.strides[0],QImage.Format_Grayscale8)
                self.output_image.setPixmap(QPixmap.fromImage(img.scaled(self.output_image.size())))

                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
            else:
                self.output_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.output_image.size())))
                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
        else:
            print("Please open image :")

    def RGB2YCbCr(self,selected):
        if(self.load):
            if(selected):
                dst=cv2.cvtColor(self.o_image,cv2.COLOR_RGB2YCrCb)

                img=QImage(dst,dst.shape[1],dst.shape[0],dst.strides[0],QImage.Format_RGB888)
                self.output_image.setPixmap(QPixmap.fromImage(img.scaled(self.output_image.size())))

                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
            else:
                self.output_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.output_image.size())))
                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
        else:
            print("Please open image :")

    def YCbCr2RGB(self,selected):
        if(self.load):
            if(selected):
                dst=cv2.cvtColor(self.o_image,cv2.COLOR_YCrCb2RGB)
                dst=cv2.cvtColor(self.o_image,cv2.COLOR_BGR2RGB)

                img=QImage(dst,dst.shape[1],dst.shape[0],dst.strides[0],QImage.Format_RGB888)
                self.output_image.setPixmap(QPixmap.fromImage(img.scaled(self.output_image.size())))

                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
            else:
                self.output_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.output_image.size())))
                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
        else:
            print("Please open image :")

    
    def GrayScale(self,selected):
        if(self.load):
            if(selected):
                dst=cv2.cvtColor(self.o_image,cv2.COLOR_RGB2GRAY)


                img=QImage(dst,dst.shape[1],dst.shape[0],dst.strides[0],QImage.Format_Grayscale8)
                self.output_image.setPixmap(QPixmap.fromImage(img.scaled(self.output_image.size())))

                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
            else:
                self.output_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.output_image.size())))
                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
        else:
            print("Please open image :")

  
   
    def SaltNoise(self,selected):
        if(self.load):
            if(selected):

                prob = 0.05
                dst = np.zeros(self.o_image.shape,np.uint8)
                thres = 1 - prob
                for i in range(self.o_image.shape[0]):
                    for j in range(self.o_image.shape[1]):
                        rdn = random.random()
                        if rdn < prob:
                            dst[i][j] = 0
                        elif rdn > thres:
                            dst[i][j] = 255
                        else:
                            dst[i][j] = self.o_image[i][j] 
                

                dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
                img=QImage(dst,dst.shape[1],dst.shape[0],dst.strides[0],QImage.Format_RGB888)
                self.output_image.setPixmap(QPixmap.fromImage(img.scaled(self.output_image.size())))

                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
            else:
                self.output_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.output_image.size())))
                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
        else:
            print("Please open image :")

    def GaussianNoise(self,selected):
        if(self.load):
            if(selected):

                img = np.array(self.o_image)
                noise = np.random.randn(img.shape[0], img.shape[1], img.shape[2])
                img = img.astype('int16')
                img_noise = img + noise * 50
                img_noise = np.clip(img_noise, 0, 255)
                dst = img_noise.astype('uint8')
            
                dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
                img=QImage(dst,dst.shape[1],dst.shape[0],dst.strides[0],QImage.Format_RGB888)
                self.output_image.setPixmap(QPixmap.fromImage(img.scaled(self.output_image.size())))

                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
            else:
                self.output_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.output_image.size())))
                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
        else:
            print("Please open image :")

    def PencilSketch(self,selected):
        if(self.load):
            if(selected):

                gray_image = cv2.cvtColor(self.o_image, cv2.COLOR_BGR2GRAY)
                pencilsketch_image  = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)
            
                img=QImage(pencilsketch_image,pencilsketch_image.shape[1],pencilsketch_image.shape[0],pencilsketch_image.strides[0],QImage.Format_Grayscale8)
                self.output_image.setPixmap(QPixmap.fromImage(img.scaled(self.output_image.size())))

                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
            else:
                self.output_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.output_image.size())))
                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
        else:
            print("Please open image :")   

    def CartoonSketch(self,selected):
        if(self.load):
            if(selected):

                cartoonic_image = cv2.stylization(self.o_image, sigma_s=200, sigma_r=0.20)

                cartoonic_image = cv2.cvtColor(cartoonic_image, cv2.COLOR_BGR2RGB)
                img=QImage(cartoonic_image,cartoonic_image.shape[1],cartoonic_image.shape[0],cartoonic_image.strides[0],QImage.Format_RGB888)
                self.output_image.setPixmap(QPixmap.fromImage(img.scaled(self.output_image.size())))

                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
            else:
                self.output_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.output_image.size())))
                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
        else:
            print("Please open image :") 

    def OilpaintingSketch(self,selected):
        if(self.load):
            if(selected):

                Oilpaint_image = cv2.xphoto.oilPainting(self.o_image, 7, 1)

                Oilpaint_image = cv2.cvtColor(Oilpaint_image, cv2.COLOR_BGR2RGB)

                img=QImage(Oilpaint_image,Oilpaint_image.shape[1],Oilpaint_image.shape[0],Oilpaint_image.strides[0],QImage.Format_RGB888)
                self.output_image.setPixmap(QPixmap.fromImage(img.scaled(self.output_image.size())))

                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
            else:
                self.output_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.output_image.size())))
                self.input_image.setPixmap(QPixmap.fromImage(self.image.scaled(self.input_image.size())))
        else:
            print("Please open image :") 
  

