# -*- coding: utf-8 -*-
"""
Computer Vision & Image Processing Gui App
                Qt

Created on Fri Feburary 17, 2023
Time: 10:46:22
@author: Shahid Abbas

"""




from PyQt5.QtWidgets import QApplication
import sys
import Opencv_gui_app as gui

def main():
    app = QApplication(sys.argv)
    ui = gui.Window()
    ui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
