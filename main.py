# -*- coding: utf-8 -*-

#-----A Sample Gui Application-----
#-----Computer Vision and Image Processing-----
#-----Shahid Abbas-----
#-----15th Feb 2023-----




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
