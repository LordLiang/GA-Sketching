from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QColor, QPainter, QPen, QImage

import os
import numpy as np
from PIL import Image, ImageQt


class CanvasManager:
    def __init__(self):
        super(CanvasManager, self).__init__()
        self.tool = 0 # ['non-editable', 'freestyle', 'straight', 'eraser', 'mask']

        # canvas
        self.canvas_sketch = QImage(600, 600, QImage.Format_ARGB32)
        self.canvas_sketch.fill(Qt.transparent)
        self.tmp_canvas_sketch = QImage(600, 600, QImage.Format_ARGB32)
        self.tmp_canvas_sketch.fill(Qt.transparent)

        self.canvas_mask = QImage(600, 600, QImage.Format_ARGB32)
        self.canvas_mask.fill(Qt.transparent)
        self.tmp_canvas_mask = QImage(600, 600, QImage.Format_ARGB32)
        self.tmp_canvas_mask.fill(Qt.transparent)

        self.color = [QColor(0, 0, 0), QColor(255, 255, 255)]
 
        self.pen_size = 5
        self.eraser_size = 20
         
    def set_tool(self, lock, freestyle, straight, eraser, mask):
        if not lock:
            self.tool = 0
        else:
            if freestyle:
                self.tool = 1
                self.current_size = self.pen_size
            elif straight:
                self.tool = 2
                self.current_size = self.pen_size
            elif eraser:
                self.tool = 3
                self.current_size = self.eraser_size
            elif mask:
                self.tool = 4
                self.current_size = self.eraser_size

    def set_mouse_xy(self, mouse_x, mouse_y):
        self.mouse_x = mouse_x
        self.mouse_y = mouse_y
    
    def solve_mouse_event(self, event):
        if self.tool > 0:
            if event == 'press':
                self.last_pos = QPoint(self.mouse_x, self.mouse_y)
                self.start_pos = QPoint(self.mouse_x, self.mouse_y)
                self.draw_on_canvas()

            elif event == 'move':
                self.draw_on_canvas()
      
            elif event == 'release':
                self.tmp_canvas_sketch = self.canvas_sketch.copy()
                if self.tool == 4:
                    self.tmp_canvas_mask = self.canvas_mask.copy()
            else:
                pass

    def draw_on_canvas(self):
        current_pos = QPoint(self.mouse_x, self.mouse_y)
        if self.tool == 2:#straight line
            self.canvas_sketch = self.tmp_canvas_sketch.copy()

        painter = QPainter(self.canvas_sketch)
        painter.setPen(QPen(self.color[0],
                            self.current_size,
                            Qt.SolidLine,
                            Qt.RoundCap,
                            Qt.RoundJoin))
        if self.tool == 1:#freestyle
            painter.drawLine(self.last_pos, current_pos)
        elif self.tool == 2:#straight line
            painter.drawLine(self.start_pos, current_pos)
        elif self.tool == 3:#eraser
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.drawLine(self.last_pos, current_pos)
        else:
            pass
        painter.end()

        if self.tool == 4:
            painter = QPainter(self.canvas_mask)
            painter.setPen(QPen(self.color[1],
                            self.current_size,
                            Qt.SolidLine,
                            Qt.RoundCap,
                            Qt.RoundJoin))
            painter.drawLine(self.last_pos, current_pos)
            painter.end()

        self.last_pos = QPoint(self.mouse_x, self.mouse_y)

    
    def qimg2array_sketch(self):
        sketch = ImageQt.fromqimage(self.canvas_sketch)
        sketch = sketch.resize((256, 256), Image.NEAREST).split()[3]
        sketch = np.array(sketch, dtype=np.float32)/255
        return sketch[np.newaxis]
    
    def qimg2array_mask(self):
        mask = ImageQt.fromqimage(self.canvas_mask)
        mask = mask.resize((256, 256), Image.NEAREST).split()[3]
        mask = np.array(mask, dtype=np.float32)/255
        return mask[np.newaxis]
    
    def clean_sketch(self):
        self.canvas_sketch.fill(Qt.transparent)
        self.tmp_canvas_sketch.fill(Qt.transparent)

    def clean_mask(self):
        self.canvas_mask.fill(Qt.transparent)
        self.tmp_canvas_mask.fill(Qt.transparent)

    def reset(self):
        self.clean_sketch()
        self.clean_mask()



        
