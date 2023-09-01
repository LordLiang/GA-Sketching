from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import math
import openmesh as om

from sketch_3d_ui.base_opengl_widget import BaseOpenGLWidget as BASEOPENGL
from sketch_3d_ui.view.camera import Camera_Z_UP

class ShowWidget(BASEOPENGL):
    def __init__(self, parent=None):
        super(ShowWidget, self).__init__(parent)

        self.mode = 'view'

        # eye
        self.azi = -45.
        self.ele = 15.

        self.camera = Camera_Z_UP(theta=self.azi*math.pi/180., \
                                  phi= (90. - self.ele)*math.pi/180., \
                                  distance=2.0)

        self.mouse_state = None
        self.mesh = None
        self.zoom_factor = 1.0


    def set_output(self, points, faces):
        # rotate -45 around x axis
        pc = points.copy()
        pc[:, 1] = -points[:, 2]
        pc[:, 2] = points[:,1]

        self.mesh = om.PolyMesh(points=pc, face_vertex_indices=faces)
        self.mesh.request_vertex_normals()
        self.mesh.update_vertex_normals()
        self.update()

    ########################################################################
    #    DRAW SCENE                                                        #
    ########################################################################
    def loadScene(self):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

        glViewport(0, 0, self.width, self.height)
        glClearColor (0.8, 0.8, 0.8, 0.0)
        
        self.set_projection()

    def paintGL(self):
        self.loadScene()
        camera_pos = self.camera.get_cartesian_camera_pos()
        self.set_model_view(camera_pos)
        self.draw_main_scene()

    def draw_main_scene(self):
        if self.mesh:
            self.draw_mesh(self.mesh, self.zoom_factor)
 
    ########################################################################
    #    Mouse Event                                                       #
    ########################################################################
    def mousePressEvent(self, e):
        if e.buttons() == Qt.LeftButton:
            self.mouse_state ='LEFT'
        elif e.buttons() == Qt.RightButton:
            self.mouse_state = 'RIGHT'
        else:
            return

        fn = getattr(self, "%s_mousePressEvent" % self.mode, None)
        if fn:
            return fn(e)
    
    def mouseMoveEvent(self, e):
        fn = getattr(self, "%s_mouseMoveEvent" % self.mode, None)
        if fn:
            return fn(e)

    def mouseReleaseEvent(self, e):
        fn = getattr(self, "%s_mouseReleaseEvent" % self.mode, None)
        if fn:
            return fn(e)
    
    def wheelEvent(self, e):
        fn = getattr(self, "%s_wheelEvent" % self.mode, None)
        if fn:
            return fn(e)

    def view_mousePressEvent(self, event):
        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()

    def view_mouseMoveEvent(self, event):
        if self.mouse_state == 'RIGHT':
            self.makeCurrent()
            self.update_camera(mouse_x=event.pos().x(), mouse_y=event.pos().y())

        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()

    def view_mouseReleaseEvent(self, event): 
        self.lastPos_x = event.pos().x() 
        self.lastPos_y = event.pos().y() 
        self.update()
    
    def view_wheelEvent(self, event):
        if event.angleDelta().y()>0:
            self.zoom_factor = min(self.zoom_factor+0.1, 1.5)
        else:
            self.zoom_factor = max(self.zoom_factor-0.1, 0.5)
        
        self.update()

    def reset(self):
        self.mesh = None
        self.azi = -45.
        self.ele = 15.
        self.zoom_factor = 1.0
        self.update()

    
    
