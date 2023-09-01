import math
import numpy as np

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from sketch_3d_ui.view.camera import Camera_Z_UP
from sketch_3d_ui.manager.canvas_manager import CanvasManager
from sketch_3d_ui.manager.geometry_manager import GeometryManager

class BaseOpenGLWidget(QOpenGLWidget):

    canvas_manager = CanvasManager()
    geometry_manager = GeometryManager()
    def __init__(self, parent=None):
        super(BaseOpenGLWidget, self).__init__(parent)

        self.width = 600
        self.height = 600
        self.sub_width = 300
        self.sub_height = 300

        self.setFixedWidth(600)
        self.setFixedHeight(600)

        # camera
        self.camera = None
        self.mouse_state = None
        self.mode = 'view'

    ########################################################################
    #    Mode                                                              #
    ########################################################################
    def set_mode(self, mode):
        self.mode = mode

    def init_mode(self, mode):
        pass

    ########################################################################
    #    SET SCREEN                                                        #
    ########################################################################
    def minimumSizeHint(self):
        return QSize(self.width, self.height)

    def sizeHint(self):
        return QSize(self.width, self.height)

    ########################################################################
    #    CAMERA                                                            #
    ########################################################################
    def update_camera(self, mouse_x, mouse_y):
        d_theta = float(self.lastPos_x - mouse_x) / 300.
        d_phi = float(self.lastPos_y - mouse_y) / 600.
        self.camera.rotate(d_theta, d_phi)
    
    def get_model_view_matrix(self):
        # glGetFloatv output the column order vector
        # for matrix multiplication, need to be transposed
        model_view_matrix = np.array(glGetFloatv(GL_MODELVIEW_MATRIX))

        return model_view_matrix.transpose()

    def get_projection_matrix(self):
        # glGetFloatv output the column order vector
        # for matrix multiplication, need to be transposed
        projection_matrix = np.array(glGetFloatv(GL_PROJECTION_MATRIX))

        return projection_matrix.transpose()
    
    # def set_projection(self):
    #     # set up projection matrix
    #     near = 0.5
    #     far = 100.
    #     A = (near + far)
    #     B = near*far
    #     persp = np.array([
    #                        [512., 0., -128., 0.],
    #                        [0., 512., -128., 0.],
    #                        [0., 0., A, B],
    #                        [0., 0., -1., 0.]
    #                     ]).transpose()  

    #     glMatrixMode(GL_PROJECTION)
    #     glLoadIdentity()

    #     glOrtho(0, 256, 0, 256, near, far)
    #     glMultMatrixf(persp)

    def set_projection(self):
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-0.5, 0.5, -0.5, 0.5, -100., 100.)
        glMatrixMode(GL_MODELVIEW)

    
    def set_model_view(self, camera_pos):
        # set up model view matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(camera_pos[0], camera_pos[1], camera_pos[2], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    def set_model_view_by_candidate_plane(self, candidate_center):
        camera_pos = (candidate_center / np.sqrt(np.sum(candidate_center**2)))*2.

        # set up model view matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(camera_pos[0], camera_pos[1], camera_pos[2], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


    ########################################################################
    #    SET OPENGL                                                        #
    ########################################################################
    def initializeGL(self):
        pass
        
    def resizeGL(self, width, height):
        pass

    def paintGL(self):
        pass

    ########################################################################
    #    DRAW SCENE                                                        #
    ########################################################################
    def draw_point_cloud(self, point_cloud, colors, size=0.005):
        for i, (pos, color) in enumerate(zip(point_cloud, colors)):
            self.draw_sphere(pos, color, size=size)

    def draw_sphere(self, pos, color, size):
        glPushMatrix()
        glTranslated(pos[0], pos[1], pos[2])
        glColor3fv(color)
        glutSolidSphere(size, 8, 6)
        glPopMatrix()

    def draw_mesh(self, mesh, zoom_factor=1):
        faces = mesh.face_vertex_indices().reshape(-1)
        vertices = mesh.points() * zoom_factor
        normals = mesh.vertex_normals()

        vertices = vertices[faces,:].reshape(-1)
        normals = normals[faces,:].reshape(-1)

        vertex_data = (ctypes.c_float*len(vertices))(*(vertices))
        vertex_size = normal_size = len(vertices) * 4
        vertex_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_size, vertex_data, GL_STATIC_DRAW)

        normal_data = (ctypes.c_float*len(normals))(*(normals)) 
        normal_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, normal_vbo)
        glBufferData(GL_ARRAY_BUFFER, normal_size, normal_data, GL_STATIC_DRAW)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, normal_vbo)
        glNormalPointer(GL_FLOAT, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glDrawArrays(GL_TRIANGLES, 0, len(vertices) // 3)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    def draw_mesh_with_color(self, mesh):
        faces = mesh.face_vertex_indices().reshape(-1)
        vertices = mesh.points()
        normals = mesh.vertex_normals()
        colors = mesh.vertex_colors()

        vertices = vertices[faces,:].reshape(-1)
        normals = normals[faces,:].reshape(-1)
        colors = colors[faces,:3].reshape(-1)
        vertex_size = len(vertices) * 4

        vertex_data = (ctypes.c_float*len(vertices))(*(vertices))
        vertex_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_size, vertex_data, GL_STATIC_DRAW)

        normal_data = (ctypes.c_float*len(normals))(*(normals)) 
        normal_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, normal_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_size, normal_data, GL_STATIC_DRAW)

        color_data = (ctypes.c_float*len(colors))(*(colors)) 
        color_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, color_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_size, color_data, GL_STATIC_DRAW)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, normal_vbo)
        glNormalPointer(GL_FLOAT, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, color_vbo)
        glColorPointer(3, GL_FLOAT, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glDrawArrays(GL_TRIANGLES, 0, len(vertices) // 3)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)



    
