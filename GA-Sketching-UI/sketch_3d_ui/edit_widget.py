from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np
import math

from configs.config import cfg
from sketch_3d_ui.base_opengl_widget import BaseOpenGLWidget as BASEOPENGL
from sketch_3d_ui.view.camera import Camera_Z_UP

from PIL import Image, ImageQt


ALL_FILES = [
    'datasets/shapenet/splits/all_03001627.lst',
    'datasets/shapenet/splits/all_02691156.lst',
    'datasets/animalhead/splits/test.lst'
]

class EditWidget(BASEOPENGL):
    updateSignal = pyqtSignal()
    def __init__(self, parent=None):
        super(EditWidget, self).__init__(parent)
        self.cursor_pen_pix = QPixmap(cfg.ICONS_PATH  + 'cursor_pen.png')
        self.cursor_eraser_pix = QPixmap(cfg.ICONS_PATH  + 'cursor_eraser.png')
        self.mode = 'view'
        
        self.azi = -45.
        self.ele = 15.

        self.camera = Camera_Z_UP(theta=self.azi*math.pi/180., \
                                  phi= (90.-self.ele)*math.pi/180., \
                                  distance=2.0)
        
        self.zoom_factor = 1.0
        self.category_id = 0
        self.dataset = 'shapenet'
       
        self.set_random_shadow()
        

        self.vis_point_cloud = None
        self.cache_dir = None

    def set_random_shadow(self):
        split_file = ALL_FILES[self.category_id]
        shadow_uids = np.loadtxt(split_file,  dtype=str)
        self.shadow_ids = np.random.choice(shadow_uids, 15)
        self.set_shadow()

    def change_category(self, category_id):
        if category_id != self.category_id:
            self.category_id = category_id
            if self.category_id < 2:
                self.dataset = 'shapenet'
            else:
                self.dataset = 'animalhead'
            self.set_random_shadow()

    def set_shadow(self):
        name = '{}_{}_0001.png'.format(int(self.azi)%360, int(self.ele))
        png_fn = os.path.join('datasets/{}/pytorch3d_render/sketch_transparent'.format(self.dataset), self.shadow_ids[0], name)
        img0 = Image.open(png_fn).convert('RGBA').resize((600, 600), Image.Resampling.LANCZOS)

        for shadow_id in self.shadow_ids[1:]:
            png_fn = os.path.join('datasets/{}/pytorch3d_render/sketch_transparent'.format(self.dataset), shadow_id, name)
            img = Image.open(png_fn).convert('RGBA').resize((600, 600), Image.Resampling.LANCZOS)
            img0 = Image.blend(img0, img, alpha=0.25)
        bg=np.zeros((600,600,4),dtype=np.uint8)
        bg = Image.fromarray(bg).convert('RGBA')
        img0 = Image.blend(img0, bg, alpha=0.16)
        img_qt = ImageQt.ImageQt(img0)
        self.canvas_shadow = img_qt

    def set_mode(self, lock, freestyle, straight, eraser, mask):
        BASEOPENGL.canvas_manager.set_tool(lock, freestyle, straight, eraser, mask)
        if lock:
            self.mode = 'edit'
            if freestyle or straight:
                cursor_pix = self.cursor_pen_pix
                size = 30
                cursor_pix = cursor_pix.scaled(size, size)
                cursor = QCursor(cursor_pix, size/2, size/2)
            elif eraser or mask:
                cursor_pix = self.cursor_eraser_pix
                size = BASEOPENGL.canvas_manager.eraser_size
                cursor_pix = cursor_pix.scaled(size, size)
                cursor = QCursor(cursor_pix, size/2, size/2)
            else:
                cursor = Qt.ArrowCursor
        else:
            self.mode = 'view'
            cursor = Qt.ArrowCursor

        self.setCursor(cursor)
        self.changeCursor()
        self.update()

    def set_camera(self, azi, ele):
        self.azi = azi
        self.ele = ele
        self.camera = Camera_Z_UP(theta=self.azi*math.pi/180., \
                                  phi= (90.-self.ele)*math.pi/180., \
                                  distance=2.0)
        if BASEOPENGL.geometry_manager.count == -1:
            self.set_shadow()
            self.draw_img()
            

    def set_sketch_from_png(self, png_fn):
        tmp = png_fn.split('/')[-1].split('_')
        azi, ele = int(tmp[0]), int(tmp[1])
        self.set_camera(azi, ele)
        img_pil = Image.open(png_fn).convert('RGBA')
        img_pil = img_pil.resize((600, 600), Image.NEAREST)# new added!!!
        img_qt = ImageQt.ImageQt(img_pil)
        
        BASEOPENGL.canvas_manager.canvas_sketch = img_qt
        BASEOPENGL.canvas_manager.tmp_canvas_sketch = img_qt
        self.draw_canvas(BASEOPENGL.canvas_manager.canvas_sketch, 0.9)
        self.update()

    def set_vis_pc(self, pc):
        indices = np.random.randint(0, len(pc), 1000)
        sampled_pc = pc[indices]
        # rotate -45 around x axis
        self.vis_point_cloud = sampled_pc.copy()
        self.vis_point_cloud[:, 1] = -sampled_pc[:, 2]
        self.vis_point_cloud[:, 2] = sampled_pc[:,1]
        self.vis_colors = np.ones(self.vis_point_cloud.shape)
        self.vis_colors[:,0] = 0.5 + self.vis_point_cloud[:,0]
        self.vis_colors[:,1] = 0.5 + self.vis_point_cloud[:,1]
        
    ########################################################################
    #    DRAW SCENE                                                        #
    ########################################################################
    def loadScene(self):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glViewport(0, 0, self.width, self.height)
        glClearColor (0.8, 0.8, 0.8, 0.0)
        
        self.set_projection()

    def paintGL(self):
        self.loadScene()
        camera_pos = self.camera.get_cartesian_camera_pos()
        self.set_model_view(camera_pos)
        if self.vis_point_cloud is not None:
            self.draw_point_cloud(self.vis_point_cloud, self.vis_colors)
        self.draw_img()

    # scene  
    def draw_img(self):
        if BASEOPENGL.geometry_manager.count == -1:
            self.draw_canvas(self.canvas_shadow, 0.3)
        self.draw_canvas(BASEOPENGL.canvas_manager.canvas_mask, 0.9)
        self.draw_canvas(BASEOPENGL.canvas_manager.canvas_sketch, 0.9)

        self.update()
        
        
    def draw_canvas(self, canvas, opacity_val=1.0):
        canvasPainter = QPainter(self)
        canvasPainter.setOpacity(opacity_val)
        canvasPainter.drawImage(self.rect(), 
                                canvas,
                                canvas.rect())
        canvasPainter.end()


    ########################################################################
    #    Manager initialization                                            #
    ########################################################################
    def changeCursor(self):
        QApplication.restoreOverrideCursor()

    def reset(self):
        BASEOPENGL.canvas_manager.clean_sketch()
        BASEOPENGL.canvas_manager.clean_mask()
        self.vis_point_cloud = None
        # self.azi = -45.
        # self.ele = 15.
        self.zoom_factor = 1.0
        self.mode = 'view'
        self.setCursor(Qt.ArrowCursor)
        self.changeCursor()
        self.update()

    ########################################################################
    #    CAMERA                                                            #
    ########################################################################
    def update_camera(self, mouse_x, mouse_y):
        d_theta = float(self.lastPos_x - mouse_x) / 300.
        d_phi = float(self.lastPos_y - mouse_y) / 600.
        self.camera.rotate_with_resctrict(d_theta, d_phi)

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
            BASEOPENGL.canvas_manager.reset()

        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()

        self.update()

    def view_mouseReleaseEvent(self, event): 
        if BASEOPENGL.geometry_manager.vertices is not None:
            azi, ele = self.camera.get_azi_ele()
            img_qt = BASEOPENGL.geometry_manager.generate_sketch(azi, ele, self.cache_dir)
            BASEOPENGL.canvas_manager.canvas_sketch = img_qt
            BASEOPENGL.canvas_manager.tmp_canvas_sketch = img_qt
            self.draw_img()

        self.lastPos_x = event.pos().x() 
        self.lastPos_y = event.pos().y()
        self.update()
    
    def view_wheelEvent(self, event):
        if self.vis_point_cloud is not None:
            if event.angleDelta().y()>0:
                zoom_factor = min(self.zoom_factor+0.1, 1.5)
            else:
                zoom_factor = max(self.zoom_factor-0.1, 0.75)

            ratio = zoom_factor/self.zoom_factor
            self.vis_point_cloud = self.vis_point_cloud*ratio
            self.zoom_factor = zoom_factor
            self.updateSignal.emit()

            azi, ele = self.camera.get_azi_ele()
            img_qt = BASEOPENGL.geometry_manager.generate_sketch(azi, ele, self.cache_dir)
            BASEOPENGL.canvas_manager.canvas_sketch = img_qt
            BASEOPENGL.canvas_manager.tmp_canvas_sketch = img_qt
            self.draw_img()

    def edit_mousePressEvent(self, event):
        self.makeCurrent()
        if self.mouse_state == 'LEFT':
            BASEOPENGL.canvas_manager.set_mouse_xy(mouse_x=event.pos().x(), mouse_y=event.pos().y())
            BASEOPENGL.canvas_manager.solve_mouse_event('press')
            
        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        
        self.update()
    
    def edit_mouseMoveEvent(self, event):
        self.makeCurrent()
        
        if self.mouse_state == 'LEFT':
            BASEOPENGL.canvas_manager.set_mouse_xy(mouse_x=event.pos().x(), mouse_y=event.pos().y())
            BASEOPENGL.canvas_manager.solve_mouse_event('move')

        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()

    def edit_mouseReleaseEvent(self, event):
        self.makeCurrent()
        BASEOPENGL.canvas_manager.set_mouse_xy(mouse_x=event.pos().x(), mouse_y=event.pos().y())
        if self.mouse_state == 'LEFT':
            BASEOPENGL.canvas_manager.solve_mouse_event('release')
            
        self.lastPos_x = event.pos().x()
        self.lastPos_y = event.pos().y()
        self.update()
    
