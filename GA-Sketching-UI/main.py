from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import * 
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import os
import sys
import time
import numpy as np
import glob
import trimesh

from configs.config import cfg
from sketch_3d_ui.edit_widget import EditWidget
from sketch_3d_ui.show_widget import ShowWidget
from sketch_3d_ui.base_opengl_widget import BaseOpenGLWidget as BASEOPENGL
from eval.evaluation import eval_mesh

MODES = [
    'freestyle',
    'straight',
    'eraser',
    'mask',
    'lock',
]

TEST_FILES = [
    'datasets/shapenet/splits/test_03001627.lst',
    'datasets/shapenet/splits/test_02691156.lst',
    'datasets/animalhead/splits/test.lst'
]


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.initUI()

        self.freestyleState = False
        self.straightState = False
        self.eraserState = False
        self.maskState = False
        self.lockState = False

        self.sample_id = 'DIY'
        self.sub_cache_dir = None
        self.set_cache_dir()
        
        self.azi_val = -45
        self.ele_val = 15

        self.category_id = 0
        split_file = TEST_FILES[self.category_id]
        self.test_uids = np.loadtxt(split_file,  dtype=str)
        self.dataset = 'shapenet'
        #else:
        #    self.dataset = 'animalhead'
        

    def initUI(self):
        ########## 0 menu bar ##############
        self.createMenuBars()
        ########## 1 tool bar ##############
        self.createInputToolBars()
        ########## 2 central widget ##############
        tool_area = self.createToolWidgets()

        self.edit_widget = EditWidget()
        self.show_widget = ShowWidget()
        canvas_area = QWidget()
        canvas_layout = QHBoxLayout()
        canvas_layout.addWidget(self.edit_widget)
        canvas_layout.addWidget(self.show_widget)
        canvas_area.setLayout(canvas_layout)

        central_widget = QWidget()
        central_layout = QVBoxLayout()

        central_layout.addWidget(tool_area)
        central_layout.addWidget(canvas_area)
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

        ###### action trigger #############
        self.menuBarTriggerActions()
        self.inputToolBarTriggerActions()
        self.toolWidgetTriggerActions()

        self.setFixedHeight(800)

        self.edit_widget.updateSignal.connect(self.zoom)
    
    def zoom(self):
        zoom_factor = self.edit_widget.zoom_factor
        BASEOPENGL.geometry_manager.zoom(zoom_factor)
        BASEOPENGL.canvas_manager.reset()
        self.edit_widget.update()
        self.show_widget.update()

    ##################################################
    #****************** Menu Bar ********************#
    ##################################################
    def createMenuBars(self):
        # for file branch
        fileMenu = QMenu("&File", self)
        self.loadRandomSketchAction = QAction("&Load random sketch", self)
        fileMenu.addAction(self.loadRandomSketchAction)

        # for setting branch
        settingMenu = QMenu("&Setting", self)
        self.resetAction = QAction("&Reset", self)
        settingMenu.addAction(self.resetAction)

        menuBar = self.menuBar()
        menuBar.addMenu(fileMenu)
        menuBar.addMenu(settingMenu)

    def menuBarTriggerActions(self):
        self.loadRandomSketchAction.triggered.connect(lambda: self.loadRandomSketch())
        self.resetAction.triggered.connect(lambda: self.resetWindow())

    def loadRandomSketch(self):
        self.sample_id = np.random.choice(self.test_uids)
        #self.sample_id = '03001627/3a3ddc0c5875bae64316d3320fdfa899'
        print(self.sample_id)
        name = '{}_{}_0001.png'.format(self.azi_val%360, self.ele_val)
        filename = os.path.join('datasets/{}/pytorch3d_render/sketch_transparent'.format(self.dataset), self.sample_id, name)
        self.clearCanvas()
        self.edit_widget.reset()
        self.edit_widget.set_sketch_from_png(filename)
    
    def update_sub_cache_dir(self):
        tmp_sub_cache_dir = os.path.join(self.cache_dir, self.sample_id)
        if self.sub_cache_dir != tmp_sub_cache_dir:
            self.sub_cache_dir = tmp_sub_cache_dir
            os.makedirs(self.sub_cache_dir, exist_ok=True)
            BASEOPENGL.geometry_manager.reset()
            self.edit_widget.cache_dir = self.sub_cache_dir

    def freestyleSketchGeneration(self, use_normal=True):
        self.update_sub_cache_dir()
        BASEOPENGL.geometry_manager.use_normal = use_normal
        azi, ele = self.edit_widget.camera.get_azi_ele()
        img_qt = BASEOPENGL.geometry_manager.generate_sketch(azi, ele, self.sub_cache_dir)
        BASEOPENGL.canvas_manager.canvas_sketch = img_qt
        BASEOPENGL.canvas_manager.tmp_canvas_sketch = img_qt
        self.edit_widget.draw_img()

    def shapeGeneration(self):
        self.update_sub_cache_dir()
        azi, ele = self.edit_widget.camera.get_azi_ele()
        sketch = BASEOPENGL.canvas_manager.qimg2array_sketch()
        edit_mask = BASEOPENGL.canvas_manager.qimg2array_mask()
        if sketch.sum() > 0:
            BASEOPENGL.geometry_manager.generate_reconstructed_mesh(azi, ele, sketch, edit_mask, self.sub_cache_dir)
            self.show_widget.set_output(BASEOPENGL.geometry_manager.vertices, BASEOPENGL.geometry_manager.triangles)
            self.edit_widget.set_vis_pc(BASEOPENGL.geometry_manager.vertices)
            if self.sample_id != 'DIY':
                if self.dataset == 'shapenet':
                    self.shapenet_eval()
                else:
                    self.animalhead_eval()

    def shapenet_eval(self):
        gt_mesh_path = os.path.join(cfg.SHAPENET_GT_PATH, self.sample_id, "watertight_simplified.off")
        gt_mesh = trimesh.load(gt_mesh_path, process=False)
        pred_mesh = BASEOPENGL.geometry_manager.pred_mesh
        eval = eval_mesh(pred_mesh, gt_mesh, -0.5, 0.5)
        print(eval)

    def animalhead_eval(self):
        gt_mesh_path = os.path.join(cfg.ANIMALHEAD_GT_PATH, self.sample_id, "watertight_scaled.off")
        gt_mesh = trimesh.load(gt_mesh_path, process=False)
        pred_mesh = BASEOPENGL.geometry_manager.pred_mesh
        eval = eval_mesh(pred_mesh, gt_mesh, -0.5, 0.5)
        print(eval)

    def resetWindow(self):
        self.freestyleState = False
        self.straightState = False
        self.eraserState = False
        self.lockState = False

        BASEOPENGL.geometry_manager.reset()
        self.edit_widget.reset()
        self.show_widget.reset()
       
        
        self.sample_id = 'DIY'
        self.sub_cache_dir = None
        self.set_cache_dir()
        self.cnt = 0

        self.updateToolBarIcons()


    ##################################################
    #*************** Input Tool Bar *****************#
    ##################################################
    def createInputToolBars(self):
        inputToolBar = QToolBar("inputToolBar", self)
        inputToolBar.setStyleSheet("QToolBar {background: rgb(205, 205, 205) }")
        inputToolBar.setIconSize(QSize(40, 40))

        self.freestyleAction = QAction(QIcon(cfg.ICONS_PATH  + 'freestyle_off.png'), "&Free", self)
        self.straightAction = QAction(QIcon(cfg.ICONS_PATH  + 'straight_off.png'), "&Straight", self)
        self.eraserAction = QAction(QIcon(cfg.ICONS_PATH  + 'eraser_off.png'), "&Eraser", self)
        self.maskAction = QAction(QIcon(cfg.ICONS_PATH  + 'mask_off.png'), "&Eraser", self)
        self.clearAction = QAction(QIcon(cfg.ICONS_PATH  + 'clear.png'), "&Clear", self)
        self.lockAction = QAction(QIcon(cfg.ICONS_PATH  + 'lock_off.png'), "&Lock", self)
        self.generationAction = QAction(QIcon(cfg.ICONS_PATH  + 'generation_hover.png'), "&Generation", self)

        inputToolBar.addAction(self.freestyleAction)
        inputToolBar.addAction(self.straightAction)
        inputToolBar.addAction(self.eraserAction)
        inputToolBar.addAction(self.maskAction)
        inputToolBar.addAction(self.clearAction)
        inputToolBar.addAction(self.lockAction)
        inputToolBar.addAction(self.generationAction)

        self.addToolBar(inputToolBar)

    def inputToolBarTriggerActions(self):
        self.freestyleAction.triggered.connect(lambda: self.updateFreestyleState())
        self.straightAction.triggered.connect(lambda: self.updateStraightState())
        self.eraserAction.triggered.connect(lambda: self.updateEraserState())
        self.maskAction.triggered.connect(lambda: self.updateMaskState())
        self.lockAction.triggered.connect(lambda: self.updateLockState())
        self.clearAction.triggered.connect(lambda: self.clearCanvas())
        self.generationAction.triggered.connect(lambda: self.updateGenerationState())

    def updateGenerationState(self):
        self.shapeGeneration()
        
    def updateFreestyleState(self):
        self.freestyleState = not self.freestyleState
        if self.freestyleState:
            self.lockState = True
            self.straightState = False
            self.eraserState = False
            self.maskState = False
        self.updateToolBarIcons()
        if self.lockState:
            self.edit_widget.set_mode(self.lockState, self.freestyleState, self.straightState, self.eraserState, self.maskState)

    def updateStraightState(self):
        self.straightState = not self.straightState
        if self.straightState:
            self.lockState = True
            self.freestyleState = False
            self.eraserState = False
            self.maskState = False
        self.updateToolBarIcons()
        if self.lockState:
            self.edit_widget.set_mode(self.lockState, self.freestyleState, self.straightState, self.eraserState, self.maskState)

    def updateEraserState(self):
        self.eraserState = not self.eraserState
        if self.eraserState:
            self.lockState = True
            self.freestyleState = False
            self.straightState = False
            self.maskState = False
        self.updateToolBarIcons()
        if self.lockState:
            self.edit_widget.set_mode(self.lockState, self.freestyleState, self.straightState, self.eraserState, self.maskState)

    def updateMaskState(self):
        self.maskState = not self.maskState
        if self.maskState:
            self.lockState = True
            self.freestyleState = False
            self.straightState = False
            self.eraserState = False
        self.updateToolBarIcons()
        if self.lockState:
            self.edit_widget.set_mode(self.lockState, self.freestyleState, self.straightState, self.eraserState, self.maskState)

    def updateLockState(self):
        self.lockState = not self.lockState
        if self.lockState is False:
            self.freestyleState = False
            self.straightState = False
            self.eraserState = False
            self.maskState = False
        self.updateToolBarIcons()
        self.edit_widget.set_mode(self.lockState, self.freestyleState, self.straightState, self.eraserState, self.maskState)
        
    def updateToolBarIcons(self):
        for mode in MODES:
            action = getattr(self, '%sAction' % mode)
            state = getattr(self, '%sState' % mode)

            if state:
                action.setIcon(QIcon(cfg.ICONS_PATH + "%s.png" % mode))
            else:
                action.setIcon(QIcon(cfg.ICONS_PATH + "%s_off.png" % mode))

    def clearCanvas(self):
        BASEOPENGL.canvas_manager.reset()

    ##################################################
    #*************** Tool Widgets ********************#
    ##################################################
    def createToolWidgets(self):
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)

        # eraser controller
        eraser_widget = QWidget()
        eraser_layout = QHBoxLayout()
        self.eraser_size_label = QLabel('Eraser Size: 20')
        self.eraser_size_label.setFont(font)
        self.eraser_size_slider = QSlider(Qt.Horizontal)
        self.eraser_size_slider.setRange(10, 99)
        self.eraser_size_slider.setValue(20)
        self.eraser_size_slider.setTickPosition(QSlider.TicksBelow)
        self.eraser_size_slider.setTickInterval(10)
        eraser_layout.addWidget(self.eraser_size_label)
        eraser_layout.addWidget(self.eraser_size_slider)
        eraser_widget.setLayout(eraser_layout)
        eraser_widget.setMaximumSize(250, 80)

        # pen controller
        pen_widget = QWidget()
        pen_layout = QHBoxLayout()
        self.pen_size_label = QLabel('Pen Size: 5')
        self.pen_size_label.setFont(font)
        self.pen_size_slider = QSlider(Qt.Horizontal)
        self.pen_size_slider.setRange(3, 13)
        self.pen_size_slider.setValue(5)
        self.pen_size_slider.setTickPosition(QSlider.TicksBelow)
        self.pen_size_slider.setTickInterval(1)
        pen_layout.addWidget(self.pen_size_label)
        pen_layout.addWidget(self.pen_size_slider)
        pen_widget.setLayout(pen_layout)
        pen_widget.setMaximumSize(250, 80)

        # azi controller
        azi_widget = QWidget()
        azi_layout = QHBoxLayout()
        self.azi_size_label = QLabel('Azimuth Angle: -45')
        self.azi_size_label.setFont(font)
        self.azi_size_slider = QSlider(Qt.Horizontal)
        self.azi_size_slider.setRange(-12, 12)
        self.azi_size_slider.setValue(-3)
        self.azi_size_slider.setTickPosition(QSlider.TicksBelow)
        self.azi_size_slider.setTickInterval(6)
        azi_layout.addWidget(self.azi_size_label)
        azi_layout.addWidget(self.azi_size_slider)
        azi_widget.setLayout(azi_layout)
        azi_widget.setMaximumSize(250, 80)

        # ele controller
        ele_widget = QWidget()
        ele_layout = QHBoxLayout()
        self.ele_size_label = QLabel('Elevation Angle: 15')
        self.ele_size_label.setFont(font)
        self.ele_size_slider = QSlider(Qt.Horizontal)
        self.ele_size_slider.setRange(-1, 3)
        self.ele_size_slider.setValue(1)
        self.ele_size_slider.setTickPosition(QSlider.TicksBelow)
        self.ele_size_slider.setTickInterval(1)
        ele_layout.addWidget(self.ele_size_label)
        ele_layout.addWidget(self.ele_size_slider)
        ele_widget.setLayout(ele_layout)
        ele_widget.setMaximumSize(250, 80)

        # shadow type
        category_widget = QWidget()
        category_layout = QHBoxLayout()
        self.category_comboBox = QComboBox(self)
        self.category_comboBox.addItem('Chair')
        self.category_comboBox.addItem('Airplane')
        #self.category_comboBox.addItem('Animal Head')
        category_label = QLabel('Category:')
        category_label.setFont(font)
        category_layout.addWidget(category_label)
        category_layout.addWidget(self.category_comboBox)
        category_widget.setLayout(category_layout)
        category_widget.setMaximumSize(250, 80)
        
        # combine all
        tool_widget = QWidget()
        tool_layout = QHBoxLayout()
        tool_layout.addWidget(eraser_widget)
        tool_layout.addWidget(pen_widget)
        tool_layout.addWidget(azi_widget)
        tool_layout.addWidget(ele_widget)
        tool_layout.addWidget(category_widget)
        tool_widget.setLayout(tool_layout)
        tool_widget.setMaximumSize(1500, 80)

        return tool_widget

    def toolWidgetTriggerActions(self):
        self.eraser_size_slider.valueChanged.connect(self.changeEraserSize)
        self.pen_size_slider.valueChanged.connect(self.changePenSize)
        self.azi_size_slider.valueChanged.connect(self.changeAziAngle)
        self.ele_size_slider.valueChanged.connect(self.changeEleAngle)
        self.category_comboBox.currentIndexChanged.connect(self.changeCategory)

    def changeEraserSize(self):
        eraser_size = self.eraser_size_slider.value()
        BASEOPENGL.canvas_manager.eraser_size = eraser_size
        self.eraser_size_label.setText('Eraser Size: ' + str(eraser_size))
        if self.lockState:
            self.edit_widget.set_mode(self.lockState, self.freestyleState, self.straightState, self.eraserState, self.maskState)

    def changePenSize(self):
        pen_size = self.pen_size_slider.value()
        BASEOPENGL.canvas_manager.pen_size = pen_size
        self.pen_size_label.setText('Pen Size: ' + str(pen_size))
        if self.lockState:
            self.edit_widget.set_mode(self.lockState, self.freestyleState, self.straightState, self.eraserState, self.maskState)

    def changeAziAngle(self):
        self.azi_val = self.azi_size_slider.value()*15
        self.azi_size_label.setText('Azimuth Angle: ' + str(self.azi_val))
        self.edit_widget.set_camera(self.azi_val%360, self.ele_val)

    def changeEleAngle(self):
        self.ele_val = self.ele_size_slider.value()*15
        self.ele_size_label.setText('Elevation Angle: ' + str(self.ele_val))
        self.edit_widget.set_camera(self.azi_val%360, self.ele_val)

    def changeCategory(self):
        category_id = self.category_comboBox.currentIndex()
        self.edit_widget.change_category(category_id)
        BASEOPENGL.geometry_manager.change_category(category_id)

        self.category_id = category_id
        split_file = TEST_FILES[self.category_id]
        self.test_uids = np.loadtxt(split_file,  dtype=str)
        
    ################ Other functions #####################
    def set_cache_dir(self):
        time_stamp = str(int(time.time()))
        self.cache_dir = os.path.join('cache', time_stamp)
        if os.path.exists(self.cache_dir):
            os.rmdir(self.cache_dir)
        os.makedirs(self.cache_dir)
        print('Now cache folder is ' + self.cache_dir)
    

if __name__ == '__main__':
    glutInit( sys.argv )
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
