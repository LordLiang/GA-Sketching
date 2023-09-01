import math
import numpy as np

class Camera_Z_UP:
    def __init__(self, theta, phi, distance):
        self.theta = theta
        self.phi = phi
        self.distance = distance

    def rotate(self, d_theta, d_phi):
        self.theta += d_theta
        self.phi += d_phi

    def rotate_with_resctrict(self, d_theta, d_phi):
        self.theta += d_theta
        self.phi += d_phi
        self.phi = np.clip(self.phi, math.pi/4, math.pi*7/12)

    def zoom(self, d_distance):
        self.distance += d_distance
    
    def get_azi_ele(self):
        azi = self.theta*180./math.pi
        ele = 90. - self.phi*180./math.pi
        return azi%360, ele
    
    def get_cartesian_camera_pos(self):
        camera_pos = [
                      self.distance*math.cos(self.theta)*math.sin(self.phi), \
                      self.distance*math.sin(self.theta)*math.sin(self.phi), \
                      self.distance*math.cos(self.phi)
                     ]

        return camera_pos

        
