
class ViewPort:
    def __init__(self,
                 camera_pos,
                 screen_width,
                 screen_height,
                 projection_matrix,
                 model_view_matrix):
        
        self.camera_pos = camera_pos
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.projection_matrix = projection_matrix
        self.model_view_matrix = model_view_matrix    

        
        
        