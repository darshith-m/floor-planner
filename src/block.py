'''Contains the Block class, which is used to store details of blocks in the floorplan.'''
class Block:
    '''Class to store details of blocks'''
    def __init__(self, is_soft):
        self.block_name = None
        # True for a soft-macro, None (or False) for a hard-macro
        self.is_soft = is_soft
        # For soft-macros only, otherwise None
        if self.is_soft:
            self.min_aspect_ratio = None    # Minimum aspect ratio
            self.max_aspect_ratio = None    # Maximum aspect ratio
        self.length = None
        self.width = None
        self.area = None
        # To print the coordinates of each block.
        self.x_coordinate = 0.0  # lower left
        self.y_coordinate = 0.0  # lower left
