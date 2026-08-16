import numpy as np
from typing import Optional

class ColorVis:
    def __init__(self, color_table: Optional[bool] = None):
        """ 
        Color the grayscale images.
        
        Args:
        color_table (list or str): 
        list example: [[0,0,0], [0,255,0], [255,0,0]]
        str example: 'xxx/xxx/xxx/color_table.txt',each line in the format 0/0/0#00_background
        """
        if isinstance(color_table, str):
            color_table_list = []
            with open(color_table, 'r') as f:
                for line in f:
                    color_table_list.append(tuple([int(i) for i in line.strip().split('#')[0].split('/')]))
            self.color_table = color_table_list
        else:
            self.color_table = color_table

    def label2colormap(self, img, color_table):
        """
        Convert the grayscale image to a colored image using the color table.

        Args:
            img (numpy array): shape=(H, W) or (C, H, W).
            color_table (list): A list of RGB color tuples.

        Returns:
            numpy array: The colored image with shape (3, H, W).
        """
        m = img.astype(np.uint8)

        if m.ndim == 2:
            r, c = m.shape
            cmap = np.zeros((3, r, c), dtype=np.uint8)
            for i, color in enumerate(color_table):
                mask = (m == i)
                cmap[0][mask] = color[0]
                cmap[1][mask] = color[1]
                cmap[2][mask] = color[2]
        elif m.ndim == 3:
            a, r, c = m.shape
            cmap = np.zeros((3, r, c), dtype=np.uint8)
            for i, color in enumerate(color_table):
                mask = (m == i)
                cmap[0][mask] = color[0]
                cmap[1][mask] = color[1]
                cmap[2][mask] = color[2]

        else:
            raise ValueError("Unsupported image dimensions")

        return cmap

    def label2colormap_default(self, img):
        """
        Convert the grayscale image to a colored image using a bitwise color mapping.

        Args:
            img (numpy array): shape=(H, W) or (C, H, W).

        Returns:
            numpy array: The colored image with shape (3, H, W).
        """
        m = img.astype(np.uint8)

        if m.ndim == 2:
            r, c = m.shape
            cmap = np.zeros((3, r, c), dtype=np.uint8)
            cmap[0, :, :] = (m & 1) << 7 | (m & 8) << 3 | (m & 64) >> 1
            cmap[1, :, :] = (m & 2) << 6 | (m & 16) << 2 | (m & 128) >> 2
            cmap[2, :, :] = (m & 4) << 5 | (m & 32) << 1
        elif m.ndim == 3:
            a, r, c = m.shape
            cmap = np.zeros((3, r, c), dtype=np.uint8)
            cmap[0, :, :] = (m & 1) << 7 | (m & 8) << 3 | (m & 64) >> 1
            cmap[1, :, :] = (m & 2) << 6 | (m & 16) << 2 | (m & 128) >> 2
            cmap[2, :, :] = (m & 4) << 5 | (m & 32) << 1
        else:
            raise ValueError("Unsupported image dimensions")

        return cmap
    
    def run(self, img):
        """ 
            Run function.
        Args:
            img (numpy array): shape=(H, W)/(C, H, W).
        """
        if self.color_table:
            return self.label2colormap(img, self.color_table)
        else:
            return self.label2colormap_default(img)
