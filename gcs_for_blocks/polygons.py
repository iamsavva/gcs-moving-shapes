#!/usr/bin/env python3
# pyright: reportMissingImports=false
import typing as T

import numpy as np
import numpy.typing as npt

class CenteredPolygon(object):
    # assumed to be centered around 0

    def __init__(self, vertices:npt.NDArray, t = -1, center = None):
        self.vertices = vertices
        self.n = len(self.vertices)
        self.center = center
        self.t = t
        if self.t == -1:
            assert self.center is not None
        else:
            assert self.center is None
        
class EqPolygon(CenteredPolygon):
    def __init__(self, num_sides:int, radius:float, t, center = None):
        # equalateral polygon
        self.theta = 2 * np.pi / num_sides
        self.r = radius
        # vertices = self.r * np.array([ [ np.cos(self.theta*i), np.sin(self.theta*i) ] for i in range(num_sides)  ])
        # vertices = self.r * np.array([ [ np.cos(np.pi/4+self.theta*i), np.sin(np.pi/4+self.theta*i) ] for i in range(num_sides)  ])
        vertices = self.r * np.array([ [ np.cos(np.pi/10+self.theta*i), np.sin(np.pi/10+self.theta*i) ] for i in range(num_sides)  ])
        super(EqPolygon, self).__init__(vertices, t, center)

class Box(CenteredPolygon):
    def __init__(self, width, height, t, center = None):
        self.w = width
        self.h = height
        vertices = np.array([ [-width/2, -height/2], [-width/2, height/2], [width/2, height/2], [width/2, -height/2] ])
        super(Box, self).__init__(vertices, t, center)
