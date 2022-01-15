from scipy import interpolate
import numpy as np
import scipy as sp
import pygame as pg
import warnings


class Line(object):

    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1
        self.length = np.sqrt(np.sum(np.square(p0 - p1)))
        self.limits = np.asarray([[min(p0[0], p1[0]), max(p0[0], p1[0])],
                                  [min(p0[1], p1[1]), max(p0[1], p1[1])],
                                  [min(p0[2], p1[2]), max(p0[2], p1[2])]])
        self.relative = self.p1 - self.p0
        self.angle_yx = np.arctan2(self.relative[0], self.relative[1])
        self.angle_zy = np.arctan2(self.relative[1], self.relative[2])
        self.angle_xz = np.arctan2(self.relative[2], self.relative[0])

    def draw(self, window, offset, scale, view, color):
        if view == "x":
            pg.draw.line(window, color, (offset[0] + self.p0[2] * scale, offset[1] - self.p0[1] * scale),
                         (offset[0] + self.p1[2] * scale, offset[1] - self.p1[1] * scale))
        elif view == "y":
            pg.draw.line(window, color, (offset[0] + self.p0[0] * scale, offset[1] - self.p0[2] * scale),
                         (offset[0] + self.p1[0] * scale, offset[1] - self.p1[2] * scale))
        elif view == "z":
            pg.draw.line(window, color, (offset[0] + self.p0[0] * scale, offset[1] - self.p0[1] * scale),
                         (offset[0] + self.p1[0] * scale, offset[1] - self.p1[1] * scale))

    def get_relative(self, dis):
        return self.relative / self.length * dis

    def get_point(self, dis):
        return self.p0 + self.get_relative(dis)


class AirfoilFactory(object):

    def __init__(self, name, acc=100):
        self.points = []
        with open(f"airfoils/{name}") as f:
            lines = f.readlines()
            for l in range(0, len(lines)):
                if not l == 0:
                    p = list(filter(None, lines[l].split(" ")))
                    self.points.append([float(p[0]), float(p[1][:-2])])
                    if self.points[-1][0] == 0.0 and self.points[-1][1] == 0.0:
                        self.flip = l - 1
        self.points = np.asarray(self.points)
        self.upper_func = sp.interpolate.interp1d(self.points[:self.flip+1:, 0], self.points[:self.flip+1, 1])
        self.lower_func = sp.interpolate.interp1d(self.points[self.flip:, 0], self.points[self.flip:, 1])
        points = np.arange(min(self.points[:, 0]), max(self.points[:, 0]), 1 / acc)
        upper = self.upper_func(points)
        lower = self.lower_func(points)
        dis = list(upper - lower)
        self.max_t_point = points[dis.index(max(dis))]
        self.ratio = upper[dis.index(max(dis))] / max(dis)
        self.points[:, 1] = self.points[:, 1] / upper[dis.index(max(dis))]
        self.upper_func = sp.interpolate.interp1d(self.points[:self.flip + 1:, 0], self.points[:self.flip + 1, 1])
        self.lower_func = sp.interpolate.interp1d(self.points[self.flip:, 0], self.points[self.flip:, 1])

    def generate(self, cord_line, thickness_line, acc=100):
        points3d = []
        for c in np.arange(0.0, cord_line.length + 1 / acc, 1 / acc):
            points3d.append(cord_line.get_point(c) +
                            thickness_line.get_relative(self.upper_func(c) * thickness_line.length))
        for c in np.arange(cord_line.length,  0.0, -1 / acc):
            points3d.append(cord_line.get_point(c) +
                            thickness_line.get_relative(self.lower_func(c) * thickness_line.length))
        return Airfoil(np.asarray(points3d))


class Airfoil(object):

    def __init__(self, points):
        self.points = points
        self.lines = []
        for p in range(0, len(points)-1):
            self.lines.append(Line(self.points[p], self.points[p+1]))
        self.lines.append(Line(self.points[-1], self.points[0]))
        self.limits = np.asarray([[min(points[:, 0]), max(points[:, 0])],
                                  [min(points[:, 1]), max(points[:, 1])],
                                  [min(points[:, 2]), max(points[:, 2])]])

    def draw(self, window, offset, scale, view, color):
        for l in self.lines:
            l.draw(window, offset, scale, view, color)


def cos_rule(a=None, b=None, c=None, theta=None):
    if a is not None and b is not None and c is not None:
        return np.arccos(min((a**2 + b**2 - c**2) / (2 * a * b), 1.0))
    elif a is not None and b is not None and theta is not None:
        return np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(theta))
    elif a is not None and c is not None and theta is not None:
        warnings.warn("This is not yet 100% confirmed to be correct")
        return np.sqrt(c**2 - b**2 + b**2 * np.cos(theta)**2) - b * np.cos(theta)
    elif b is not None and c is not None and theta is not None:
        warnings.warn("This is not yet 100% confirmed to be correct")
        return np.sqrt(c**2 - a**2 + a**2 * np.cos(theta)**2) - a * np.cos(theta)
    else:
        raise ValueError(f"Only {len([i for i in [a, b, c, theta] if i])} values where given, but 3 are necessary")