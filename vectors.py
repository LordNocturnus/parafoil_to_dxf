from ezdxf import units
from scipy import interpolate
import numpy as np
import scipy as sp
import pygame as pg
import ezdxf


class Point(object):

    def __init__(self, pos):
        self.pos = pos

    def __add__(self, other):
        return Point(self.pos + other)

    def span(self, other):
        return Vector(other.pos - self.pos, origin=self,
                      length=np.sqrt(np.sum(np.square(other.pos - self.pos))))


class Vector(object):

    def __init__(self, direction, origin=None, length=None):
        if np.sqrt(np.sum(np.square(direction))) == 0.0:
            self.direction = direction
        else:
            self.direction = direction / np.sqrt(np.sum(np.square(direction)))
        self.origin = origin
        if length is not None:
            self.length = length
        else:
            self.length = 1.0

    @property
    def limits(self):
        return np.asarray([[min(self.p0.pos[0], self.p1.pos[0]), max(self.p0.pos[0], self.p1.pos[0])],
                           [min(self.p0.pos[1], self.p1.pos[1]), max(self.p0.pos[1], self.p1.pos[1])],
                           [min(self.p0.pos[2], self.p1.pos[2]), max(self.p0.pos[2], self.p1.pos[2])]])

    @property
    def p0(self):
        return self.get_rel_point(0.0)

    @property
    def p1(self):
        return self.get_rel_point(1.0)

    @property
    def angle_yx(self):
        return np.arctan2(self.direction[0], self.direction[1])

    @property
    def angle_zy(self):
        return np.arctan2(self.direction[1], self.direction[2])

    @property
    def angle_xz(self):
        return np.arctan2(self.direction[2], self.direction[0])

    def scale(self, other):
        self.length = self.length * other

    def get_rel_vec(self, other):
        if self.origin:
            return Vector(self.direction, self.origin, length=other * self.length)
        else:
            return Vector(self.direction, length=other * self.length)

    def get_abs_vec(self, other):
        if self.origin:
            return Vector(self.direction, self.origin, length=other)
        else:
            return Vector(self.direction, length=other)

    def get_rel_point(self, other):
        if not self.origin:
            raise ValueError("Can not get relative point from vector without origin")
        return self.origin + self.direction * self.length * other

    def get_abs_point(self, other):
        if not self.origin:
            raise ValueError("Can not get absolut point from vector without origin")
        return self.origin + self.direction * other

    def dot_prod(self, other):
        return np.sum((self.direction * self.length) * (other.direction * other.length))

    def cross_prod(self, other):
        cross = np.cross(self.direction * self.length, other.direction * other.length)
        length = np.sqrt(np.sum(np.square(cross)))
        if self.origin:
            return Vector(cross, self.origin, length)
        else:
            return Vector(cross, length=length)

    def vec_sub(self, other):
        direction = self.direction * self.length - other.direction * other.length
        length = np.sqrt(np.sum(np.square(direction)))
        if self.origin:
            return Vector(direction, self.origin, length)
        else:
            return Vector(direction, length=length)

    def vec_add(self, other):
        direction = self.direction * self.length + other.direction * other.length
        length = np.sqrt(np.sum(np.square(direction)))
        if self.origin:
            return Vector(direction, self.origin, length)
        else:
            return Vector(direction, length=length)

    def rotate(self, other, angle):
        if not self.origin:
            raise ValueError("Can not rotate around a vector without origin")
        elif type(other) == Point:
            k = self.get_abs_vec(1.0)
            other = self.p0.span(other)
            t0 = other.get_rel_vec(np.cos(angle))
            t1 = k.cross_prod(other).get_rel_vec(np.sin(angle))
            t2 = k.get_abs_vec(k.dot_prod(other) * (1 - np.cos(angle)))
            vec = t0.vec_add(t1).vec_add(t2)
            return vec.p1
        elif type(other) == Vector:
            if not other.origin:
                raise ValueError("Can not rotate a vector without origin")
            p0 = self.rotate(other.p0, angle)
            p1 = self.rotate(other.p1, angle)
            return p0.span(p1)
        elif type(other) == Plane:
            origin = self.rotate(other.origin, angle)
            v1 = self.rotate(other.v1, angle)
            v2 = self.rotate(other.v2, angle)
            return Plane(v1, v2, origin)
        elif type(other) == VectorAirfoil:
            v1 = self.rotate(other.v1, angle)
            v2 = self.rotate(other.v2, angle)
            return VectorAirfoil(other.upper, other.lower, v1, v2, other.length)

    def project(self, other):
        if not self.origin:
            raise ValueError("Can not project onto a vector with out origin")
        if type(other) == Point:
            k = self.get_abs_vec(1.0)
            vec = self.origin.span(other)
            dis = vec.dot_prod(k)
            c1 = self.get_abs_vec(dis)
            c2 = vec.vec_sub(c1)
            c2.origin = c1.p1
            return c1, c2
        elif type(other) == Vector:
            if not other.origin:
                raise ValueError("Can not project a vector without origin onto a vector")
            p0c1, c2 = self.project(other.p0)
            p1c1, c3 = self.project(other.p1)
            c2.origin = p0c1.p1
            c3.origin = p1c1.p1
            c1 = p1c1.vec_sub(p0c1)
            c1.origin = c2.p0
            return c1, c2, c3

    def point_check(self, other):
        c1, c2 = self.project(other)
        if c2.length == 0.0:
            return True
        else:
            return False

    def get_angle(self, other):
        if self.length == 0 or other.length == 0:
            raise ValueError("Can not calculate the angle between vectors with length 0")
        return np.arccos(self.dot_prod(other) / (self.length * other.length))

    def parallel_check(self, other):
        if self.direction[0] == other.direction[0] and self.direction[1] == other.direction[1] and self.direction[2] == \
                other.direction[2]:
            return True
        elif self.direction[0] == other.direction[0] * -1 and self.direction[1] == other.direction[1] * -1 and \
                self.direction[2] == other.direction[2] * -1:
            return True
        else:
            return False

    def draw(self, window, offset, scale, view, color):
        if view == "x":
            pg.draw.line(window, color, (offset[0] + self.p0.pos[2] * scale, offset[1] - self.p0.pos[1] * scale),
                         (offset[0] + self.p1.pos[2] * scale, offset[1] - self.p1.pos[1] * scale))
        elif view == "y":
            pg.draw.line(window, color, (offset[0] + self.p0.pos[0] * scale, offset[1] - self.p0.pos[2] * scale),
                         (offset[0] + self.p1.pos[0] * scale, offset[1] - self.p1.pos[2] * scale))
        elif view == "z":
            pg.draw.line(window, color, (offset[0] + self.p0.pos[0] * scale, offset[1] - self.p0.pos[1] * scale),
                         (offset[0] + self.p1.pos[0] * scale, offset[1] - self.p1.pos[1] * scale))


class Plane(object):

    def __init__(self, v1, v2, origin=None):
        self.v1 = v1
        self.v1.length = 1
        self.v1.origin = origin
        self.v2 = v2
        self.v2.length = 1
        self.v2.origin = origin
        self.origin = origin
        self.n = v1.cross_prod(v2)

    def get_point(self, coords):
        if self.origin is None:
            raise ValueError("Can not get point from Plane without origin")
        self.v2.origin = self.v1.get_abs_point(coords[0])
        p = self.v2.get_abs_point(coords[1])
        self.v2.origin = self.origin
        return p

    def project(self, other):
        if type(other) == Point:
            c0, _ = self.v1.project(other)
            c1, _ = self.v2.project(other)
            c2, _ = self.n.project(other)
            c1.origin = c0.get_rel_point(1.0)
            return c1.get_rel_point(1.0), c2
        elif type(other) == Vector:
            if not other.origin:
                raise ValueError("Can not project a vector without origin onto a plane")
            p0, d0 = self.project(other.p0)
            p1, d1 = self.project(other.p1)
            v1 = p0.span(p1)
            return v1, v1.length/other.length, d0, d1
        elif type(other) == Plane:
            raise NotImplementedError("Not sure if necessary to implement")
        elif type(other) == VectorAirfoil:
            raise NotImplementedError("Not sure if necessary to implement")

    def mirror(self, other):
        if type(other) == Point:
            p, v = self.project(other)
            v.origin = p
            return v.get_rel_point(-1.0)
        elif type(other) == Vector:
            if not other.origin:
                raise ValueError("Can not mirror a vector without origin through a plane")
            p0 = self.mirror(other.p0)
            p1 = self.mirror(other.p1)
            return p0.span(p1)
        elif type(other) == Plane:
            if not other.origin:
                raise ValueError("Can not mirror a plane without origin through a plane")
            v0 = self.mirror(other.v1)
            v1 = self.mirror(other.v2)
            origin = self.mirror(other.origin)
            return Plane(v0, v1, origin)
        elif type(other) == VectorAirfoil:
            if not other.origin:
                raise ValueError("Can not mirror an airfoil without origin through a plane")
            v0 = self.mirror(other.v1)
            v1 = self.mirror(other.v2)
            return VectorAirfoil(other.upper, other.lower, v0, v1, other.length)


class VectorAirfoilFactory(object):

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

    def generate(self, cord_line, thickness_line, offset=0.0, acc=100):
        points2dupper = []
        points2dlower = []
        for c in np.arange(0.0, 1.0 + 1 / acc, 1 / acc):
            points2dupper.append([c * (cord_line.length - offset), self.upper_func(c) * thickness_line.length])
        points2dupper.append([cord_line.length, 0.0])
        for c in np.arange(0.0,  1.0, 1 / acc):
            points2dlower.append([c * (cord_line.length - offset), self.lower_func(c) * thickness_line.length])
        points2dlower.append([cord_line.length - offset, 0.0])
        points2dlower.append([cord_line.length, 0.0])
        return VectorAirfoil(np.asarray(points2dupper), np.asarray(points2dlower), cord_line, thickness_line,
                             cord_line.length)


class VectorAirfoil(Plane):

    def __init__(self, upper, lower, cord_line, thickness_line, length):
        self.length = length
        super().__init__(cord_line, thickness_line, cord_line.p0)
        if type(upper) == np.ndarray:
            self.upper = sp.interpolate.interp1d(upper[:, 0], upper[:, 1])
        else:
            self.upper = upper
        if type(lower) == np.ndarray:
            self.lower = sp.interpolate.interp1d(lower[:, 0], lower[:, 1])
        else:
            self.lower = lower

    @property
    def limits(self, acc=100):
        ret = np.zeros((3, 2), dtype=float)
        cord = np.arange(0, self.length, 1 / acc)
        cord = np.append(cord, [self.length])
        for c in cord:
            ret[0, 0] = min(ret[0, 0], self.get_point((c, self.upper(c))).pos[0],
                            self.get_point((c, self.lower(c))).pos[0])
            ret[0, 1] = max(ret[0, 1], self.get_point((c, self.upper(c))).pos[0],
                            self.get_point((c, self.lower(c))).pos[0])
            ret[1, 0] = min(ret[1, 0], self.get_point((c, self.upper(c))).pos[1],
                            self.get_point((c, self.lower(c))).pos[1])
            ret[1, 1] = max(ret[1, 1], self.get_point((c, self.upper(c))).pos[1],
                            self.get_point((c, self.lower(c))).pos[1])
            ret[2, 0] = min(ret[2, 0], self.get_point((c, self.upper(c))).pos[2],
                            self.get_point((c, self.lower(c))).pos[2])
            ret[2, 1] = max(ret[2, 1], self.get_point((c, self.upper(c))).pos[2],
                            self.get_point((c, self.lower(c))).pos[2])
        return ret

    def lines(self, acc=100):
        ret = []
        cord = np.arange(self.length,  -1 / acc, -1 / acc)
        cord[-1] = 0.0
        for c in range(0, len(cord) - 1):
            ret.append(self.get_point((cord[c], self.upper(cord[c]))).span(self.get_point(
                (cord[c+1], self.upper(cord[c+1])))))
        cord = np.arange(0, self.length, 1 / acc)
        cord = np.append(cord, [self.length])
        for c in range(0, len(cord) - 1):
            ret.append(self.get_point((cord[c], self.lower(cord[c]))).span(self.get_point(
                (cord[c + 1], self.lower(cord[c + 1])))))
        return ret

    def draw(self, window, offset, scale, view, color, acc=100):
        for l in self.lines(acc):
            l.draw(window, offset, scale, view, color)

    def to_dxf(self, allowance_top, allowance_bot, allowance_front, offset, top_cutoff_side, top_cutoff, bottom_cutoff,
               rib_cutoff, acc=100):
        points = []
        rotator = Vector(np.asarray([0.0, 0.0, 1.0]))
        if top_cutoff_side == "t":
            c = np.linspace(self.length - offset - rib_cutoff,  top_cutoff, acc)
            pt = np.transpose(np.asarray([c, self.upper(c)]))
        elif top_cutoff_side == "b":
            c = np.linspace(self.length - offset - rib_cutoff, 0.0, acc, endpoint=False)
            pt = np.transpose(np.asarray([c, self.upper(c)]))
            c = np.linspace(0.0, top_cutoff, acc)
            pt = np.append(pt, np.transpose(np.asarray([c, self.lower(c)])), 0)

        if top_cutoff == 0.0 and bottom_cutoff == 0.0 or (top_cutoff_side == "b" and top_cutoff == bottom_cutoff):
            pf = []
        elif top_cutoff_side == "t":
            c = np.linspace(top_cutoff, 0.0, acc, endpoint=False)
            pf = np.transpose(np.asarray([c, self.upper(c)]))
            c = np.linspace(0.0, bottom_cutoff, acc)
            pf = np.append(pf, np.transpose(np.asarray([c, self.lower(c)])), 0)
        elif top_cutoff_side == "b":
            c = np.linspace(top_cutoff, bottom_cutoff, acc)
            pf = np.transpose(np.asarray([c, self.lower(c)]))

        c = np.linspace(bottom_cutoff, self.length - offset - rib_cutoff, acc)
        pb = np.transpose(np.asarray([c, self.lower(c)]))

        points.append(Point(np.asarray([pt[0][0], pt[0][1], 0.0])))
        prev_angle = np.pi/2
        for p in range(0, len(pt)-1):
            line = Point(np.asarray([pt[p][0], pt[p][1], 0.0])).span(Point(np.asarray([pt[p + 1][0],
                                                                                       pt[p + 1][1], 0.0])))
            rotator.origin = line.p0
            line = rotator.rotate(line, -np.pi / 2)
            p0 = line.get_abs_point(allowance_top)
            if line.angle_yx >= prev_angle:
                p1 = points[-1]
                points[-1] = p0
                points.append(p1)
            else:
                points.append(p0)
            line = Point(np.asarray([pt[p + 1][0], pt[p + 1][1], 0.0])).span(Point(np.asarray([pt[p][0],
                                                                                               pt[p][1], 0.0])))
            rotator.origin = line.p0
            line = rotator.rotate(line, np.pi / 2)
            prev_angle = line.angle_yx
            points.append(line.get_abs_point(allowance_top))

        points.append(Point(np.asarray([pf[0][0], pf[0][1], 0.0])))
        for p in range(0, len(pf)-1):
            line = Point(np.asarray([pf[p][0], pf[p][1], 0.0])).span(Point(np.asarray([pf[p + 1][0],
                                                                                       pf[p + 1][1], 0.0])))
            rotator.origin = line.p0
            line = rotator.rotate(line, -np.pi / 2)
            p0 = line.get_abs_point(allowance_front)
            if line.angle_yx >= prev_angle:
                p1 = points[-1]
                points[-1] = p0
                points.append(p1)
            else:
                points.append(p0)
            line = Point(np.asarray([pf[p + 1][0], pf[p + 1][1], 0.0])).span(Point(np.asarray([pf[p][0],
                                                                                               pf[p][1], 0.0])))
            rotator.origin = line.p0
            line = rotator.rotate(line, np.pi / 2)
            prev_angle = line.angle_yx
            points.append(line.get_abs_point(allowance_front))

        points.append(Point(np.asarray([pb[0][0], pb[0][1], 0.0])))
        for p in range(0, len(pb) - 1):
            line = Point(np.asarray([pb[p][0], pb[p][1], 0.0])).span(Point(np.asarray([pb[p + 1][0],
                                                                                       pb[p + 1][1], 0.0])))
            rotator.origin = line.p0
            line = rotator.rotate(line, -np.pi / 2)
            p0 = line.get_abs_point(allowance_bot)
            if line.angle_yx >= prev_angle:
                p1 = points[-1]
                points[-1] = p0
                points.append(p1)
            else:
                points.append(p0)
            line = Point(np.asarray([pb[p + 1][0], pb[p + 1][1], 0.0])).span(Point(np.asarray([pb[p][0],
                                                                                               pb[p][1], 0.0])))
            rotator.origin = line.p0
            line = rotator.rotate(line, np.pi / 2)
            prev_angle = line.angle_yx
            points.append(line.get_abs_point(allowance_bot))
        points.append(Point(np.asarray([pb[-1][0], pb[-1][1], 0.0])))

        lines = []
        for p in range(0, len(points)):
            lines.append(points[p-1].span(points[p]))
        return Dxf(lines)


class Dxf(object):

    def __init__(self, lines):
        self.lines = lines
        self.limits = np.zeros((3, 2), dtype=float)
        for l in self.lines:
            self.limits[0, 0] = min(self.limits[0, 0], l.p0.pos[0])
            self.limits[0, 1] = max(self.limits[0, 1], l.p0.pos[0])
            self.limits[1, 0] = min(self.limits[1, 0], l.p0.pos[1])
            self.limits[1, 1] = max(self.limits[1, 1], l.p0.pos[1])
            self.limits[2, 0] = min(self.limits[2, 0], l.p0.pos[2])
            self.limits[2, 1] = max(self.limits[2, 1], l.p0.pos[2])

    def draw(self, window, offset, scale, view, color):
        for l in self.lines:
            l.draw(window, offset, scale, view, color)

    def export(self, name):
        doc = ezdxf.new()
        doc.units = ezdxf.units.M
        msp = doc.modelspace()
        for i in range(0, len(self.lines)):
            p0 = np.asarray([self.lines[i].p0.pos[0], self.lines[i].p0.pos[1]])# * 1000
            p1 = np.asarray([self.lines[i].p1.pos[0], self.lines[i].p1.pos[1]])# * 1000
            msp.add_line(p0, p1)
        doc.saveas(f"dxf/{name}.dxf")


if __name__ == "__main__":
    p0 = Point(np.asarray([0.0, 0.0, 0.0]))

    px = Point(np.asarray([2.0, 0.0, 0.0]))
    py = Point(np.asarray([0.0, 2.0, 0.0]))
    pz = Point(np.asarray([0.0, 0.0, 2.0]))

    pxy = Point(np.asarray([2.0, 2.0, 0.0]))
    pyz = Point(np.asarray([0.0, 2.0, 2.0]))
    pzx = Point(np.asarray([2.0, 0.0, 2.0]))

    pxyz = Point(np.asarray([2.0, 2.0, 2.0]))

    vx = Vector(np.asarray([2.0, 0.0, 0.0]), length=2)
    vy = Vector(np.asarray([0.0, 2.0, 0.0]), length=2)
    vz = Vector(np.asarray([0.0, 0.0, 2.0]), length=2)

    vxy = Vector(np.asarray([2.0, 2.0, 0.0]), length=np.sqrt(np.sum(np.square(np.full(2, 2.0, dtype=float)))))
    vyz = Vector(np.asarray([0.0, 2.0, 2.0]), length=np.sqrt(np.sum(np.square(np.full(2, 2.0, dtype=float)))))
    vzx = Vector(np.asarray([2.0, 0.0, 2.0]), length=np.sqrt(np.sum(np.square(np.full(2, 2.0, dtype=float)))))

    vxyz = Vector(np.asarray([2.0, 2.0, 2.0]), length=np.sqrt(np.sum(np.square(np.full(3, 2.0, dtype=float)))))

    vx.origin = p0
    vy.origin = p0
    vxyz.origin = p0

    planexy = Plane(vx, vy, p0)
    res = planexy.mirror(vxyz)
