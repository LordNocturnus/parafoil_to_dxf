import numpy as np
import pygame as pg
import geometry
import vectors


class Parafoil(object):

    def __init__(self, data, acc=100, scale=1.0):
        self.airfoil = vectors.VectorAirfoilFactory(data["airfoil"], acc)
        self.cord = np.asarray(data["cord"]) * scale
        self.thickness = np.asarray(data["thickness"]) * scale
        self.trailing_edge = np.asarray(data["trailing_edge"]) * scale
        self.trailing_edge_horizontal = np.asarray(data["trailing_edge_horizontal"]) * scale
        self.leading_edge = np.asarray(data["leading_edge"]) * scale
        self.top = np.asarray(data["top"]) * scale
        self.limits = np.zeros((3, 2), dtype=float)
        self.airfoils = []
        self.lines = []
        self.trailing_edge_lines = []
        self.leading_edge_lines = []
        self.cord_lines = []
        self.thickness_lines = []
        self.tip_right = []
        self.tip_left = []
        self.construct(acc)

    def construct(self, acc=100):
        # --- Center Cell ---
        self.leading_edge_lines.append(vectors.Vector(np.asarray([1.0, 0.0, 0.0]),
                                                      vectors.Point(np.zeros(3, dtype=float)),
                                                      self.leading_edge[0] / 2))
        dis = np.sqrt(self.cord[0] ** 2 - (self.leading_edge[0] / 2 - self.trailing_edge[0] / 2) ** 2)
        self.trailing_edge_lines.append(vectors.Vector(np.asarray([1.0, 0.0, 0.0]),
                                                       vectors.Point(np.asarray([0.0, 0.0, dis])),
                                                       self.trailing_edge[0] / 2))
        self.cord_lines.append(self.leading_edge_lines[0].p1.span(self.trailing_edge_lines[0].p1))
        t0 = self.cord_lines[0].get_rel_point(self.airfoil.max_t_point)
        relative = np.zeros(3, dtype=float)
        relative[0] = self.top[0] / 2 - t0.pos[0]
        relative[2] = relative[0] * np.tan(self.cord_lines[-1].angle_xz - np.pi / 2)
        relative[1] = np.sqrt((self.airfoil.ratio * self.thickness[0]) ** 2 -
                              relative[0] ** 2 - relative[2] ** 2)
        self.thickness_lines.append(vectors.Vector(relative, t0, np.sqrt(np.sum(np.square(relative)))))
        self.lines.append(vectors.Vector(np.asarray([-1.0, 0.0, 0.0]), self.thickness_lines[0].p1, self.top[0] / 2))
        self.lines.append(vectors.Vector(np.asarray([-1.0, 0.0, 0.0]),
                                         self.thickness_lines[0].get_rel_point(
                                             self.airfoil.lower_func(self.airfoil.max_t_point)),
                                         self.thickness_lines[0].get_rel_point(
                                             self.airfoil.lower_func(self.airfoil.max_t_point)).pos[0]))
        self.airfoils.append(self.airfoil.generate(self.cord_lines[0].get_rel_vec(1.0),
                                                   self.thickness_lines[0].get_rel_vec(1.0), acc))
        # -------------------

        # --- other cells except last ---
        for t in range(1, len(self.leading_edge)):
            relative_trail = np.zeros(3, dtype=float)
            relative_trail[0] = self.trailing_edge[t] * np.cos(self.cord_lines[-1].angle_xz +
                                                               np.arcsin(self.trailing_edge_horizontal[t] /
                                                                         self.trailing_edge[t]) - np.pi)
            relative_trail[2] = self.trailing_edge[t] * np.sin(self.cord_lines[-1].angle_xz +
                                                               np.arcsin(self.trailing_edge_horizontal[t] /
                                                                         self.trailing_edge[t]) - np.pi)
            self.trailing_edge_lines.append(vectors.Vector(relative_trail, self.trailing_edge_lines[-1].p1,
                                                           self.trailing_edge[t]))

            angle = geometry.cos_rule(self.cord_lines[-1].p0.span(self.trailing_edge_lines[-1].p1).length,
                                      self.cord[t - 1], self.trailing_edge[t])
            angle += geometry.cos_rule(self.cord_lines[-1].p0.span(self.trailing_edge_lines[-1].p1).length,
                                       self.leading_edge[t], self.cord[t])
            relative_lead = np.zeros(3, dtype=float)
            relative_lead[0] = self.leading_edge[t] * np.cos(self.cord_lines[-1].angle_xz - angle)
            relative_lead[2] = self.leading_edge[t] * np.sin(self.cord_lines[-1].angle_xz - angle)
            self.leading_edge_lines.append(vectors.Vector(relative_lead, self.leading_edge_lines[-1].p1,
                                                          self.leading_edge[t]))

            self.cord_lines.append(self.leading_edge_lines[-1].p1.span(self.trailing_edge_lines[-1].p1))
            temp_plane = vectors.Plane(self.cord_lines[-2].get_rel_vec(1.0),
                                       self.leading_edge_lines[-1].get_rel_vec(1.0), self.cord_lines[-2].p0)
            self.thickness_lines.append(vectors.Vector(temp_plane.n.direction,
                                                       self.cord_lines[-1].get_rel_point(self.airfoil.max_t_point),
                                                       self.thickness[t] * self.airfoil.ratio))
            self.airfoils.append(self.airfoil.generate(self.cord_lines[-1].get_rel_vec(1.0),
                                                       self.thickness_lines[-1].get_rel_vec(1.0), acc))

            theta0 = self.airfoils[-2].n.get_angle(temp_plane.n)
            self.cord_lines[-1] = self.cord_lines[-2].rotate(self.cord_lines[-1], theta0)
            self.thickness_lines[-1] = self.cord_lines[-2].rotate(self.thickness_lines[-1], theta0)
            self.airfoils[-1] = self.cord_lines[-2].rotate(self.airfoils[-1], theta0)
            temp_plane = self.cord_lines[-2].rotate(temp_plane, theta0)
            self.leading_edge_lines[-1] = self.cord_lines[-2].rotate(self.leading_edge_lines[-1], theta0)
            self.trailing_edge_lines[-1] = self.cord_lines[-2].rotate(self.trailing_edge_lines[-1], theta0)
            theta = temp_plane.n.get_angle(self.airfoils[-1].n)
            self.thickness_lines[-1] = self.cord_lines[-1].rotate(self.thickness_lines[-1], theta)
            self.airfoils[-1] = self.cord_lines[-1].rotate(self.airfoils[-1], theta)
            lim = np.asarray([0.0, -np.pi])
            for _ in range(0, acc):
                l0 = self.cord_lines[-2].rotate(self.cord_lines[-1].rotate(self.thickness_lines[-1], lim[0]),
                                                lim[0]).p1.span(self.thickness_lines[-2].p1).length - self.top[t]
                l1 = self.cord_lines[-2].rotate(self.cord_lines[-1].rotate(self.thickness_lines[-1], np.average(lim)),
                                                np.average(lim)).p1.span(self.thickness_lines[-2].p1).length - \
                     self.top[t]
                l2 = self.cord_lines[-2].rotate(self.cord_lines[-1].rotate(self.thickness_lines[-1], lim[1]),
                                                lim[1]).p1.span(self.thickness_lines[-2].p1).length - self.top[t]
                if not np.sign(l0) == np.sign(l1):
                    lim[1] = np.average(lim)
                elif not np.sign(l1) == np.sign(l2):
                    lim[0] = np.average(lim)
                else:
                    raise ValueError(f"Invalid size of top panel {t}")
            theta = np.average(lim)

            self.airfoils[-1] = self.cord_lines[-2].rotate(self.airfoils[-1], theta)
            self.thickness_lines[-1] = self.cord_lines[-2].rotate(self.thickness_lines[-1], theta)
            self.cord_lines[-1] = self.cord_lines[-2].rotate(self.cord_lines[-1], theta)
            self.leading_edge_lines[-1] = self.cord_lines[-2].rotate(self.leading_edge_lines[-1], theta)
            self.trailing_edge_lines[-1] = self.cord_lines[-2].rotate(self.trailing_edge_lines[-1], theta)
            self.airfoils[-1] = self.cord_lines[-1].rotate(self.airfoils[-1], theta)
            self.thickness_lines[-1] = self.cord_lines[-1].rotate(self.thickness_lines[-1], theta)

            self.lines.append(self.thickness_lines[-1].p1.span(self.thickness_lines[-2].p1))
            self.lines.append(self.thickness_lines[-1].get_rel_point(self.airfoil.lower_func(
                self.airfoil.max_t_point)).span(self.thickness_lines[-2].get_rel_point(self.airfoil.lower_func(
                self.airfoil.max_t_point))))
        # -------------------------------

        # --- tip ---
        coords = (self.cord_lines[-1].length - self.trailing_edge[-1] * np.cos(np.arcsin(
            self.trailing_edge_horizontal[-1] / self.trailing_edge[-1])),
                  self.trailing_edge[-1] * np.sin(np.arcsin(self.trailing_edge_horizontal[-1] /
                                                            self.trailing_edge[-1])))
        self.trailing_edge_lines.append(self.trailing_edge_lines[-1].p1.span(self.airfoils[-1].get_point(coords)))
        dis = self.trailing_edge_horizontal[-1] / np.sin(self.cord_lines[-1].get_angle(self.leading_edge_lines[-1]))
        p = self.cord_lines[-1].get_abs_point(dis)
        v = p.span(self.cord_lines[-1].p0)
        v.length = self.trailing_edge_horizontal[-1]
        vt = self.leading_edge_lines[-1].get_rel_vec(1.0)
        vt = self.cord_lines[-1].rotate(vt, self.thickness_lines[-1].get_angle(vt))
        rotator = self.airfoils[-1].n.get_rel_vec(1.0)
        rotator.origin = p
        lim = self.cord_lines[-1].project(self.trailing_edge_lines[-1].p1)[0].length
        straight = False
        curve = not np.isnan(np.arccos(dis / v.length))
        prev_length = np.inf
        for i in np.arange(0.0, lim, self.cord_lines[-1].length/acc):
            theta = np.arccos((dis - i) / v.length)
            vt.length = self.leading_edge_lines[-1].length + i / np.cos(self.cord_lines[-1].get_angle(
                self.leading_edge_lines[-1]))
            if prev_length > vt.p1.span(p).length and not curve:
                prev_length = vt.p1.span(p).length
                if i == 0.0:
                    self.tip_right.append(self.cord_lines[-1].p0.span(self.airfoils[-1].get_point((i, i * np.tan(
                        self.cord_lines[-1].get_angle(self.leading_edge_lines[-1]))))))
                self.tip_right.append(self.tip_right[-1].p1.span(self.airfoils[-1].get_point((i, i * np.tan(
                        self.cord_lines[-1].get_angle(self.leading_edge_lines[-1]))))))
            else:
                curve = True
                if theta >= np.pi / 2 and not straight:
                    straight = True
                    v = rotator.rotate(v, -np.pi/2)
                if i == 0.0:
                    self.tip_right.append(self.cord_lines[-1].p0.span(rotator.rotate(v, -theta).p1))
                elif straight:
                    v.origin = self.cord_lines[-1].get_abs_point(i)
                    self.tip_right.append(self.tip_right[-1].p1.span(v.p1))
                else:
                    self.tip_right.append(self.tip_right[-1].p1.span(rotator.rotate(v, -theta).p1))
        self.tip_right.append(self.tip_right[-1].p1.span(self.trailing_edge_lines[-1].p1))
        while self.tip_right[0].length <= 10**(-30):
            self.tip_right.pop(0)
        self.trailing_edge_lines[-1] = self.cord_lines[-1].rotate(self.trailing_edge_lines[-1], -np.pi / 2)
        for l in range(0, len(self.tip_right)):
            self.tip_right[l] = self.cord_lines[-1].rotate(self.tip_right[l], -np.pi / 2)
        # -----------"""
        self.mirror()
        self.update_limits()

    def mirror(self):
        mirror_plane = vectors.Plane(vectors.Vector(np.asarray([0.0, 1.0, 0.0])),
                                     vectors.Vector(np.asarray([0.0, 0.0, 1.0])),
                                     vectors.Point(np.zeros(3)))
        trailing_edge_lines = []
        for l in range(1, len(self.trailing_edge_lines)):
            trailing_edge_lines.append(mirror_plane.mirror(self.trailing_edge_lines[-l]))
        self.trailing_edge_lines[0].origin = self.trailing_edge_lines[0].p1
        self.trailing_edge_lines[0].scale(-2)
        self.trailing_edge_lines = trailing_edge_lines + self.trailing_edge_lines

        leading_edge_lines = []
        for l in range(1, len(self.leading_edge_lines)):
            leading_edge_lines.append(mirror_plane.mirror(self.leading_edge_lines[-l]))
        self.leading_edge_lines[0].origin = self.leading_edge_lines[0].p1
        self.leading_edge_lines[0].scale(-2)
        self.leading_edge_lines = leading_edge_lines + self.leading_edge_lines

        cord_lines = []
        for l in range(1, len(self.cord_lines) + 1):
            cord_lines.append(mirror_plane.mirror(self.cord_lines[-l]))
        self.cord_lines = cord_lines + self.cord_lines

        thickness_lines = []
        for l in range(1, len(self.thickness_lines) + 1):
            thickness_lines.append(mirror_plane.mirror(self.thickness_lines[-l]))
        self.thickness_lines = thickness_lines + self.thickness_lines

        lines = []
        for l in range(1, len(self.lines) + 1):
            lines.append(mirror_plane.mirror(self.lines[-l]))
        self.lines = lines + self.lines

        for l in range(0, len(self.tip_right)):
            self.tip_left.append(mirror_plane.mirror(self.tip_right[l]))

        airfoils = []
        for l in range(1, len(self.airfoils) + 1):
            airfoils.append(mirror_plane.mirror(self.airfoils[-l]))
        self.airfoils = airfoils + self.airfoils

    def update_limits(self):
        for a in self.airfoils:
            for d in range(0, len(self.limits)):
                self.limits[d][0] = min(self.limits[d][0], a.limits[d][0])
                self.limits[d][1] = max(self.limits[d][1], a.limits[d][1])

        for l in self.lines:
            for d in range(0, len(self.limits)):
                self.limits[d][0] = min(self.limits[d][0], l.limits[d][0])
                self.limits[d][1] = max(self.limits[d][1], l.limits[d][1])

        for l in self.tip_left:
            for d in range(0, len(self.limits)):
                self.limits[d][0] = min(self.limits[d][0], l.limits[d][0])
                self.limits[d][1] = max(self.limits[d][1], l.limits[d][1])

        for l in self.tip_right:
            for d in range(0, len(self.limits)):
                self.limits[d][0] = min(self.limits[d][0], l.limits[d][0])
                self.limits[d][1] = max(self.limits[d][1], l.limits[d][1])

        for l in self.cord_lines:
            for d in range(0, len(self.limits)):
                self.limits[d][0] = min(self.limits[d][0], l.limits[d][0])
                self.limits[d][1] = max(self.limits[d][1], l.limits[d][1])

        for l in self.trailing_edge_lines:
            for d in range(0, len(self.limits)):
                self.limits[d][0] = min(self.limits[d][0], l.limits[d][0])
                self.limits[d][1] = max(self.limits[d][1], l.limits[d][1])

        for l in self.leading_edge_lines:
            for d in range(0, len(self.limits)):
                self.limits[d][0] = min(self.limits[d][0], l.limits[d][0])
                self.limits[d][1] = max(self.limits[d][1], l.limits[d][1])

    def draw(self, window, offset, scale, view, c1, c2, c3, c4, inner=True):
        if inner:
            for l in self.cord_lines:
                l.draw(window, offset, scale, view, c3)
            for l in self.thickness_lines:
                l.draw(window, offset, scale, view, c3)
        for l in self.leading_edge_lines:
            l.draw(window, offset, scale, view, c2)
        for l in self.trailing_edge_lines:
            l.draw(window, offset, scale, view, c2)
        for l in self.lines:
            l.draw(window, offset, scale, view, c2)
        for l in self.tip_left:
            l.draw(window, offset, scale, view, c2)
        for l in self.tip_right:
            l.draw(window, offset, scale, view, c2)
        for a in self.airfoils:
            a.draw(window, offset, scale, view, c1)

    def cell_to_dxf(self, id, side, acc, allowance_sides, allowance_front, allowance_back, debug=False):
        if id == 0:
            return self.tip_to_dxf("left", side, acc, allowance_sides, allowance_back, debug)
        elif id == len(self.trailing_edge_lines) - 1:
            return self.tip_to_dxf("right", side, acc, allowance_sides, allowance_back, debug)
        rotator = vectors.Vector(np.asarray([0.0, 0.0, 1.0]))
        points_right = [vectors.Point(np.zeros(3, dtype=float)),
                        vectors.Point(np.asarray([allowance_front, 0.0, 0.0]))]
        points_left = [vectors.Point(np.asarray([0.0, self.leading_edge_lines[id - 1].length, 0.0])),
                       vectors.Point(np.asarray([allowance_front, self.leading_edge_lines[id - 1].length, 0.0]))]
        prev_angle_right = np.pi
        prev_angle_left = -np.pi
        if side == "top":
            lines_left = self.airfoils[id - 1].lines[:acc]
            lines_right = self.airfoils[id].lines[:acc]
        elif side == "bot":
            lines_left = self.airfoils[id - 1].lines[acc:]
            lines_left.reverse()
            lines_right = self.airfoils[id].lines[acc:]
            lines_right.reverse()
        for l in range(0, len(lines_left)):  # TODO: make with curved center
            temp_straight = points_right[-1].span(points_left[-1])
            if side == "top":
                straight_3d = lines_left[l].p0.span(lines_right[l].p0).length
            elif side == "bot":
                straight_3d = lines_left[l].p1.span(lines_right[l].p1).length
            straight_3d = np.average(np.asarray([temp_straight.length, straight_3d]))
            rotator.origin = temp_straight.p0
            if side == "top":
                p0 = rotator.rotate(temp_straight, -geometry.cos_rule(lines_right[l].length, straight_3d,
                                                                      lines_right[l].p1.span(
                                                                          lines_left[l].p0).length)
                                    ).get_abs_point(lines_right[l].length)
            elif side == "bot":
                p0 = rotator.rotate(temp_straight, -geometry.cos_rule(lines_right[l].length, straight_3d,
                                                                      lines_right[l].p0.span(
                                                                          lines_left[l].p1).length)
                                    ).get_abs_point(lines_right[l].length)
            rotator.origin = points_right[-1]
            v = points_right[-1].span(p0)
            p1 = rotator.rotate(v, -np.pi / 2).get_abs_point(allowance_sides)
            if v.angle_yx > prev_angle_right:
                points_right.append(points_right[-2])
                points_right[-3] = p1
            else:
                points_right.append(p1)
            prev_angle_right = v.angle_yx
            rotator.origin = p0
            points_right.append(rotator.rotate(p0.span(points_right[-2]), np.pi / 2).get_abs_point(allowance_sides))
            points_right.append(p0)

            temp_straight = points_left[-1].span(points_right[-4])
            rotator.origin = temp_straight.p0
            if side == "top":
                p0 = rotator.rotate(temp_straight, geometry.cos_rule(lines_left[l].length, straight_3d,
                                                                     lines_left[l].p1.span(
                                                                         lines_right[l].p0).length)
                                    ).get_abs_point(lines_left[l].length)
            elif side == "bot":
                p0 = rotator.rotate(temp_straight, geometry.cos_rule(lines_left[l].length, straight_3d,
                                                                     lines_left[l].p0.span(
                                                                         lines_right[l].p1).length)
                                    ).get_abs_point(lines_left[l].length)
            rotator.origin = points_left[-1]
            v = points_left[-1].span(p0)
            p1 = rotator.rotate(v, np.pi / 2).get_abs_point(allowance_sides)
            if v.angle_yx < prev_angle_left:
                points_left.append(points_left[-2])
                points_left[-3] = p1
            else:
                points_left.append(p1)
            prev_angle_left = v.angle_yx
            rotator.origin = p0
            points_left.append(rotator.rotate(p0.span(points_left[1]), -np.pi / 2).get_abs_point(allowance_sides))
            points_left.append(p0)
            if not debug and not l == 0:
                points_right.pop(-4)
                points_left.pop(-4)
        temp_straight = points_right[-1].span(points_left[-1])
        rotator.origin = temp_straight.p0
        points_right.append(rotator.rotate(temp_straight, -np.pi / 2).get_abs_point(allowance_back))
        temp_straight = points_left[-1].span(points_right[-2])
        rotator.origin = temp_straight.p0
        points_left.append(rotator.rotate(temp_straight, np.pi / 2).get_abs_point(allowance_back))
        points_left.reverse()

        points = points_right + points_left
        lines = []
        for p in range(0, len(points)):
            lines.append(points[p - 1].span(points[p]))
        return vectors.Dxf(lines)

    def tip_to_dxf(self, id, side, acc, allowance_sides, allowance_back, debug=False):
        rotator = vectors.Vector(np.asarray([0.0, 0.0, 1.0]))
        if id == "left":
            lines_left = self.tip_left
            prev_angle_left = -np.pi
            if side == "top":
                lines_right = self.airfoils[0].lines[:acc]
            elif side == "bot":
                lines_right = self.airfoils[0].lines[acc:]
                lines_right.reverse()
        elif id == "right":
            lines_left = self.tip_right
            prev_angle_left = -np.pi
            if side == "top":
                lines_right = self.airfoils[-1].lines[:acc]
            elif side == "bot":
                lines_right = self.airfoils[-1].lines[acc:]
                lines_right.reverse()
        prev_angle_right = lines_left[0].get_angle(lines_right[0])
        points_right = [vectors.Point(np.asarray([allowance_sides * np.cos(prev_angle_right),
                                                  -allowance_sides * np.sin(prev_angle_right),
                                                  0.0])),
                        vectors.Point(np.asarray([allowance_sides * np.cos(prev_angle_right) +
                                                     lines_right[0].length * np.sin(prev_angle_right),
                                                  -allowance_sides * np.sin(prev_angle_right) +
                                                     lines_right[0].length * np.cos(prev_angle_right),
                                                  0.0])),
                        vectors.Point(np.asarray([lines_right[0].length * np.sin(prev_angle_right),
                                                  lines_right[0].length * np.cos(prev_angle_right),
                                                  0.0]))
                        ]
        points_left = [vectors.Point(np.zeros(3, dtype=float)),
                       vectors.Point(np.asarray([-allowance_sides, 0.0, 0.0])),
                       vectors.Point(np.asarray([-allowance_sides, lines_left[0].length, 0.0])),
                       vectors.Point(np.asarray([0.0, lines_left[0].length, 0.0]))]

        for l in range(1, min(len(lines_left), len(lines_right))):
            temp_straight = points_right[-1].span(points_left[-1])
            if side == "top":
                straight_3d = lines_left[l].p0.span(lines_right[l].p0).length
            elif side == "bot":
                straight_3d = lines_left[l].p0.span(lines_right[l].p1).length
            straight_3d = np.average(np.asarray([temp_straight.length, straight_3d]))
            rotator.origin = temp_straight.p0
            if side == "top":
                p0 = rotator.rotate(temp_straight, -geometry.cos_rule(lines_right[l].length, straight_3d,
                                                                      lines_right[l].p1.span(
                                                                          lines_left[l].p0).length)
                                    ).get_abs_point(lines_right[l].length)
            elif side == "bot":
                p0 = rotator.rotate(temp_straight, -geometry.cos_rule(lines_right[l].length, straight_3d,
                                                                      lines_right[l].p0.span(
                                                                          lines_left[l].p0).length)
                                    ).get_abs_point(lines_right[l].length)
            rotator.origin = points_right[-1]
            v = points_right[-1].span(p0)
            p1 = rotator.rotate(v, -np.pi / 2).get_abs_point(allowance_sides)
            if v.angle_yx > prev_angle_right:
                points_right.append(points_right[-2])
                points_right[-3] = p1
            else:
                points_right.append(p1)
            prev_angle_right = v.angle_yx
            rotator.origin = p0
            points_right.append(
                rotator.rotate(p0.span(points_right[-2]), np.pi / 2).get_abs_point(allowance_sides))
            points_right.append(p0)

            temp_straight = points_left[-1].span(points_right[-4])
            rotator.origin = temp_straight.p0
            if side == "top":
                p0 = rotator.rotate(temp_straight, geometry.cos_rule(lines_left[l].length, straight_3d,
                                                                     lines_left[l].p1.span(
                                                                         lines_right[l].p0).length)
                                    ).get_abs_point(lines_left[l].length)
            elif side == "bot":
                p0 = rotator.rotate(temp_straight, geometry.cos_rule(lines_left[l].length, straight_3d,
                                                                     lines_left[l].p1.span(
                                                                         lines_right[l].p1).length)
                                    ).get_abs_point(lines_left[l].length)
            rotator.origin = points_left[-1]
            v = points_left[-1].span(p0)
            p1 = rotator.rotate(v, np.pi / 2).get_abs_point(allowance_sides)
            if v.angle_yx > prev_angle_left:
                points_left.append(points_left[-2])
                points_left[-3] = p1
            else:
                points_left.append(p1)
            prev_angle_left = v.angle_yx
            rotator.origin = p0
            points_left.append(rotator.rotate(p0.span(points_left[-2]),
                                              -np.pi / 2).get_abs_point(allowance_sides))
            points_left.append(p0)
            if not debug and not l == 0:
                points_right.pop(-4)
                points_left.pop(-4)

        if len(lines_left) < len(lines_right):
            for l in range(len(lines_left), len(lines_right)):
                temp_straight = points_right[-1].span(points_left[-1])
                if side == "top":
                    straight_3d = lines_left[-1].p1.span(lines_right[l].p0).length
                elif side == "bot":
                    straight_3d = lines_left[-1].p1.span(lines_right[l].p1).length
                straight_3d = np.average(np.asarray([temp_straight.length, straight_3d]))
                rotator.origin = temp_straight.p0
                if side == "top":
                    p0 = rotator.rotate(temp_straight, -geometry.cos_rule(lines_right[l].length, straight_3d,
                                                                          lines_right[l].p1.span(
                                                                              lines_left[-1].p1).length)
                                        ).get_abs_point(lines_right[l].length)
                elif side == "bot":
                    p0 = rotator.rotate(temp_straight, -geometry.cos_rule(lines_right[l].length, straight_3d,
                                                                          lines_right[l].p0.span(
                                                                              lines_left[-1].p1).length)
                                        ).get_abs_point(lines_right[l].length)
                rotator.origin = points_right[-1]
                v = points_right[-1].span(p0)
                p1 = rotator.rotate(v, -np.pi / 2).get_abs_point(allowance_sides)
                if v.angle_yx > prev_angle_right:
                    points_right.append(points_right[-2])
                    points_right[-3] = p1
                else:
                    points_right.append(p1)
                prev_angle_right = v.angle_yx
                rotator.origin = p0
                points_right.append(
                    rotator.rotate(p0.span(points_right[-2]), np.pi / 2).get_abs_point(allowance_sides))
                points_right.append(p0)
                if not debug:
                    points_right.pop(-4)
        elif len(lines_left) > len(lines_right):
            print("left:", len(lines_left), len(lines_right))

        temp_straight = points_right[-1].span(points_left[-1])
        rotator.origin = temp_straight.p0
        points_right.append(rotator.rotate(temp_straight, -np.pi / 2).get_abs_point(allowance_back))
        temp_straight = points_left[-1].span(points_right[-2])
        rotator.origin = temp_straight.p0
        points_left.append(rotator.rotate(temp_straight, np.pi / 2).get_abs_point(allowance_back))
        points_left.reverse()
        points = points_right + points_left

        lines = []
        for p in range(0, len(points)):
            lines.append(points[p - 1].span(points[p]))
        return vectors.Dxf(lines)

    def export(self, acc, allowance_sides, allowance_front, allowance_back, allowance_top, allowance_bot, name):
        for c in range(0, len(self.trailing_edge_lines)):
            dxf = self.cell_to_dxf(c, "top", acc, allowance_sides, allowance_front, allowance_back)
            dxf.export(f"{name}_top_{c}")
            dxf = self.cell_to_dxf(c, "bot", acc, allowance_sides, allowance_front, allowance_back)
            dxf.export(f"{name}_bottom_{c}")
            if not c == len(self.trailing_edge_lines) - 1:
                dxf = self.airfoils[c].to_dxf(allowance_top, allowance_bot)
                dxf.export(f"{name}_Rib_{c},{c+1}")


