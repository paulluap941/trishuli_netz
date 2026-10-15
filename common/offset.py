"""
Co-registration between november and the other two scenes.

A point p in november coordinates sits at q = p + d(s(p)) in the scene, where
s(p) is the position along the valley axis in km. Coefficients were measured
by phase correlation in the mapping project, closing error 0.4-1.4 m. They
only work with that exact valley axis file.

Used here for one thing only: moving the corridor polygons into each scene's
own system. The images are never warped.
"""
import numpy as np
import shapely
import geopandas as gpd

# a*s^2 + b*s + c, s in km, km 0 is upstream
COEFFS = {
    "mar": {"east": [0.03635, -2.0757, -4.93817], "north": [-0.11046, 7.53444, 16.00583]},
    "feb": {"east": [0.02513, -2.03498, -0.1501], "north": [-0.0459, 4.19412, 13.79615]},
    "nov": {"east": [0.0, 0.0, 0.0], "north": [0.0, 0.0, 0.0]},
}


class Offset:
    def __init__(self, axis_path, scene, crs="EPSG:32645"):
        self.axis = gpd.read_file(axis_path).to_crs(crs).geometry.iloc[0]
        self.k = COEFFS[scene]

    def d(self, x, y):
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        s = shapely.line_locate_point(self.axis, shapely.points(np.c_[x, y])) / 1000.0
        return np.polyval(self.k["east"], s), np.polyval(self.k["north"], s)

    def nov_to_scene(self, geom):
        def f(xy):
            e, n = self.d(xy[:, 0], xy[:, 1])
            return np.c_[xy[:, 0] + e, xy[:, 1] + n]
        return shapely.transform(geom, f)

    def scene_to_nov(self, geom, iterations=4):
        # no closed form for the inverse, just iterate. converges fast,
        # the offset changes slowly along the axis.
        def f(q):
            p = q.copy()
            for _ in range(iterations):
                e, n = self.d(p[:, 0], p[:, 1])
                p = np.c_[q[:, 0] - e, q[:, 1] - n]
            return p
        return shapely.transform(geom, f)
