import os
import sys
sys.path.insert(1, (os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))))

print(sys.path)

from dataclasses import dataclass, field, asdict, astuple
import pandas as pd
import math
import numpy as np
from typing import List
from itertools import product
from abc import ABC, abstractmethod

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from utils.points import Point2D, Point3D

two_pi = 2*math.pi

@dataclass
class Shape(ABC):

    @property
    def _points(self):
        return self.points

    @property
    def _tuples(self) -> List[tuple]:
        return [p._tuple for p in self.points]

    @property
    def _dicts(self) -> List[dict]:
        return [p._dict for p in self.points]

    @property
    def _dataframe(self):
        return pd.DataFrame(self._dicts)

    @abstractmethod
    def plot(self):
        pass

@dataclass
class Circle(Shape):

    radius: float = 0.0
    rot_min: float = 0.0
    rot_max: float = two_pi
    num_points: int = 0
    #[Point2D, Point2D, ...]
    points: list = field(default_factory=list)

    def __post_init__(self):
        rotation_range = np.linspace(self.rot_min, self.rot_max, self.num_points)
        for rot in rotation_range:
            point = Point2D(
                x = self.radius * math.sin(rot),
                y = self.radius * math.cos(rot))
            self.points.append(point)

    def plot(self):
        for p in self._tuples:
            plt.scatter(p[0], p[1])
        plt.show()

@dataclass
class Sphere(Shape):
    """
    Creates continuous 3D points on a sphere given angle ranges, radius and number of points
    """
    radius: float = 0.0
    theta_min: float = 0.0
    theta_max: float = two_pi
    num_points_theta: int = 0
    phi_min: float = 0.0
    phi_max: float = math.pi
    num_points_phi: int = 0
    #[Point3D, Point3D, ...]
    points: list = field(default_factory=list)

    def __post_init__(self):
        theta_range = np.linspace(self.theta_min, self.theta_max, self.num_points_theta)
        phi_range = np.linspace(self.phi_min, self.phi_max, self.num_points_phi)
        for theta, phi in product(theta_range, phi_range):
            point = Point3D(
                x = self.radius * math.cos(theta) * math.sin(phi),
                y = self.radius * math.sin(theta) * math.sin(phi),
                z = self.radius * math.cos(phi))
            self.points.append(point)

    def plot(self):
        fig = go.Figure()
        fig = px.scatter_3d(self._dataframe, x="x", y="y", z="z")
        fig.update_layout(
            title_text="Sphere",
        )
        fig.show()

@dataclass
class SphericalSpiral(Shape):
    """
    Generates points on a spherical spiral
    https://mathworld.wolfram.com/SphericalSpiral.html for more information
    """

    a: float = 1.0 #constant in the parametric equation
    t_min: float = 0.0 #t is the parametric variable
    t_max: float = math.pi
    # [Point3D, Point3D, ...]
    num_points: int = 30
    points: list = field(default_factory=list)

    def __post_init__(self):
        #Generate the range for the parametric var t
        t_range = np.linspace(self.t_min, self.t_max, self.num_points)
        for t in t_range:
            denom = math.sqrt(1 + pow(self.a, 2) + pow(t, 2))
            point = Point3D(
                x = float(math.cos(t) / denom),
                y = float(math.sin(t) / denom),
                z = float(-(self.a * t) / denom)
            )
            self.points.append(point)

    def plot(self):
        fig = go.Figure()
        fig = px.scatter_3d(self._dataframe, x="x", y="y", z="z")
        fig.update_layout(
            title_text="Spherical Sphere",
        )
        fig.show()


if __name__ == "__main__":
    Circle(radius=1.5,num_points=50).plot()
    Sphere(
        radius=2,
        num_points_theta = 1,
        num_points_phi = 20,
        phi_max = math.pi).plot()
