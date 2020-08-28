import os
import sys
sys.path.insert(1, (os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))))

from dataclasses import dataclass, astuple, asdict
from abc import ABC, abstractmethod

@dataclass
class Point(ABC):

    @property
    def _tuple(self) -> tuple:
        return astuple(self)

    @property
    def _dict(self) -> dict:
        return asdict(self)

@dataclass
class Point2D(Point):

    x: float = 0.0
    y: float = 0.0

    def __post_init__(self):
        if not isinstance(self.x, float) or\
                not isinstance(self.y, float):
            raise TypeError("Point2D: Point Coordinates should be float")

@dataclass
class Point3D(Point):

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __post_init__(self):
        if not isinstance(self.x, float) or\
                not isinstance(self.y, float) or\
                not isinstance(self.z, float):
            raise TypeError("Point2D: Point Coordinates should be float")

