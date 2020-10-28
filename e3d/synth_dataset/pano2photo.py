import argparse
import logging
import os
import sys
from os.path import join

import cv2
import matplotlib.pyplot as plt
import numpy as np

# import pyexr


def get_args():

    parser = argparse.ArgumentParser(
        description="Convert exr Panorama into perspective photos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p", "--path", dest="path", type=str, default="", help="Path to file"
    )
    parser.add_argument(
        "-s",
        "--show",
        dest="show",
        type=bool,
        default=False,
        help="Show frames during loading",
    )
    parser.add_argument(
        "-size",
        "--img_size",
        dest="img_size",
        type=tuple,
        default=(1024, 960),
        help="Output Image size",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=240,
        help="Number of images to create from panorama",
    )
    return parser.parse_args()


class Pano2Photo:
    """https://github.com/fuenwang/Equirec2Perspec
    """

    def __init__(self, path: str, show: bool = True):
        self.path = path
        self.show = show
        self._img = cv2.imread(path)
        if self.show:
            cv2.imshow("pano", self._img)
            cv2.waitKey(0)
        [self._height, self._width, _] = self._img.shape

    """
    def init_exr(self, path: str, show: bool = False):
        self.path = path
        self.show = show
        file = pyexr.open(self.path)
        np_exr = np.array(file.get("default", precision=pyexr.HALF))
        print("original min: {} max: {}".format(np.min(np_exr), np.max(np_exr)))
        self._img = np.clip((np_exr * 255).astype(np.uint8), 0, 255)
        print("new min: {} max: {}".format(np.min(self._img), np.max(self._img)))
        logging.info(f"Succesfully opened image {path}")
        if self.show:
            cv2.imshow("pano.jpg", self._img)
            cv2.waitKey(0)
        [self._height, self._width, _] = self._img.shape
    """

    def SplitToFolder(self, batch_size: int, img_size: tuple):
        """Splits the panorama into n={batch_size} images of size = {img_size}
        Parameters:
            -batch_size: number of images to be created
            -img_size: tuple (height, width) - same row/column order as np
        """

        self.base = self.path.rstrip(".jpg")

        try:
            os.makedirs(self.base)
        except FileExistsError:
            print("File exists continuing")

        for theta in range(180, -180, -1):
            perp = self.GetPerspective(90, theta, 0, img_size[0], img_size[1])

            save_path = join(self.base, f"{abs(theta - 180)}.jpg")
            print(save_path)
            cv2.imwrite(save_path, perp)

    def GetPerspective(
        self, FOV: int, THETA: int, PHI: int, height: int, width: int, RADIUS: int = 128
    ):
        """Transforms Panorama to perspective photo
        Parameters:
            -FOV: [0, 360] = Field of View of view to be captured by the image
            -THETA: [-180, 180] = longitude (z-axis) angle (positive/negative = right/left)
            -PHI: [-180, 180] = latitude (y-axis) angle (positive/negative = up/down)
            -height: int =  height dim of the output perspective image
            -width: int = width dim of the output perspective image
            -RADIUS: int =
        """
        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        wFOV = FOV
        hFOV = float(height) / width * wFOV

        c_x = (width - 1) / 2.0
        c_y = (height - 1) / 2.0

        wangle = (180 - wFOV) / 2.0
        w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
        w_interval = w_len / (width - 1)

        hangle = (180 - hFOV) / 2.0
        h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
        h_interval = h_len / (height - 1)
        x_map = np.zeros([height, width], np.float32) + RADIUS
        y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
        z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
        D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
        xyz = np.zeros([height, width, 3], np.float)
        xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
        xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
        xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2] / RADIUS)
        lon = np.zeros([height * width], np.float)
        theta = np.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(np.bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)

        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy
        # for x in range(width):
        #    for y in range(height):
        #        cv2.circle(self._img, (int(lon[y, x]), int(lat[y, x])), 1, (0, 255, 0))
        # return self._img

        persp = cv2.remap(
            self._img,
            lon.astype(np.float32),
            lat.astype(np.float32),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP,
        )

        return persp


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = get_args()

    for p in os.listdir(args.path):
        if os.path.isdir(join(args.path, p)):
            continue
        try:
            pano2photo = Pano2Photo(join(args.path, p), show=args.show)
            pano2photo.SplitToFolder(args.batch_size, args.img_size)
        except KeyboardInterrupt:
            logging.error("Terminating")
