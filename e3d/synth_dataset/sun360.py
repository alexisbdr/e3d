import os
import shutil
import sys
from os.path import join

import cv2
import imutils
import numpy as np


class Equirectangular:
    """https://github.com/fuenwang/Equirec2Perspec
    """

    def __init__(self, path):
        self.path = path
        self._img = cv2.imread(path, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape
        # cp = self._img.copy()
        # w = self._width
        # self._img[:, :w/8, :] = cp[:, 7*w/8:, :]
        # self._img[:, w/8:, :] = cp[:, :7*w/8, :]

    def SplitToFolder(self, fov=90, theta=0, phi=0):

        self.name = self.path.split("/")[-1]
        self.base = self.path.strip(self.name)
        self.name = self.name.strip(".exr")
        print(f"base: {self.base}, name: {self.name}")

        perp = self.GetPerspective(fov, theta, phi, 560, 560)

        save_path = join(self.base, f"{self.name}.jpg")
        print(save_path)
        cv2.imwrite(save_path, perp)

    def GetPerspective(self, FOV, THETA, PHI, height, width, RADIUS=128):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

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


def stitch_and_split():

    done = {}
    backgrounds = "./data/background"

    for folder in sorted(os.listdir(backgrounds)):

        path = join(backgrounds, folder)

        if folder == ".ipynb_checkpoints":
            continue
        if folder in done:
            shutil.move(path, f"./data/new_background/{folder}")
            print(f"already did {folder}")
            continue

        images = [
            cv2.resize(cv2.imread(join(path, p)), (128, 128))
            for p in sorted(os.listdir(path))[0:24]
        ]

        stitcher = cv2.Stitcher_create(mode=1)
        # stitcher.setRegistrationResol(1); # 0.6
        # stitcher.setSeamEstimationResol(-1); #0.1
        # stitcher.setCompositingResol(-1);   #1
        # stitcher.setPanoConfidenceThresh(-1);  #1
        # stitcher.setWaveCorrection(True);
        # stitcher.setWaveCorrectKind(detail::WAVE_CORRECT_HORIZ);

        try:
            (status, orig_stitched) = stitcher.stitch(images)
        except:
            done[folder] = 0

        if status != 0:
            shutil.move(path, f"./data/new_background/{folder}")
            print(f"{path} did not work")
            done[path] = 0
            continue
        if orig_stitched.shape[0] < 15:
            shutil.move(path, f"./data/new_background/{folder}")
            print(f"{path} too small, {orig_stitched.shape}")
            done[path] = 0
            continue

        stitched = cv2.copyMakeBorder(
            orig_stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0)
        )

        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(thresh.shape, dtype="uint8")
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        minRect = mask.copy()
        sub = mask.copy()

        while cv2.countNonZero(sub) > 0:
            minRect = cv2.erode(minRect, None)
            sub = cv2.subtract(minRect, thresh)

        cnts = cv2.findContours(
            minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = imutils.grab_contours(cnts)
        if not cnts:
            done[path] = 0
            continue
        c = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)

        stitched = stitched[y : y + h, x : x + w]
        # print(stitched.shape)

        # os.mkdir("test/")

        if stitched.shape[0] < 20:

            continue

        # stitched = orig_stitched
        if stitched.shape[1] < 128:
            # done[folder] = 0
            continue

        split_size = max(1, int((stitched.shape[1] - 128) / 72))

        print(f"orig {orig_stitched.shape}")
        print(f"croppped {stitched.shape}")
        curr_size = 0
        new_images = []
        while curr_size < stitched.shape[1] - 128:
            split = stitched[:, curr_size : curr_size + 128].copy()
            blur = cv2.GaussianBlur(split, (7, 7), 0)
            split = cv2.addWeighted(split, 1.5, blur, -0.5, 0)
            new_images.append(cv2.resize(split, (128, 128)))
            curr_size += split_size
        for num in range(min(72, len(new_images))):
            img_path = join(path, f"{num}.png")
            os.remove(img_path)
            cv2.imwrite(img_path, new_images[num])
        new_images = []

        # done[folder] = 0
        print(f"completed {folder}")
        shutil.move(path, f"./data/new_background/{folder}")


if __name__ == "__main__":
    stitch_and_split()
