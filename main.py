from ctypes import CDLL
from ctypes import c_size_t
from ctypes import c_void_p
from ctypes import c_char_p
from ctypes import create_string_buffer

import cv2 as cv
import cv2 as cv2
import os
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sultan.api import Sultan
from scipy import signal
# from scipy import misc
import platform
from pathlib import Path, PureWindowsPath
# from PIL import Image, ImageDraw, ImageFont

# System
cwd = str(Path.cwd())


def path(path):
    _platform = platform.system()
    if _platform == "linux" or _platform == "linux2" or 'cygwin' or "darwin":
        path = str(Path(path))
        pass
    elif _platform == "win32" or _platform == "win64":
        path = str(PureWindowsPath(path))
        pass
    else:
        pass
    return path


def debug(func):

    def wrapper(*args, **kwargs):
        print("[DEBUG]: enter {}()".format(func.__name__))
        return func(*args, **kwargs)

    return wrapper  # 返回


@debug
def walk(str):
    shpfiles = []
    for dirpath, subdirs, files in os.walk(cwd):
        for x in files:
            if x.endswith(str):
                shpfiles.append(os.path.join(dirpath, x))
    return shpfiles


@debug
def list_name(sub):
    ListOfPath = walk(sub)
    ListOfFile = list()
    for x in range(len(ListOfPath)):
        ListOfFile.append(ListOfPath[x].replace(cwd, "").lstrip("/").lstrip("\\"))
    # print(ListOfFile)
    return ListOfFile


@debug
def cmd_magick_convert_array(x, y, in_path="./img/le.jpg", out_path="./img/le.bmp"):
    ws = " "
    size_str = str(x) + "x" + str(y)
    main_path = path("./src/main.c")
    main_out_path = path("a.out")
    data_out_path = path(ws + "rgba" + size_str + ".h")
    try:
        # convert_flag = False
        s = Sultan()
        # List *.h
        list_header = list_name("h")
        for items in list_header:
            val = items
            if any(size_str in val for items in list_header):
                print(items)
            # if convert_flag is False:
            # Convert image
            s.convert("-quality 100 -resize" + ws + size_str + "!" + ws + in_path + ws + out_path).run()
            #  Creat *.h
            s.convert(out_path + ws + "-define h:format=rgba -depth 8 -size" + ws + size_str + ws + data_out_path).run()
        # else:
        # pass
        # Creat link file
        s.gcc("-fPIC -shared -o" + ws + main_out_path + ws + main_path).run()
    except Exception as e:
        print(e)
    return CDLL("a.out")


class NpArray:

    def __init__(self, arr):
        """
        Matrix Operator
        """
        self.col = arr.shape[0]
        self.row = arr.shape[1]
        self.model = arr.shape[2]
        self.arr_src = np.empty(self.row * self.col * self.model).astype(np.uint8)
        # self.arr_dest = np.empty(shape=(row, col, model), dtype=np.uint8)
        self.arr_dest = np.reshape(self.arr_src, (self.row, self.col, self.model))

    # input is a RGB numpy array with shape (height,width,3), can be uint,int, float or double, values expected in the range 0..255
    # output is a double YUV numpy array with shape (height,width,3), values in the range 0..255
    def RGB2YUV(self, rgb):
        m = np.array([
            [0.29900, -0.16874, 0.50000],
            [0.58700, -0.33126, -0.41869],
            [0.11400, 0.50000, -0.08131],
        ])
        yuv = np.dot(rgb, m)
        yuv[:, :, 1:] += 128.0
        return yuv

    # input is an YUV numpy array with shape (height,width,3) can be uint,int, float or double,  values expected in the range 0..255
    # output is a double RGB numpy array with shape (height,width,3), values in the range 0..255
    def YUV2RGB(self, selfrr, yuv):
        m = np.array([
            [1.0, 1.0, 1.0],
            [
                -0.000007154783816076815,
                -0.3441331386566162,
                1.7720025777816772,
            ],
            [
                1.4019975662231445,
                -0.7141380310058594,
                0.00001542569043522235,
            ],
        ])
        rgb = np.dot(yuv, m)
        rgb[:, :, 0] -= 179.45477266423404
        rgb[:, :, 1] += 135.45870971679688
        rgb[:, :, 2] -= 226.8183044444304
        return rgb

    def getterS(self):
        return self.arr_dest

    def getterD(self):
        return self.arr_src

    def InfoD(self):
        print(self.arr_dest)

    def InfoS(self):
        print(self.arr_src)

    def Add(self):
        pass
        self.arr_dest += self.arr_src
        return self.arr_dest

    def Sub(self):
        self.arr_dest -= self.arr_src
        return self.arr_dest

    def Dot(self):
        self.arr_dest = np.dot(self.arr_dest, self.arr_src)
        return self.arr_dest

    def ColSwap(self, a, b):
        # TODO if a  <= col max
        if 1:
            self.arr_dest[:, [a, b]] = self.arr_dest[:, [a, b]]
        else:
            pass
        return self.arr_dest

    def RowSwap(self):
        # TODO
        pass
        return self.arr_dest

    def Alpha(self, alpha):
        self.arr_dest[0:self.row, 0:self.col, 3] = alpha


class FileHandler:

    def __init__(self, x, y, model, name):
        self.model = model
        self.x = x
        self.y = y
        self.col = y
        self.row = x
        self.width = x
        self.height = y
        # VY1UY0
        self.size = self.x * self.y * model
        self.name = name
        self.clib = cmd_magick_convert_array(self.x, self.y)
        self.count = int(self.size / 4)
        if model == 2:
            self.arr_dest = np.empty(shape=(self.col, self.row, 2), dtype=np.uint8)
        elif model == 4:
            self.arr_dest = np.empty(shape=(self.col, self.row, self.model), dtype=np.uint8)
        else:
            print("fail FileHandler")

    def print(self):
        print(self.clib)

    def getter(self):
        return self.clib


class ReadFile(FileHandler):
    """
    API void *np_memcpy_bin(uint8_t *arr_dest, size_t size, char *name)
    """

    def __init__(self, col, row, model, name=""):
        super().__init__(col, row, model, name)
        np.set_printoptions(formatter={"int": hex})

    def np_memcpy_bin(self):
        self.clib.np_memcpy_bin.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS", ndim=3),
            c_size_t,
            c_char_p,
        ]
        # self.clib.np_memcpy_bin.restype = c_void_p
        self.clib.np_memcpy_bin(self.arr_dest, self.size, self.name)
        return self

    def np_memcpy_fixed_rgba(self):
        self.clib.np_memcpy_fixed_rgba.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS", ndim=3),
            c_size_t,
        ]
        self.clib.np_memcpy_fixed_rgba(self.arr_dest, self.size)
        return self

    def func(self, buf):
        self.clib.func.argtype = c_char_p
        # self.clib.func.restype = c_long
        self.clib.func(buf)
        return self


@debug
def mat_plot(np_arr, mode=0):
    img_org = Image.open("le.bmp").convert("RGBA")
    np_arr_image = np.array(img_org)
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(np_arr_image)
    plt.subplot(212)
    if mode == 0:
        plt.imshow(np_arr)
    else:
        plt.imshow(np_arr, cmap="gray")
    plt.show()


@debug
def pil_plot(np_arr):
    image = Image.fromarray(np_arr)
    print(np_arr)
    image.show()
    image.save("test.png")


def cv_plot(np_arr, code=cv2.COLOR_RGB2RGBA, mode=1):
    if mode == 0:
        np_arr = cv2.imread("le.bmp")
        cv2.imwrite("le.bmp", np_arr)
        np_arr_new = cv2.cvtColor(np_arr, cv2.COLOR_BGR2RGB)
    elif mode == 1:
        np_arr = cv2.cvtColor(np_arr, code)
    else:
        print("fail plot")
    b, g, r = cv2.split(np_arr)
    np_arr_new = cv2.merge([r, g, b])
    # mat_plot(np_arr_new)
    return np_arr_new


def make_lut_u():
    return np.array([[[i, 255 - i, 0] for i in range(256)]], dtype=np.uint8)


def make_lut_v():
    return np.array([[[0, 255 - i, i] for i in range(256)]], dtype=np.uint8)


def yuv_plot(in_name="le.bmp", out_name="out.bmp"):
    """
    #https://stackoverflow.com/questions/43983265/rgb-to-yuv-conversion-and-accessing-y-u-and-v-channels/43988642
    """
    image = cv.imread(in_name)
    cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(image)
    y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
    v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
    lut_u, lut_v = make_lut_u(), make_lut_v()
    u_mapped = cv2.LUT(u, lut_u)
    v_mapped = cv2.LUT(v, lut_v)
    result = np.vstack([image, y, u_mapped, v_mapped])
    cv2.imwrite(out_name, result)
    cv2.imshow(out_name, result)
    cv2.waitKey()


def binary_oprator(arr, mode):
    """
    開運算---先腐蝕，後膨脹。去除圖像中小的亮點（CV_MOP_OPEN）；
    閉運算---先膨脹，後腐蝕。去除圖像中小的暗點（CV_MOP_CLOSE）；
    形態學梯度---原圖膨脹圖像 — 原圖腐蝕圖像（CV_MOP_GRADIENT）；
    頂帽---原圖像 — 原圖像開操作。保留小亮點，去除大亮點。（CV_MOP_TOPHAT）；
    黑帽---原圖像閉操作 — 原圖像（CV_MOP_BLACKHAT）；
    """
    # 核的大小和形狀
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    ret1 = cv.morphologyEx(arr, cv.MORPH_OPEN, kernel, iterations=1)
    ret2 = cv.morphologyEx(arr, cv.MORPH_CLOSE, kernel, iterations=1)
    ret3 = cv.morphologyEx(arr, cv.MORPH_CLOSE, kernel, iterations=1)
    ret4 = cv.morphologyEx(arr, cv.MORPH_CLOSE, kernel, iterations=1)
    ret5 = cv.morphologyEx(arr, cv.MORPH_CLOSE, kernel, iterations=1)
    ret_hstack = np.hstack(ret1, ret2, ret3, ret4, ret5)
    cv2.imshow("merged_img", ret_hstack)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def binary_normalize(src, dst):
    """
        # normalize and contrst, 直方圖歸一化
    """
    dst = np.zeros_like(src)
    cv2.normalize(src, dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # 公式
    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 計算灰度直方圖
    grayHist = cv2.calcHist([src], [0], None, [256], [0, 256])
    grayHist1 = cv2.calcHist([dst], [0], None, [256], [0, 256])
    # 畫出直方圖
    x_range = range(256)
    plt.plot(x_range, grayHist, "r", linewidth=1.5, c="black")
    plt.plot(x_range, grayHist1, "r", linewidth=1.5, c="b")
    # 設置坐標軸的範圍
    y_maxValue = np.max(grayHist)
    plt.axis([0, 255, 0, y_maxValue])  # 畫圖範圍
    plt.xlabel("gray Level")
    plt.ylabel("number of pixels")
    plt.show()


def bit_operator(img1, img2, mode):
    """
        # https://docs.opencv.org/master/d0/d86/tutorial_py_image_arithmetics.html
        # https://blog.csdn.net/kwame211/article/details/86307946
        # https://blog.csdn.net/qq_28949847/article/details/103095651
        mode = 0, gray
        mode = 1, color
    """
    # Load two images
    # img1 = cv.imread("messi5.jpg")
    # img2 = cv.imread("opencv-logo-white.png")
    # I want to put logo on top-left corner, So I create a ROI
    if mode == 0:
        rows, cols = img2.shape
    else:
        rows, cols, channels = img2.shape
        # Now create a mask of logo and create its inverse mask also
        img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    roi = img1[0:rows, 0:cols]
    ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv.bitwise_and(img2, img2, mask=mask)
    # Put logo in ROI and modify the main image
    dst = cv.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst
    # # store
    # cv2.imencode('.jpg', dst)[1].tofile(r'dd_img.jpg')
    # cv2.imshow('dst', dst)
    # cv2.waitKey(0
    return dst


def binarization():
    """
        https://blog.csdn.net/weixin_43046653/article/details/83277827
    """
    pass


def sobel(img):
    """
        # https://ithelp.ithome.com.tw/articles/10205752
        # https://medium.com/@fromtheast/computer-vision-resources-411ae9bfef51
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # x方向
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobelx_8u = np.uint8(np.absolute(sobelx))

    # y方向
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    abs_sobely_8u = np.uint8(np.absolute(sobely))

    # x、y方向
    sobelxy = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
    abs_sobelxy_8u = np.uint8(np.absolute(sobelxy))

    cv2.imshow("gray", gray)
    cv2.imshow("sobelx", abs_sobelx_8u)
    cv2.imshow("sobely", abs_sobely_8u)
    cv2.imshow("sobelxy", abs_sobelxy_8u)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Gassian(arr, mode):
    """
        mode=0, self-build 
        mode=1, opencv
    """
    # https://medium.com/@bob800530/python-gaussian-filter-概念與實作-676aac52ea17
    if mode == 0:
        # 3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2 + y**2))
        # Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        # plt.imshow(gaussian_kernel, cmap=plt.get_cmap('jet'), interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        grad = signal.convolve2d(arr, gaussian_kernel, boundary="symm", mode="same")  # 卷積
    else:
        arr = cv2.GaussianBlur(arr, (3, 3), 0)
    return arr


if __name__ == "__main__":
    readFile = ReadFile(320, 240, 4, path("./data/rgba320x240.h")).np_memcpy_fixed_rgba()
    a = cv_plot(readFile.arr_dest, cv2.COLOR_RGB2BGR)
    plt.imshow(a)
    plt.show()

    readFile = ReadFile(720, 480, 2, create_string_buffer(b"./data/dump_2p0_th_0918.dat")).np_memcpy_bin()
    th_rgb = cv_plot(readFile.arr_dest, cv2.COLOR_YUV2RGB_Y422)
    # plt.imshow(th_rgb)
    # plt.show()

    readFile = ReadFile(720, 480, 2, create_string_buffer(b"./data/dump_2p0_img_0918.dat")).np_memcpy_bin()
    csi_rgb = cv2.cvtColor(readFile.arr_dest, cv2.COLOR_YUV2RGB_Y422)
    # plt.imshow(csi_rgb)
    # plt.show()

    th_gray = cv2.cvtColor(th_rgb, cv2.COLOR_RGB2GRAY)
    # plt.imshow(th_gray, cmap="gray")
    # plt.show()

    csi_gray = cv2.cvtColor(csi_rgb, cv2.COLOR_RGB2GRAY)
    # plt.imshow(csi_gray, cmap="gray")
    # plt.show()

    pass
