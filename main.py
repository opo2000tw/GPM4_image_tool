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
        if (model == 1 or model == 2 or model == 3 or model == 4):
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
        if isinstance(self.name, str):
            self.name = self.name.encode(encoding="utf-8")
            self.name = create_string_buffer(self.name)
        if isinstance(self.name, bytes):
            self.name = create_string_buffer(self.name)
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


def binary_oprator(arr):
    """
    開運算---先腐蝕，後膨脹。去除圖像中小的亮點（CV_MOP_OPEN）；
    閉運算---先膨脹，後腐蝕。去除圖像中小的暗點（CV_MOP_CLOSE）；
    形態學梯度---原圖膨脹圖像 — 原圖腐蝕圖像（CV_MOP_GRADIENT）；
    頂帽---原圖像 — 原圖像開操作。保留小亮點，去除大亮點。（CV_MOP_TOPHAT）；
    黑帽---原圖像閉操作 — 原圖像（CV_MOP_BLACKHAT）；
    """
    # 核的大小和形狀
    ret0 = arr
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    ret1 = cv.morphologyEx(arr, cv.MORPH_OPEN, kernel, iterations=3)
    ret2 = cv.morphologyEx(arr, cv.MORPH_CLOSE, kernel, iterations=3)
    ret3 = cv.morphologyEx(arr, cv.MORPH_GRADIENT, kernel, iterations=10)
    ret4 = cv.morphologyEx(arr, cv.MORPH_TOPHAT, kernel, iterations=5)
    ret5 = cv.morphologyEx(arr, cv.MORPH_BLACKHAT, kernel, iterations=5)
    ret_hstack = np.hstack((ret0, ret1, ret2, ret3, ret4, ret5))
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
    if (len(img.shape) >= 3):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # x方向
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobelx_8u = np.uint8(np.absolute(sobelx))

    # y方向
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    abs_sobely_8u = np.uint8(np.absolute(sobely))

    # x、y方向
    sobelxy = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
    abs_sobelxy_8u = np.uint8(np.absolute(sobelxy))

    scaled_sobel = np.uint8(255 * abs_sobelxy_8u / np.max(abs_sobelxy_8u))
    thresh_min = 40
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # cv2.imshow("gray", gray)
    # cv2.imshow("sobelx", abs_sobelx_8u)
    # cv2.imshow("sobely", abs_sobely_8u)
    # cv2.imshow("sobelxy", abs_sobelxy_8u)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return sxbinary


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


# 1. Global histogram equalization
def globalEqualHist(image):
    # If you want to equalize the picture, you must convert the picture to a grayscale image
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(gray)  # There are related notes and examples in the documentation
    # equalizeHist(src, dst=None) function can only process single-channel data, src is the input image object matrix, which must be single-channel uint8 type matrix data
    # dst: Output image matrix (src has the same shape)
    cv.imshow("global equalizeHist", dst)
    # print(len(image.shape)) # The shape length of the color image is 3
    # print(len(gray.shape)) # The shape length of the grayscale image is 2
    # print(gray.shape) # Grayscale image has only height and width
    return dst


def localEqualHist(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=5, tileGridSize=(7, 7))
    dst = clahe.apply(gray)
    cv.imshow("clahe image", dst)
    return dst


def THRESH_OTSU(image):
    if (len(image.shape) >= 3):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    return dst


#創建直方圖
def create_rgb_hist(image):
    h, w, c = image.shape
    rgbHist = np.zeros([16 * 16 * 16, 1], np.float32)
    bsize = 256 / 16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index = np.int(b / bsize) * 16 * 16 + np.int(g / bsize) * 16 + np.int(r / bsize)
            rgbHist[index, 0] += 1
    return rgbHist


@debug
def mat_plot(arr1, arr2, arr3, mode=0):
    plt.figure(1)
    if mode == 0:
        plt.subplot(311)
        plt.imshow(arr1)
        plt.subplot(312)
        plt.imshow(arr2)
        plt.subplot(313)
        plt.imshow(arr3)
    elif mode == 1:
        plt.subplot(311)
        plt.imshow(arr1, cmap="gray")
        plt.subplot(312)
        plt.imshow(arr2, cmap="gray")
        plt.subplot(313)
        plt.imshow(arr3, cmap="gray")
    elif mode == 2:
        plt.subplot(311)
        plt.hist(arr1.ravel(), 256)
        plt.subplot(312)
        plt.hist(arr2.ravel(), 256)
        plt.subplot(313)
        plt.hist(arr3.ravel(), 256)
    else:
        print("fail")
    plt.show()


def sol_bel(image):
    sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    sobelCombined = cv2.bitwise_or(sobelX, sobelY)
    return sobelCombined


def test(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    smooth = cv2.addWeighted(blur, 1.5, img, -0.5, 0)
    return smooth


def matchAB(grayA, grayB):
    # 讀取圖像數據
    # imgA = cv2.imread(fileA)
    # imgB = cv2.imread(fileB)

    # 轉換成灰色
    # grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    # grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    # 獲取圖片A的大小
    height, width = grayA.shape

    # 取局部圖像，尋找匹配位置
    result_window = np.zeros((height, width), dtype=grayA.dtype)
    for start_y in range(0, height - 100, 10):
        for start_x in range(0, width - 100, 10):
            window = grayA[start_y:start_y + 100, start_x:start_x + 100]
            match = cv2.matchTemplate(grayB, window, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(match)
            matched_window = grayB[max_loc[1]:max_loc[1] + 100, max_loc[0]:max_loc[0] + 100]
            result = cv2.absdiff(window, matched_window)
            result_window[start_y:start_y + 100, start_x:start_x + 100] = result

    plt.imshow(result_window)
    plt.show()


def Thin(image, array):
    h, w = image.shape
    iThin = image.copy()
    for i in range(h):
        for j in range(w):
            if image[i, j] == 0:
                a = [1] * 9
                for k in range(3):
                    for l in range(3):
                        if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and iThin[i - 1 + k, j - 1 + l] == 0:
                            a[k * 3 + l] = 0
                sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                iThin[i, j] = array[sum] * 255
    return iThin


def Two(image):
    h, w = image.shape
    iTwo = np.empty(shape=(h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            iTwo[i, j] = 0 if image[i, j] < 200 else 255
    return iTwo


def VThin(image, array):
    h, w = image.shape
    NEXT = 1
    for i in range(h):
        for j in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i, j - 1] + image[i, j] + image[i, j + 1] if 0 < j < w - 1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and image[i - 1 + k, j - 1 + l] == 255:
                                a[k * 3 + l] = 1
                    sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                    image[i, j] = array[sum] * 255
                    if array[sum] == 1:
                        NEXT = 0
    return image


def HThin(image, array):
    h, w = image.shape
    NEXT = 1
    for j in range(w):
        for i in range(h):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i - 1, j] + image[i, j] + image[i + 1, j] if 0 < i < h - 1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and image[i - 1 + k, j - 1 + l] == 255:
                                a[k * 3 + l] = 1
                    sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                    image[i, j] = array[sum] * 255
                    if array[sum] == 1:
                        NEXT = 0
    return image


array = [0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,\
         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0,\
         1,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0]


def Xihua(image, array, num=1):
    iXihua = image.copy()
    for i in range(num):
        VThin(iXihua, array)
        HThin(iXihua, array)
    return iXihua


if __name__ == "__main__":
    readFile = ReadFile(32, 24, 4, ("./data/ORG_32x24_dump-4.dat")).np_memcpy_bin()
    ORG_32x24_dump_g = cv2.cvtColor(readFile.arr_dest, cv2.COLOR_RGBA2GRAY)
    ORG_32x24_dump_r = 255 - ORG_32x24_dump_g
    ORG_32x24_dump_th = THRESH_OTSU(ORG_32x24_dump_g)
    ORG_32x24_dump_s1 = sol_bel(ORG_32x24_dump_g)
    ORG_32x24_dump_s2 = sol_bel(ORG_32x24_dump_r)
    a = test(ORG_32x24_dump_s1)
    # ta = plt.hist(ORG_32x24_dump_g.ravel(), 256)

    readFile = ReadFile(320, 240, 4, "./data/ARGB_320x240_dump-4.dat").np_memcpy_bin()
    ARGB_320x240_dump_g = cv2.cvtColor(readFile.arr_dest, cv2.COLOR_RGBA2GRAY)
    ARGB_320x240_dump_r = 255 - ORG_32x24_dump_g
    ARGB_320x240_dump_th = THRESH_OTSU(ARGB_320x240_dump_g)
    ARGB_320x240_dump_s1 = sol_bel(ARGB_320x240_dump_g)
    ARGB_320x240_dump_s2 = sol_bel(ARGB_320x240_dump_r)

    # tb = plt.hist(ARGB_320x240_dump_g.ravel(), 256)
    b = test(ARGB_320x240_dump_s1)

    readFile = ReadFile(320, 240, 2, "./data/YUYV_320x240_csi_dump-4.dat").np_memcpy_bin()
    YUYV_64x48_dump = cv2.cvtColor(readFile.arr_dest, cv2.COLOR_YUV2RGB_Y422)
    YUYV_64x48_dump_g = cv2.cvtColor(YUYV_64x48_dump, cv2.COLOR_RGBA2GRAY)
    YUYV_64x48_dump_r = 255 - ORG_32x24_dump_g
    YUYV_64x48_dump_th = THRESH_OTSU(YUYV_64x48_dump_g)
    YUYV_64x48_dump_s1 = sol_bel(YUYV_64x48_dump_g)
    YUYV_64x48_dump_s2 = sol_bel(YUYV_64x48_dump_r)
    c = test(YUYV_64x48_dump_s1)
    # tc = plt.hist(YUYV_64x48_dump_g.ravel(), 256)

    # plt.imshow(ARGB_320x240_dump_g + (THRESH_OTSU(YUYV_64x48_dump_g)), cmap="gray")
    # plt.show()

    mat_plot(ORG_32x24_dump_g, ARGB_320x240_dump_g, YUYV_64x48_dump_g, 1)
    # mat_plot(ORG_32x24_dump_g, ARGB_320x240_dump_g, YUYV_64x48_dump_g, 3)
    # mat_plot(ORG_32x24_dump_th, ARGB_320x240_dump_th, YUYV_64x48_dump_th, 1)
    # mat_plot(ORG_32x24_dump_r, ARGB_320x240_dump_r, YUYV_64x48_dump_r, 1)
    # mat_plot(ORG_32x24_dump_s1, ARGB_320x240_dump_s1, YUYV_64x48_dump_s1, 1)
    # mat_plot(THRESH_OTSU(ORG_32x24_dump_s1), THRESH_OTSU(ARGB_320x240_dump_s1), THRESH_OTSU(YUYV_64x48_dump_s1), 1)
    # mat_plot(ORG_32x24_dump_s2, ARGB_320x240_dump_s2, YUYV_64x48_dump_s2, 1)
    # mat_plot(THRESH_OTSU(ORG_32x24_dump_s1), THRESH_OTSU(ARGB_320x240_dump_s1), THRESH_OTSU(YUYV_64x48_dump_s1), 2)
    # binary_oprator(THRESH_OTSU(ARGB_320x240_dump_s1))

    image = 255 - THRESH_OTSU(YUYV_64x48_dump_s1)
    iTwo = Two(image)
    iThin = Xihua(iTwo, array)
    # plt.imshow(iThin, cmap="gray")
    # plt.show()
    # plt.imshow(ARGB_320x240_dump_g + iThin, cmap="gray")
    # plt.show()

    
    mat_plot(ARGB_320x240_dump_g, iTwo, iThin, 1)
    iThin = cv2.cvtColor(iThin, cv2.COLOR_GRAY2RGB)
    h,w,c = iThin.shape
    for i in range(h):
        for j in range(w):
            if iThin[i, j, 0]>=250:
                iThin[i, j, 0] = 0x50

    # ARGB_320x240_dump_g = cv2.cvtColor(ARGB_320x240_dump_g + iThin, cv2.COLOR_GRAY2RGB)
    # iThin = cv2.cvtColor(iThin, cv2.COLOR_GRAY2RGB)
    # matchAB(ARGB_320x240_dump_g, iThin)
    
    # np.hstack(ORG_32x24_dump,ARGB_320x240_dump)
    # th_gray = cv2.cvtColor(th_rgb, cv2.COLOR_RGB2GRAY)
    # # plt.imshow(th_gray, cmap="gray")
    # # plt.show()

    # csi_gray = cv2.cvtColor(csi_rgb, cv2.COLOR_RGB2GRAY)
    # # plt.imshow(csi_gray, cmap="gray")
    # # plt.show()

    # src = cv.imread("handsomeboy.png")
    # cv.imshow("original image", src)

    # src0 = THRESH_OTSU(src)
    # cv.imshow("THRESH_OTSU", THRESH_OTSU(src0))
    # src1 = localEqualHist(src)
    # cv.imshow("lo_THRESH_OTSU", THRESH_OTSU(src1))
    # src2 = globalEqualHist(src)
    # cv.imshow("gl_THRESH_OTSU", THRESH_OTSU(src2))

    pass
    cv2.waitKey(0)
    cv2.destroyAllWindows()
