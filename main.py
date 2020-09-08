import ctypes
from sultan.api import Sultan
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import time

# from PIL import Image, ImageDraw, ImageFont


def debug(func):
    def wrapper(*args, **kwargs):
        print("[DEBUG]: enter {}()".format(func.__name__))
        return func(*args, **kwargs)

    return wrapper  # 返回


@debug
def pwd():
    return os.getcwd()


@debug
def walk(str):
    shpfiles = []
    for dirpath, subdirs, files in os.walk(pwd()):
        for x in files:
            if x.endswith(str):
                shpfiles.append(os.path.join(dirpath, x))
    return shpfiles
    # shpfiles = [os.path.join(d, x)
    #         for d, dirs, files in os.walk(path)
    #         for x in files if x.endswith(".shp")]
    # for root, dirs, files in os.walk(".", topdown=False):
    #     for name in files:
    #         # print(os.path.join(root, name))
    #         if name[0] != ".":
    #             if name[0] == "t":
    #                 print(name)


@debug
def list_image_name(sub):
    ListOfImagePath = walk(sub)
    CurrentPath = pwd()
    listOfImageName = list()
    for x in range(len(ListOfImagePath)):
        listOfImageName.append(
            ListOfImagePath[x].replace(CurrentPath, "").lstrip("/").lstrip("\\")
        )
    print(listOfImageName)


@debug
def list_image_path(sub):
    ListOfImagePath = walk(sub)
    print(ListOfImagePath)


@debug
def cmd_magick_convert_array(col, row):
    try:
        size_str = str(col) + "x" + str(row)
        s = Sultan()
        # s.convert("-quality 100 -resize 500x500 le.jpg le.bmp").and_().convert(
        # "le.bmp -define h:format=rgba -depth 8 -size 500x500! rgba.h"
        # ).run()
        s.convert(
            "-quality 100 -resize " + size_str + "! le.jpg le.bmp"
        ).and_().convert(
            "le.bmp -define h:format=rgba -depth 8 -size "
            + size_str
            + " rgba"
            + size_str
            + ".h"
        ).run()
        s.gcc("-fPIC -shared -o ./main.so ./main.c").run()
        print("gcc " + "-fPIC -shared -o ./main.so ./main.c")
        # print(s.stdout)  # the stdout
        # print(s.stderr)  # the stderr
        # print(s.traceback)  # the traceback
        # print(str(s.rc))  # the return code
        time.sleep(1)
    except Exception as e:
        print(e)


@debug
def np_memcpy(row=4, col=4, rgba=4, lib_str="main.so"):
    arr_src = np.arange(row * col * rgba).astype(np.uint8)
    arr_dest = np.empty(shape=(row, col, rgba), dtype=np.uint8)
    # print("arr_src:\n", arr_src)
    # print("arr_dest:\n", arr_dest)
    clib = ctypes.cdll.LoadLibrary("main.so")
    clib.np_memcpy.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8, ndim=3, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
    ]
    clib.np_memcpy.restype = ctypes.c_void_p
    print("\ncalling clib.np_memcpy ...\n")
    clib.np_memcpy(arr_dest, arr_src, row * col * rgba)
    print("arr_dest:\n", arr_dest)
    image_change(arr_dest)
    print("arr_dest:\n", arr_dest)
    return arr_dest


@debug
def np_memcpy_fixed_rgba(row, col, rgba=4, lib_str="main.so"):
    arr_dest = np.empty(shape=(row, col, rgba), dtype=np.uint8, order="C")
    clib = ctypes.cdll.LoadLibrary(lib_str)
    clib.np_memcpy_fixed_rgba.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS", ndim=3),
        ctypes.c_size_t,
    ]
    clib.np_memcpy_fixed_rgba.restype = ctypes.c_void_p
    clib.np_memcpy_fixed_rgba(arr_dest, row * col * rgba)
    return arr_dest


@debug
def np_memcpy_fixed_argb(row, col, argb=4, lib_str="main.so"):
    arr_dest = np.empty(shape=(row, col, argb), dtype=np.uint8)
    clib = ctypes.cdll.LoadLibrary(lib_str)
    clib.np_memcpy_fixed_argb.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS", ndim=3),
        ctypes.c_size_t,
    ]
    clib.np_memcpy_fixed_argb.restype = ctypes.c_void_p
    clib.np_memcpy_fixed_argb(arr_dest, row * col * argb)
    return arr_dest


@debug
def image_init(row, col, dim=3, in_name="le", in_sub="jpg"):
    np.set_printoptions(formatter={"int": hex})
    list_image_path("jpg")
    cmd_magick_convert_array(row, col)
    return np_memcpy_fixed_rgba(col, row)


@debug
def image_change(np_arr_rgba, alpha=0xFF):
    if type(np_arr_rgba).__module__ == np.__name__:
        print("image_change succes")
    else:
        print("image_change fail")
    # dims = np_arr_rgba.shape
    # row = dims[0]
    # col = dims[1]
    # Set transparency depending on row and col positio
    np_arr_rgba[0:50, 0:300, 3] = alpha


@debug
def arr_plot(np_arr1, np_arr2):
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(np_arr1)
    plt.subplot(212)
    plt.imshow(np_arr2)
    plt.show()


@debug
def image_plot(np_arr1):
    img_org = Image.open("le.bmp").convert("RGBA")
    np_arr1 = np.array(img_org)
    plt.imshow(np_arr1)
    plt.show()


if __name__ == "__main__":
    # m*n row major
    np_arr_rgba = image_init(300, 100, 3, "le", "jpg")
    np_arr_rgba1 = np_arr_rgba.copy()
    image_change(np_arr_rgba, 0x128)
    np_arr_rgba2 = np_arr_rgba.copy()
    arr_plot(np_arr_rgba1, np_arr_rgba2)

    # np_arr_rgba = np_memcpy(4, 5)
    # np_arr_rgba1 = np_arr_rgba.copy()
    # np_arr_rgba2 = np_arr_rgba.copy()
    # print(type(np_arr_rgba1))
    # image_plot(np_arr_rgba1, np_arr_rgba2)

