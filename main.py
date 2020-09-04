import ctypes
from sultan.api import Sultan
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from PIL import Image, ImageDraw, ImageFont


def pwd():
    return os.getcwd()


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


def cmd_magick_convert_array():
    try:
        s = Sultan()
        s.convert("le.jpg -define h:format=rgba -depth 8 -size 100x100 rgba100x100.h").run()
    except Exception as e:
        print(e)


def cmd_magick_convert_image(aname, astring):
    try:
        if (astring == "jpeg" or astring == "jpg" or astring == "bmp" or astring == "png"):
            s = Sultan()
            s.convert(aname+"."+astring+" "+aname+"."+astring).run()
            print("Converted image:"+ aname+"."+astring)
        else:
            print("Unsupported Filename Extension")
    except Exception as e:
        print(e)


def np_memcpy(nrows=3, ncols=4,  n_bytes_type=1):
    arr_src = np.arange(nrows * ncols).astype(np.uint8)
    arr_dest = np.empty(shape=(nrows, ncols), dtype=np.uint8)
    print('arr_src:', arr_src)
    print('arr_dest:', arr_dest)
    clib = ctypes.cdll.LoadLibrary("test.so")
    clib.np_memcpy.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_size_t
    ]
    clib.np_memcpy.restype = ctypes.c_void_p
    print('\ncalling clib.np_memcpy ...\n')
    clib.np_memcpy(arr_dest, arr_src, nrows * ncols * n_bytes_type)
    print('arr_src:', arr_src)
    print('arr_dest:', arr_dest)


def np_memcpy_fixed(x=100, y=100, z=4, size=40000):
    arr_dest = np.empty(shape=(x, y, z), dtype=np.uint8)
    # print('arr_dest:', arr_dest)
    clib = ctypes.cdll.LoadLibrary("test.so")
    clib.np_memcpy_fixed.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8, ndim=3, flags='C_CONTIGUOUS'),
        ctypes.c_size_t
    ]
    clib.np_memcpy_fixed.restype = ctypes.c_void_p
    clib.np_memcpy_fixed(arr_dest, size)
    # print('arr_dest:', arr_dest)
    return arr_dest

def np_memcpy_fixed_rgba_to_argb(x=100, y=100, z=4, size=40000):
    arr_dest = np.empty(shape=(x, y, z), dtype=np.uint8)
    # print('arr_dest:', arr_dest)
    clib = ctypes.cdll.LoadLibrary("test.so")
    clib.np_memcpy_fixed_rgba_to_argb.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8, ndim=3, flags='C_CONTIGUOUS'),
        ctypes.c_size_t
    ]
    clib.np_memcpy_fixed_rgba_to_argb.restype = ctypes.c_void_p
    clib.np_memcpy_fixed_rgba_to_argb(arr_dest, size)
    # print('arr_dest:', arr_dest)
    return arr_dest


if __name__ == "__main__":
    np.set_printoptions(formatter={'int':hex})
    # This could be any command you want to execute as you were in bash
    try:
        print(pwd())
        print(walk("jpg"))
        cmd_magick_convert_array()
        cmd_magick_convert_image("le", "jpg")
        np_arr_rgba = np_memcpy_fixed(100, 100, 4)
        np_arr_argb = np_arr_rgba[:,2]
        print((np_arr_argb))
        imgplot = plt.imshow(np_arr_rgba)
        plt.show()


    except Exception as e:
        print(e)
