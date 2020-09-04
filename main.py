import ctypes
from sultan.api import Sultan
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
def cmd_magick_convert_array(x_size, y_size):
    try:
        s = Sultan()
        s.convert("le.jpg -define h:format=rgba -depth 8 -size "+str(x_size)+"x100"+str(y_size)+" rgba100x100.h").run()
        s.cc("-fPIC -shared -o ./test.so ./main.c").run()
    except Exception as e:
        print(e)


@debug
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


@debug
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


@debug
def np_memcpy_fixed_rgba(x=100, y=100, rgba=4, size=40000):
    arr_dest = np.empty(shape=(x, y, rgba), dtype=np.uint8)
    # print('arr_dest:', arr_dest)
    clib = ctypes.cdll.LoadLibrary("test.so")
    clib.np_memcpy_fixed_rgba.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8, ndim=3, flags='C_CONTIGUOUS'),
        ctypes.c_size_t
    ]
    clib.np_memcpy_fixed_rgba.restype = ctypes.c_void_p
    clib.np_memcpy_fixed_rgba(arr_dest, x*y*rgba)
    # print('arr_dest:', arr_dest)
    return arr_dest


@debug
def np_memcpy_fixed_rgba_to_argb(x=100, y=100, argb=4, size=40000):
    arr_dest = np.empty(shape=(x, y, argb), dtype=np.uint8)
    # print('arr_dest:', arr_dest)
    clib = ctypes.cdll.LoadLibrary("test.so")
    clib.np_memcpy_fixed_rgba_to_argb.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8, ndim=3, flags='C_CONTIGUOUS'),
        ctypes.c_size_t
    ]
    clib.np_memcpy_fixed_rgba_to_argb.restype = ctypes.c_void_p
    clib.np_memcpy_fixed_rgba_to_argb(arr_dest, x*y*argb)
    # print('arr_dest:', arr_dest)
    return arr_dest


@debug
def image_convert(name, sub):
    print(walk(sub))
    cmd_magick_convert_image(name, sub)


@debug
def image_init(name, sub, x_size, y_size, dim=4):
    image_convert(name, sub)
    np.set_printoptions(formatter={'int':hex})
    cmd_magick_convert_array(x_size, y_size)
    np_arr_rgba = np_memcpy_fixed_rgba(x_size, y_size, dim)
    return np_arr_rgba

@debug
def image_change(np_arr_rgba):
    if type(np_arr_rgba).__module__ == np.__name__:
        print("image_change succes")
    else:
        print("image_change fail")
    # Set transparency depending on x position
    dims = np_arr_rgba.shape
    x_size = dims[0]
    y_size = dims[1]
    for x in range(x_size):
        for y in range(y_size):
            np_arr_rgba[y, x, 3] = 128
    # # Set
    # for x in range(100):
    #     for y in range(100):
    #         np_arr_rgba[y, x, 0] = np_arr_rgba[y, x, 3]
    # print(np_arr_rgba)


@debug
def image_plot():
    imgplot = plt.imshow(np_arr_rgba)
    plt.show()


if __name__ == "__main__":
    # This could be any command you want to execute as you were in bash
    np_arr_rgba = image_init("le", "jpg", 100, 100, 4)
    image_change(np_arr_rgba)
    image_plot()
    # print(np_memcpy_fixed_rgba_to_argb(100, 100, 4))