#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include "macros.h"
#include "sod.h"

#define IO_ERROR -1
#define MAX_PIXEL_INTENSITY 255.0

#ifdef __cplusplus
#define ENUM(x) enum x
#define STRUCT(x) struct x
#else
#define ENUM(x) typedef enum x x; enum x
#define STRUCT(x) typedef struct x x; struct x
#endif

#ifdef __GNUC__
#define GNUC_ALIGN(n) __attribute__((align(n)))
#endif

ENUM(COLOR_FORMATE)
{
    FORMAT_RGB565_2_RGB888,
    FORMAT_RGB565_2_RGBA8888,
    FORMAT_RGB888_2_RGB888,
    FORMAT_RGB_2_RGB,
    FORMAT_RGBA_2_RGB,
    FORMAT_RGBA_2_RGBA,
    FORMAT_YUV422_PACKED_VY1UY0_2_RGB,
    FORMAT_YUV422_PACKED_VY1UY0_2_GRAY,
};

// Create a new instance of a sod image of the specified width
// and height, using the full RGB colour model.
sod_img create_image(int width, int height);

// Free the memory used by sod image provided as argument
void free_image(sod_img img);

#ifdef SOD_DISABLE_IMG_READER
// Create a sod image from the the image file at the specified location.
sod_img load_image(const char *path);
#endif

#ifdef SOD_DISABLE_IMG_WRITER
// Saves the given image in the given destination.
bool save_image(sod_img img, const char *path);
#endif

// Clones the image provided as argument
sod_img copy_image(sod_img img);

// Find the width of the provided image
int get_image_width(sod_img img);

// Find the height of the provided image
int get_image_height(sod_img img);

// Find the height of the provided image
int get_image_channel(sod_img img);

// Find the R/G/B value of the pixel at (x,y) in the image
// NOTE: (rgb = 0 for red, rgb = 1 for green, rgb = 2 for blue)
// int get_pixel_value(sod_img img, int rgb, int x, int y);

// Set the R/G/B pixel intensity for pixel at (x,y)
// NOTE: (rgb = 0 for red, rgb = 1 for green, rgb = 2 for blue)
// void set_pixel_value(sod_img img, int rgb, int x, int y, int val);

sod_img set_sod_struct(enum COLOR_FORMATE obj_format, byte *zBlob, int width, int height, int channels);
void print_sod_zBlob(sod_img img);
void print_sod_hwc(sod_img img);
byte *get_byte_array(sod_img img);
void lsb_msb_test(void);
int get_pixel_value(sod_img img, int x, int y, int rgb);
#endif

