/*
 * Copyright 2018 Pedro Melgueira
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "macros.h"
#include "sobel.h"
#include "sod.h"
#include "utils.h"
#include "sensor80X60_RGB565.h"
// #include "file_operations.h"

#pragma region sobel
#if 0
#define ARGS_NEEDED 4

#ifndef COLOR_TYPE
#define COLOR_TYPE YUV422_BYTES
#endif
#define RGBA_BYTES 4
#define RGB_BYTES 3
#define YUV422_BYTES 2
#endif
#pragma endregion sobel

inline static void readFile(char *file_name, byte **buffer, int buffer_size)
{
    // Open
    FILE *file = fopen(file_name, "r");
    // Allocate memory for buffer
    *buffer = malloc(sizeof(byte) * buffer_size);
    // Read every char of file ONE BY ONE (not the whole thing at once)
    // We do this because this should look closer to the assembly implementation
    for (int i = 0; i < buffer_size; i++)
    {
        (*buffer)[i] = fgetc(file);
    }
    // Close
    fclose(file);
}

/*
 * Writes the buffer to a file
 */

inline static void writeFile(char *file_name, byte *buffer, int buffer_size)
{
    // Open
    FILE *file = fopen(file_name, "w");
    // Write all
    for (int i = 0; i < buffer_size; i++)
    {
        fputc(buffer[i], file);
    }
    // Close
    fclose(file);
}


int main(int argc, char *argv[])
{
#pragma region sobel
#if 0
    char *file_in,
         *file_out,
         *file_out_h,
         *file_out_v,
         *file_gray;
    int width,
        height,
        channels = COLOR_TYPE;
    int inter_files = 0,
        gray_file = 0,
        arg_index = ARGS_NEEDED;
    // Get arguments
    if (argc < ARGS_NEEDED)
    {
        printf("Usage: TODO\n");
        return 1;
    }
    else
    {
        arg_index = ARGS_NEEDED;
        // Get size of input image
        char *width_token = strtok(argv[3], "x");
        if (width_token)
        {
            width = atoi(width_token);
        }
        else
        {
            printf("Bad image size argument\n");
            return 1;
        }
        char *height_token = strtok(NULL, "x");
        if (height_token)
        {
            height = atoi(height_token);
        }
        else
        {
            printf("Bad image size argument\n");
            return 1;
        }
        file_in = argv[1];
        file_out = argv[2];
    }
    // File names
    while (arg_index < argc)
    {
        if (strcmp(argv[arg_index], "-i") == 0)
        {
            if (arg_index + 3 > argc)
            {
                printf("Usage: TODO\n");
                return 1;
            }
            inter_files = 1;
            file_out_h = argv[arg_index + 1];
            file_out_v = argv[arg_index + 2];
            arg_index += 3;
        }
        else if (strcmp(argv[arg_index], "-g") == 0)
        {
            if (arg_index + 2 > argc)
            {
                printf("Usage: TODO\n");
                return 1;
            }
            gray_file = 1;
            file_gray = argv[arg_index + 1];
            arg_index += 2;
        }
        else
        {
            printf("Argument \"%s\", is unknown.\n", argv[arg_index]);
            return 1;
        }
    }
    sobel_res sobel =
    {
#if (COLOR_TYPE == RGBA_BYTES)
        rgbaToGray,
#elif (COLOR_TYPE == RGB_BYTES)
        rgbToGray
#elif (COLOR_TYPE == YUV422_BYTES)
        yuv422ToGray
#else
#endif
    };
    readFile(file_in, &sobel.rgb, width * height * channels);
    int gray_size = sobelFilter(sobel.callback, &sobel.rgb, &sobel.gray,
                                &sobel.sobel_h_res, &sobel.sobel_v_res, &sobel.contour_img,
                                width, height, channels);
#if 1
    // Write gray image
    if (gray_file)
    {
        writeFile(file_gray, sobel.gray, gray_size);
    }
    // Write image after each sobel operator
    if (inter_files)
    {
        writeFile(file_out_h, sobel.sobel_h_res, gray_size);
        writeFile(file_out_v, sobel.sobel_v_res, gray_size);
    }
    // Write sobel img to a file
    writeFile(file_out, sobel.contour_img, gray_size);
#endif
#endif
#pragma endregion sobel
#if 0 // sod_rgb
    byte *tt_graypp;
    // Read file to rgb and get size
    readFile("imgs/ARGB_320x240_dump-4.dat", &tt_graypp, 320 * 240 * 4);
    sod_img imgRGBPP = set_sod_struct(FORMAT_RGBA_2_RGBA, tt_graypp, 320, 240, 4);
    // print_sod_zBlob(imgRGBPP);
    sod_img_save_as_png(imgRGBPP, "imgRGBPP.png");
    // sod_img_blob_save_as_bmp("rgba.bmp", get_byte_array(imgRGBPP), 320, 240, 4);
#endif
#if 1 // sod_rgba
    byte *tt_rgba;
    // Read file to rgb and get size
    readFile("imgs/ARGB_320x240_dump-4.dat", &tt_rgba, 320 * 240 * 4);
    sod_img imgRGBA = set_sod_struct(FORMAT_RGBA_2_RGB, tt_rgba, 320, 240, 3);
    sod_img_save_as_png(imgRGBA, "imgRGBA.png");
#endif
#if 1 // sod_yuv422_rgb
    byte *tt_yuv;
    readFile("imgs/YUYV_320x240_csi_dump-4.dat", &tt_yuv, 320 * 240 * 2);
    sod_img imgYUV = set_sod_struct(FORMAT_YUV422_PACKED_VY1UY0_2_RGB, tt_yuv, 320, 240, 3);
    sod_img_save_as_png(imgYUV, "imgYUV.png");
    // print_sod_hwc(img);
    // print_sod_zBlob(imgYUV);
#endif
#if 1 // sod_yuv422_gray
    byte *tt_grayp;
    readFile("imgs/YUYV_320x240_csi_dump-4.dat", &tt_grayp, 320 * 240 * 2);
    sod_img imgGrayP = set_sod_struct(FORMAT_YUV422_PACKED_VY1UY0_2_GRAY, tt_grayp, 320, 240, 1);
    printf("%x", *tt_grayp);
    // print_sod_zBlob(imgGrayP);
    sod_img_save_as_png(imgGrayP, "imgGrayP.png");
#endif
#if 0 // sod_gray
    byte *trgb;
    // Read file to rgb and get size
    readFile("imgs/rgb.rgb", &trgb, 100 * 100 * 3);
    sod_img imgRGB888_1 = set_sod_struct(FORMAT_RGB_2_RGB, trgb, 100, 100, 3);
    sod_img_save_as_png(imgRGB888_1, "rgba8888.png");
#endif
#if 0
    byte *tt_rgb;
    readFile("imgs/rgb.rgb", &tt_rgb, 100 * 100 * 3);
    sod_img imgRGB888_2 = set_sod_struct(FORMAT_RGB888_2_RGB888, tt_rgb, 100, 100, 3);
    sod_img_save_as_png(imgRGB888_2, "imgRGB888_2.png");
#endif
#if 0
    sod_img imgRGB565_1 = set_sod_struct(FORMAT_RGB565_2_RGBA8888, (byte *)sensor80X60_RGB565, 80, 60, 4);
    sod_img_save_as_png(imgRGB565_1, "imgRGB565_1.png");
    // byte *tt_rgb;
    // readFile("imgs/rgb.rgb", &tt_rgb, 100 * 100 * 3);
    sod_img imgRGB565_2 = set_sod_struct(FORMAT_RGB565_2_RGB888, (byte *)sensor80X60_RGB565, 80, 60, 3);
    // printf("[%02X][%02X][%02X]---\r\n", *sensor80X60_RGB565, *(sensor80X60_RGB565 + 1), *(sensor80X60_RGB565 + 2));
    // printf("[%02X][%02X][%02X]---\r\n", get_pixel_value(imgRGB565_2, 0, 0, 0), get_pixel_value(imgRGB565_2, 0, 0, 1), get_pixel_value(imgRGB565_2, 0, 0, 2));
    sod_img_save_as_png(imgRGB565_2, "imgRGB565_2.png");
    // print_sod_zBlob(imgRGB565_2);
    // print_sod_hwc(imgRGB565_2);
#endif
    sod_img imgRGB565_ = set_sod_struct(FORMAT_RGB565_2_RGBA8888, &sensor80X60_RGB565, 80, 60, 4);
    sod_img imgRGB565__ = set_sod_struct(FORMAT_RGB565_2_RGB888, &sensor80X60_RGB565, 80, 60, 3);
    print_sod_zBlob(imgRGB565_);
    print_sod_zBlob(imgRGB565__);
    lsb_msb_test();

    sod_composite_image(imgRGBA, imgGrayP, 0, 0 );
    sod_img imgThresholdp = sod_threshold_image(imgSOBEL, 0.48);
    sod_img_save_as_png(imgGrayP, "1a.png");
    sod_img_save_as_png(imgThresholdp, "1b.png");
    sod_composite_image(imgRGBA, imgCanny0, 0, 0 );
    sod_img_save_as_png(imgCanny0, "2a.png");

    sod_img imgGrayShaped = sod_sharpen_filtering_image(imgGrayP);
    sod_img imgGaryEqual = sod_equalize_histogram(imgGrayP);
    sod_img imgSOBEL = sod_sobel_image (imgGrayP);
    sod_img imgThreshold = sod_threshold_image(imgSOBEL, 0.49);
    sod_img imgCanny0 = sod_canny_edge_image(imgGrayP, 0 /*  Set this to 1 if you want to reduce noise */);
    sod_img imgCanny1 = sod_canny_edge_image(imgGrayP, 1 /*  Set this to 1 if you want to reduce noise */);
    sod_img imgOTSU = sod_otsu_binarize_image (imgSOBEL);
    sod_img imgBinarize = sod_binarize_image(imgSOBEL, 0);
    sod_img imgDILATE = sod_dilate_image(imgBinarize, 1);
    sod_img imgERODE = sod_erode_image(imgDILATE, 1);
    sod_img imgThin = sod_hilditch_thin_image(imgERODE);
    sod_img_save_as_png(imgGrayShaped, "imgGrayShaped.png");
    sod_img_save_as_png(imgGaryEqual, "imgGaryEqual.png");
    sod_img_save_as_png(imgCanny0, "imgCanny0.png");
    sod_img_save_as_png(imgCanny1, "imgCanny1.png");
    sod_img_save_as_png(imgOTSU, "imgOTSU.png");
    sod_img_save_as_png(imgSOBEL, "imgSOBEL.png");
    sod_img_save_as_png(imgThin, "imgThin.png");
    sod_img_save_as_png(imgERODE, "imgERODE.png");
    sod_img_save_as_png(imgDILATE, "imgDILATE.png");
    sod_img_save_as_png(imgBinarize, "imgBinarize.png");

    printf("End\r\n");
    return 0;
}
