#include "utils.h"
#include "macros.h"
#include <unistd.h>

#define DEFAULT_COMPRESSION_QUALITY -1
#define FULL_COLOUR_CHANNELS 3

#ifndef SOD_DISABLE_IMG_READER
sod_img load_image(const char *path)
{
    sod_img input;
    if ( access(path, F_OK) == IO_ERROR )
    {
        printf("[!] error reading from file %s (check it exists)\n", path);
        input.data = 0;
        return input;
    }
    input = sod_img_load_from_file(path, SOD_IMG_COLOR);
    if (input.data == 0)
    {
        printf("[!] unsupported image format (expecting jpeg, png or bmp)\n");
    }
    return input;
}
#endif

#ifndef SOD_DISABLE_IMG_WRITER
bool save_image(sod_img img, const char *path)
{
    int ret = sod_img_save_as_jpeg(img, path, DEFAULT_COMPRESSION_QUALITY);
    if (ret != SOD_OK)
    {
        printf("[!] error saving file to %s\n", path);
        return false;
    }
    return true;
}
#endif

int get_pixel_value(sod_img img, int x, int y, int rgb)
{
    float intensity = sod_img_get_pixel(img, x, y, rgb);
    int rgb_value =  intensity * MAX_PIXEL_INTENSITY;
    return rgb_value;
}

// static void set_pixel_value(sod_img img, int rgb, int x, int y, int val)
static void set_pixel_value(sod_img img, int x, int y, int rgb, int val)
{
    float intensity = val / MAX_PIXEL_INTENSITY;
    sod_img_set_pixel(img, x, y, rgb, intensity);
}

static void get_pixel_yuv422_packed_vy1uy0_to_rgb(sod_img img, byte *zBlob, int width, int height, int channels)
{
    int block = width * height;
    if (block % 3 != 0 || block % 2 != 0)
    {
        printf("[!] error\r\n");
    }
    if (img.h != height || img.w != width || img.c != 3)
    {
        printf("[!] error\r\n");
    }
    byte Y0, U, V, Y1;
    byte R, G, B;
    for (int j = 0; j < img.h; j++)
    {
        for (int i = 0; i < img.w ; i++)
        {
            Y1 = *(zBlob + 3);
            Y0 = *(zBlob + 1);
            U = *(zBlob + 2);
            V = *(zBlob + 0);
            R = Y0 + ((360 * (V - 128)) >> 8) ;
            G = Y0 - (( ( 88 * (U - 128)  + 184 * (V - 128)) ) >> 8) ;
            B = Y0 + ((455 * (U - 128)) >> 8) ;
            set_pixel_value(img, i, j, 0, R);
            set_pixel_value(img, i, j, 1, G);
            set_pixel_value(img, i, j, 2, B);;
            i += 1;
            R = Y1 + ((360 * (V - 128)) >> 8) ;
            G = Y1 - (( ( 88 * (U - 128)  + 184 * (V - 128)) ) >> 8) ;
            B = Y1 + ((455 * (U - 128)) >> 8) ;
            set_pixel_value(img, i, j, 0, R);
            set_pixel_value(img, i, j, 1, G);
            set_pixel_value(img, i, j, 2, B);
            zBlob += 4;
        }
    }
}

static void get_pixel_rgb888_to_rgb888(sod_img img, byte *zBlob, int width, int height, int channels)
{
    if (img.h != height || img.w != width || img.c != 3)
    {
        printf("[!] error\r\n");
    }
    byte R, G, B;
    for (int j = 0; j < img.h; j++)
    {
        for (int i = 0; i < img.w ; i++)
        {
            R = *(zBlob + 0);
            G = *(zBlob + 1);
            B = *(zBlob + 2);
            set_pixel_value(img, i, j, 0, R);
            set_pixel_value(img, i, j, 1, G);
            set_pixel_value(img, i, j, 2, B);
            zBlob += 3;
        }
    }
}

static void get_pixel_rgb565_to_rgba8888(sod_img img, byte *zBlob, int width, int height, int channels, uint8_t alpha)
{
    int block = width * height;
    if (block % 2 != 0)
    {
        printf("[!] error\r\n");
    }
    if (img.h != height || img.w != width || img.c != 4)
    {
        printf("[!] error\r\n");
    }
    byte R, G, B;
    uint8_t A = alpha;
    for (int j = 0; j < img.h; j++)
    {
        for (int i = 0; i < img.w ; i++)
        {
            R = *(zBlob + 1) & 0xf8;
            G = (*(zBlob + 1) << 5) | ((*(zBlob + 1) & 0xe0) >> 3);
            B = *(zBlob + 0) << 3;
            set_pixel_value(img, i, j, 0, R);
            set_pixel_value(img, i, j, 1, G);
            set_pixel_value(img, i, j, 2, B);
            set_pixel_value(img, i, j, 3, A);
            zBlob += 2;
        }
    }
}

static void get_pixel_rgb565_to_rgb888(sod_img img, byte *zBlob, int width, int height, int channels)
{
    int block = width * height;
    if (block % 2 != 0)
    {
        printf("[!] error\r\n");
    }
    if (img.h != height || img.w != width || img.c != 3)
    {
        printf("[!] error\r\n");
    }
    byte R, G, B;
    for (int j = 0; j < img.h; j++)
    {
        for (int i = 0; i < img.w ; i++)
        {
            R = *(zBlob + 1) & 0xf8;
            G = (*(zBlob + 1) << 5) | ((*(zBlob + 1) & 0xe0) >> 3);
            B = *(zBlob + 0) << 3;
            set_pixel_value(img, i, j, 0, R);
            set_pixel_value(img, i, j, 1, G);
            set_pixel_value(img, i, j, 2, B);
            zBlob += 2;
        }
    }
}

static void get_pixel_yuv422_packed_vy1uy0_to_gray(sod_img img, byte *zBlob, int width, int height, int channels)
{
    int block = width * height;
    if (block % 3 != 0 || block % 2 != 0)
    {
        printf("[!] error\r\n");
    }
    if (img.h != height || img.w != width || img.c != 1)
    {
        printf("[!] error\r\n");
    }
    byte Y0, Y1;
    for (int j = 0; j < img.h; j++)
    {
        for (int i = 0; i < img.w ; i++)
        {
            Y0 = *(zBlob + 1);
            set_pixel_value(img, i, j, 0, Y0);
            i += 1;
            Y1 = *(zBlob + 3);
            set_pixel_value(img, i, j, 0, Y1);
            zBlob += 4;
        }
    }
    // sod_img_yuv_to_rgb(img);
}

sod_img create_image(int width, int height)
{
    return sod_make_image(width, height, FULL_COLOUR_CHANNELS);
}

void free_image(sod_img img)
{
    sod_free_image(img);
}

sod_img copy_image(sod_img img)
{
    return sod_copy_image(img);
}

int get_image_width(sod_img img)
{
    return img.w;
}

int get_image_height(sod_img img)
{
    return img.h;
}

int get_image_channel(sod_img img)
{
    return img.c;
}

byte *get_byte_array(sod_img img)
{
    return (byte *)sod_image_to_blob(img);
}

sod_img set_sod_struct(enum COLOR_FORMATE input_format, byte *zBlob, int width, int height, int channels)
{
    printf("[%d][%s] enter\r\n", input_format, __func__);
    sod_img img = sod_make_image(width, height, channels/* img*/);
    switch (input_format)
    {
    case FORMAT_RGBA_2_RGBA: // rgba
        for (int y = 0; y < img.h; y++)
        {
            for (int x = 0; x < img.w ; x++)
            {
                set_pixel_value(img, x, y, 0, (*(zBlob)));
                zBlob++;
                set_pixel_value(img, x, y, 1, (*(zBlob)));
                zBlob++;
                set_pixel_value(img, x, y, 2, (*(zBlob)));
                zBlob++;
                set_pixel_value(img, x, y, 3, (*(zBlob)));
                zBlob++;
            }
        }
        break;
    case FORMAT_RGBA_2_RGB: // rgba
        for (int y = 0; y < img.h; y++)
        {
            for (int x = 0; x < img.w ; x++)
            {
                set_pixel_value(img, x, y, 0, (*(zBlob)));
                zBlob++;
                set_pixel_value(img, x, y, 1, (*(zBlob)));
                zBlob++;
                set_pixel_value(img, x, y, 2, (*(zBlob)));
                zBlob++;
                zBlob++;
            }
        }
        break;
    case FORMAT_RGB_2_RGB: // rgba
        for (int y = 0; y < img.h; y++)
        {
            for (int x = 0; x < img.w ; x++)
            {
                set_pixel_value(img, x, y, 0, (*(zBlob)));
                zBlob++;
                set_pixel_value(img, x, y, 1, (*(zBlob)));
                zBlob++;
                set_pixel_value(img, x, y, 2, (*(zBlob)));
                zBlob++;
            }
        }
        break;
    case FORMAT_RGB888_2_RGB888: // rgb
        get_pixel_rgb888_to_rgb888(img, zBlob, width, height, channels);
        break;
    case FORMAT_RGB565_2_RGB888: // rgb
        get_pixel_rgb565_to_rgb888(img, zBlob, width, height, channels);
        break;
    case FORMAT_RGB565_2_RGBA8888: // rgb
        get_pixel_rgb565_to_rgba8888(img, zBlob, width, height, channels, 0xff);
        break;
    case FORMAT_YUV422_PACKED_VY1UY0_2_RGB: // yuv422
        get_pixel_yuv422_packed_vy1uy0_to_rgb(img, zBlob, width, height, channels);
        break;
    case FORMAT_YUV422_PACKED_VY1UY0_2_GRAY: // yuv422
        get_pixel_yuv422_packed_vy1uy0_to_gray(img, zBlob, width, height, channels);
        break;
    default:
        printf("[%d][%s] error\r\n", input_format, __func__);
        break;
    }
    return img;
}

void print_sod_zBlob(sod_img img)
{
    byte *zBlob = get_byte_array(img);
    int count = 0;
    for (int i = 0; i < img.w * img.h * img.c; i++)
    {
        printf("[%X]=[%02X]", i, *(zBlob + i));
        if (count == img.c - 1)
        {
            count = 0;
            printf("\r\n");
            break;
        }
        else
        {
            printf(",");
            count++;
        }
    }
    printf("\r\n");
}

void print_sod_hwc(sod_img img)
{
    printf("[%d][%d][%d], total size = [%x]\r\n", img.w, img.h, img.c, img.w * img.h * img.c);
}

void lsb_msb_test()
{
    int a = 0x12345678; // 輸入變量0x12345678，系統為變量分配4個字節的空間，此時不知道linux下編程是小端字節序還是大端字節序
    char *p = (char *)&a; //定義一個指針，系統為指針分配一個字節的內存空間存放變量的首地址，然後讓這個指針指向變量內存的首地址
    if (*p == 0x78)  //a變量一共有4字節內存單元，如果從首地址開始取的第一個字節單元裡存放的數據為是0x78
        printf("LSB\r\n"); //打印：LSB
    else
        printf("MSB\r\n"); //否則打印:MSB
}