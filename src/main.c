#ifdef _WIN32
#   define API __declspec(dllexport)
#else
#   define API
#endif

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "../inc/sod.h"
#include "../data/rgba320x240.h"

#define YUYV 1
#define VY1UY0 YUYV

API float *foo(void)
{
    static float bar[16] = {1, 2, 3, 0xff, 1, 2, 3, 0xff, 1, 2, 3, 0xff, 1, 2, 3, 0xff};
    // Populate bar
    return bar;
}

API void np_memcpy(uint8_t *arr_dest, uint8_t *arr_src, size_t size)
{
    memcpy(arr_dest, arr_src, size);
}

API void np_memcpy_fixed_rgba(uint8_t *arr_dest, size_t size)
{
    memcpy(arr_dest, (const void *)MagickImage, size);
}

API void np_memcpy_bin(uint8_t *arr_dest, size_t size, char *name)
{
    char buff[size];
    FILE *fp;
    fp = fopen(name, "rb");
    printf("size,%zu\r\n", size);
    if (fp == NULL)
    {
        printf("fopen error,%s\r\n", name);
        fclose(fp);
    }
    if (fread(buff, 1, size, fp) != size) // fread回傳讀取的byte數
    {
        printf("fread error,%zu\r\n", size);
    }
    memcpy(arr_dest, buff, size);
    fclose(fp);
}

API void func(uint8_t *buff)
{
    printf("%s", buff);
}

int main()
{
    func((uint8_t*)"Start");
    
    return 0;
}