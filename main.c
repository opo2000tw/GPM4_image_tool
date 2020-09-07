#ifdef _WIN32
#   define API __declspec(dllexport)
#else
#   define API
#endif

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "rgba300x100.h"

API float *foo(void)
{
    static float bar[16] = {1, 2, 3, 0xff, 1, 2, 3, 0xff, 1, 2, 3, 0xff, 1, 2, 3, 0xff};
    // Populate bar
    return bar;
}

API void *np_memcpy(uint8_t *arr_dest, uint8_t *arr_src, size_t size)
{
    return memcpy(arr_dest, arr_src, size);
}

API void *np_memcpy_fixed_rgba(uint8_t *arr_dest, size_t size)
{
    return memcpy(arr_dest, (const void *)MagickImage, size);
}

API void *np_memcpy_fixed_argb(uint8_t *arr_dest, size_t size)
{
    memcpy(arr_dest, (const void *)MagickImage, size);
    for (uint8_t *rgba_ptr = arr_dest, *argb_ptr = arr_dest + size - 1; argb_ptr >= arr_dest; rgba_ptr++, argb_ptr--)
    {
        // *argb_ptr = *rgba_ptr >> 8 | 0xff000000;  // - this version doesn't change endianess
        *arr_dest = __builtin_bswap32(*rgba_ptr) >> 8 | 0xff000000;  // This does
    }
    return (void *) arr_dest;
}

int main()
{
    return 0;
}