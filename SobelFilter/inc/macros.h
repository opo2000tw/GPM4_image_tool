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
#ifndef MACROS_H
#define MACROS_H

#define SOBEL_OP_SIZE 9

typedef unsigned char byte;

#define _USE_MATH_DEFINES
#define SOD_DISABLE_CNN

#define FRAME_BUF_ALIGN64                   0x3F
#define FRAME_BUF_ALIGN32                   0x1F
#define FRAME_BUF_ALIGN16                   0xF
#define FRAME_BUF_ALIGN8                    0x7

#if defined(OS_OTHER)
#define SOD_DISABLE_IMG_WRITER
#define SOD_DISABLE_IMG_READER
#define gp_calloc(nitems, size) pvPortCalloc(nitems, size)
#define gp_realloc(ptr, size) pvPortRealloc(ptr,size)
#elif defined(__UNIXES__)
#define SOD_DISABLE_IMG_READER
#define gp_malloc(size)         malloc(size)
#define gp_calloc(nitems, size) calloc(nitems, size)
#define gp_realloc(ptr, size)   realloc(ptr,size)
#else
//...
#endif

#endif
