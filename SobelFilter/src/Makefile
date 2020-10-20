######################################
# 生成可執行文件
######################################
#source file
#源文件，自動找所有.c和.cpp文件，並將目標定義為同名.o文件
SOURCE  := $(wildcard *.c) $(wildcard *.cpp)
OBJS    := $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SOURCE)))
  
#target you can change test to what you want
#目標文件名，輸入任意你想要的執行文件名
TARGET  := sobel

#compile and lib parameter
#編譯參數
# CC      := arm-none-eabi-gcc-4.9.3
CC      := gcc
LIBS    := 
# LIBS    := -lm
LDFLAGS :=
DEFINES := -DCOLOR_TYPE=RGBA_BYTES -D__UNIXES__
			# -D_USE_MATH_DEFINE \
			# -DSOD_DISABLE_CNN \
			# -DSOD_DISABLE_IMG_WRITER \
			# -DOS_OTHER 
			#-DSOD_DISABLE_IMG_READER  
			# -D__FPU_PRESENT=1 \
			# -DARM_MATH_CM4 \
			# -DDEBUG=1 \
			# -DARMCM4_FP \
			# -D__CS_COMPILER__ \

INCLUDE := -I. -I../inc
CFLAGS  := -std=c99 -g -Wall -O2 $(DEFINES) $(INCLUDE)
CXXFLAGS:= $(CFLAGS) -DHAVE_CONFIG_H

# -O2
# -g
# -mfpu=fpv4-sp-d16
# -mfloat-abi=hard
# -fno-common
# -fdata-sections
# -ffunction-sections

# -mcpu=cortex-m4
# -mthumb
# -IInc

#i think you should do anything here
#下面的基本上不需要做任何改動了
.PHONY : everything objs clean veryclean rebuild

everything : $(TARGET)

all : $(TARGET)

objs : $(OBJS)

rebuild: clean everything all

clean :
	rm -fr *.so
	rm -fr *.o
	rm -fr $(TARGET)

$(TARGET) : $(OBJS)
	$(CC) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LIBS)