######################################
# 生成可執行文件
######################################
#source file
#源文件，自動找所有.c和.cpp文件，並將目標定義為同名.o文件
SOURCE  := $(wildcard *.c) $(wildcard *.cpp)
OBJS    := $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SOURCE)))
  
#target you can change test to what you want
#目標文件名，輸入任意你想要的執行文件名
TARGET  := test
  
#compile and lib parameter
#編譯參數
CC      := gcc
LIBS    :=
LDFLAGS :=
DEFINES :=
INCLUDE := -I.
CFLAGS  := -g -Wall -O3 $(DEFINES) $(INCLUDE)
CXXFLAGS:= $(CFLAGS) -DHAVE_CONFIG_H
  
  
#i think you should do anything here
#下面的基本上不需要做任何改動了
.PHONY : everything objs clean veryclean rebuild
  
everything : $(TARGET)
  
all : $(TARGET)
  
objs : $(OBJS)
  
rebuild: veryclean everything
				
clean :
	rm -fr *.so
	rm -fr *.o
	
veryclean : clean
	rm -fr $(TARGET)
  
$(TARGET) : $(OBJS)
	$(CC) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LIBS)


######################################
# 生成靜態鏈接庫
######################################
  
#target you can change test to what you want
#共享庫文件名，lib*.a
TARGET  := libtest.a
  
#compile and lib parameter
#編譯參數
CC      := gcc
AR      = ar
RANLIB  = ranlib
LIBS    :=
LDFLAGS :=
DEFINES :=
INCLUDE := -I.
CFLAGS  := -g -Wall -O3 $(DEFINES) $(INCLUDE)
CXXFLAGS:= $(CFLAGS) -DHAVE_CONFIG_H
  
#i think you should do anything here
#下面的基本上不需要做任何改動了
  
#source file
#源文件，自動找所有.c和.cpp文件，並將目標定義為同名.o文件
SOURCE  := $(wildcard *.c) $(wildcard *.cpp)
OBJS    := $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SOURCE)))
  
.PHONY : everything objs clean veryclean rebuild
  
everything : $(TARGET)
  
all : $(TARGET)
  
objs : $(OBJS)
  
rebuild: veryclean everything
				
clean :
	rm -fr *.o
	
veryclean : clean
	rm -fr $(TARGET)
  
$(TARGET) : $(OBJS)
	$(AR) cru $(TARGET) $(OBJS)
	$(RANLIB) $(TARGET)

######################################
# 生成動態鏈接庫
######################################
  
#target you can change test to what you want
#共享庫文件名，lib*.so
TARGET  := libtest.so
  
#compile and lib parameter
#編譯參數
CC      := gcc
LIBS    :=
LDFLAGS :=
DEFINES :=
INCLUDE := -I.
CFLAGS  := -g -Wall -O3 $(DEFINES) $(INCLUDE)
CXXFLAGS:= $(CFLAGS) -DHAVE_CONFIG_H
SHARE   := -fPIC -shared -o
  
#i think you should do anything here
#下面的基本上不需要做任何改動了
  
#source file
#源文件，自動找所有.c和.cpp文件，並將目標定義為同名.o文件
SOURCE  := $(wildcard *.c) $(wildcard *.cpp)
OBJS    := $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SOURCE)))
  
.PHONY : everything objs clean veryclean rebuild
  
everything : $(TARGET)
  
all : $(TARGET)
  
objs : $(OBJS)
  
rebuild: veryclean everything
				
clean :
	rm -fr *.o
	
veryclean : clean
	rm -fr $(TARGET)
  
$(TARGET) : $(OBJS)
	$(CC) $(CXXFLAGS) $(SHARE) $@ $(OBJS) $(LDFLAGS) $(LIBS)