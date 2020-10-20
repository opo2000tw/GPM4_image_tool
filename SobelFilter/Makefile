# Copyright 2018 Pedro Melgueira
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

.PHONY : all check run clean

all: sobel run
# 	cd tests && make

check:
# 	cd tests && make check

sobel:
	cd src && make && cd ../

run:
	convert imgs/img.png imgs/image_rgba.png
	convert imgs/img.png -alpha disassociate imgs/image_rgb.png
	convert imgs/image_rgb.png imgs/rgb.rgb
	./src/sobel 

rebuild: clean all

clean:
	cd src && make clean
	rm imgs/img_out* imgs/img.rgb *.png *.bmp *.rgb
