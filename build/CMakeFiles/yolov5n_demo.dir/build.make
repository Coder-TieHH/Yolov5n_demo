# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rpdzkj/Desktop/yolov5n_demo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rpdzkj/Desktop/yolov5n_demo/build

# Include any dependencies generated for this target.
include CMakeFiles/yolov5n_demo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/yolov5n_demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/yolov5n_demo.dir/flags.make

CMakeFiles/yolov5n_demo.dir/src/timer.cc.o: CMakeFiles/yolov5n_demo.dir/flags.make
CMakeFiles/yolov5n_demo.dir/src/timer.cc.o: ../src/timer.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rpdzkj/Desktop/yolov5n_demo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/yolov5n_demo.dir/src/timer.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolov5n_demo.dir/src/timer.cc.o -c /home/rpdzkj/Desktop/yolov5n_demo/src/timer.cc

CMakeFiles/yolov5n_demo.dir/src/timer.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolov5n_demo.dir/src/timer.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rpdzkj/Desktop/yolov5n_demo/src/timer.cc > CMakeFiles/yolov5n_demo.dir/src/timer.cc.i

CMakeFiles/yolov5n_demo.dir/src/timer.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolov5n_demo.dir/src/timer.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rpdzkj/Desktop/yolov5n_demo/src/timer.cc -o CMakeFiles/yolov5n_demo.dir/src/timer.cc.s

CMakeFiles/yolov5n_demo.dir/src/yolo.cpp.o: CMakeFiles/yolov5n_demo.dir/flags.make
CMakeFiles/yolov5n_demo.dir/src/yolo.cpp.o: ../src/yolo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rpdzkj/Desktop/yolov5n_demo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/yolov5n_demo.dir/src/yolo.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolov5n_demo.dir/src/yolo.cpp.o -c /home/rpdzkj/Desktop/yolov5n_demo/src/yolo.cpp

CMakeFiles/yolov5n_demo.dir/src/yolo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolov5n_demo.dir/src/yolo.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rpdzkj/Desktop/yolov5n_demo/src/yolo.cpp > CMakeFiles/yolov5n_demo.dir/src/yolo.cpp.i

CMakeFiles/yolov5n_demo.dir/src/yolo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolov5n_demo.dir/src/yolo.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rpdzkj/Desktop/yolov5n_demo/src/yolo.cpp -o CMakeFiles/yolov5n_demo.dir/src/yolo.cpp.s

CMakeFiles/yolov5n_demo.dir/src/yolo_layer.cpp.o: CMakeFiles/yolov5n_demo.dir/flags.make
CMakeFiles/yolov5n_demo.dir/src/yolo_layer.cpp.o: ../src/yolo_layer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rpdzkj/Desktop/yolov5n_demo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/yolov5n_demo.dir/src/yolo_layer.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolov5n_demo.dir/src/yolo_layer.cpp.o -c /home/rpdzkj/Desktop/yolov5n_demo/src/yolo_layer.cpp

CMakeFiles/yolov5n_demo.dir/src/yolo_layer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolov5n_demo.dir/src/yolo_layer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rpdzkj/Desktop/yolov5n_demo/src/yolo_layer.cpp > CMakeFiles/yolov5n_demo.dir/src/yolo_layer.cpp.i

CMakeFiles/yolov5n_demo.dir/src/yolo_layer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolov5n_demo.dir/src/yolo_layer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rpdzkj/Desktop/yolov5n_demo/src/yolo_layer.cpp -o CMakeFiles/yolov5n_demo.dir/src/yolo_layer.cpp.s

CMakeFiles/yolov5n_demo.dir/src/yolov5n_demo.cpp.o: CMakeFiles/yolov5n_demo.dir/flags.make
CMakeFiles/yolov5n_demo.dir/src/yolov5n_demo.cpp.o: ../src/yolov5n_demo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rpdzkj/Desktop/yolov5n_demo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/yolov5n_demo.dir/src/yolov5n_demo.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolov5n_demo.dir/src/yolov5n_demo.cpp.o -c /home/rpdzkj/Desktop/yolov5n_demo/src/yolov5n_demo.cpp

CMakeFiles/yolov5n_demo.dir/src/yolov5n_demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolov5n_demo.dir/src/yolov5n_demo.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rpdzkj/Desktop/yolov5n_demo/src/yolov5n_demo.cpp > CMakeFiles/yolov5n_demo.dir/src/yolov5n_demo.cpp.i

CMakeFiles/yolov5n_demo.dir/src/yolov5n_demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolov5n_demo.dir/src/yolov5n_demo.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rpdzkj/Desktop/yolov5n_demo/src/yolov5n_demo.cpp -o CMakeFiles/yolov5n_demo.dir/src/yolov5n_demo.cpp.s

# Object files for target yolov5n_demo
yolov5n_demo_OBJECTS = \
"CMakeFiles/yolov5n_demo.dir/src/timer.cc.o" \
"CMakeFiles/yolov5n_demo.dir/src/yolo.cpp.o" \
"CMakeFiles/yolov5n_demo.dir/src/yolo_layer.cpp.o" \
"CMakeFiles/yolov5n_demo.dir/src/yolov5n_demo.cpp.o"

# External object files for target yolov5n_demo
yolov5n_demo_EXTERNAL_OBJECTS =

../bin/yolov5n_demo: CMakeFiles/yolov5n_demo.dir/src/timer.cc.o
../bin/yolov5n_demo: CMakeFiles/yolov5n_demo.dir/src/yolo.cpp.o
../bin/yolov5n_demo: CMakeFiles/yolov5n_demo.dir/src/yolo_layer.cpp.o
../bin/yolov5n_demo: CMakeFiles/yolov5n_demo.dir/src/yolov5n_demo.cpp.o
../bin/yolov5n_demo: CMakeFiles/yolov5n_demo.dir/build.make
../bin/yolov5n_demo: /usr/lib/libopencv_dnn.so.3.4.3
../bin/yolov5n_demo: /usr/lib/libopencv_ml.so.3.4.3
../bin/yolov5n_demo: /usr/lib/libopencv_objdetect.so.3.4.3
../bin/yolov5n_demo: /usr/lib/libopencv_shape.so.3.4.3
../bin/yolov5n_demo: /usr/lib/libopencv_stitching.so.3.4.3
../bin/yolov5n_demo: /usr/lib/libopencv_superres.so.3.4.3
../bin/yolov5n_demo: /usr/lib/libopencv_videostab.so.3.4.3
../bin/yolov5n_demo: /usr/lib/libopencv_viz.so.3.4.3
../bin/yolov5n_demo: /usr/lib/libopencv_calib3d.so.3.4.3
../bin/yolov5n_demo: /usr/lib/libopencv_features2d.so.3.4.3
../bin/yolov5n_demo: /usr/lib/libopencv_flann.so.3.4.3
../bin/yolov5n_demo: /usr/lib/libopencv_highgui.so.3.4.3
../bin/yolov5n_demo: /usr/lib/libopencv_photo.so.3.4.3
../bin/yolov5n_demo: /usr/lib/libopencv_video.so.3.4.3
../bin/yolov5n_demo: /usr/lib/libopencv_videoio.so.3.4.3
../bin/yolov5n_demo: /usr/lib/libopencv_imgcodecs.so.3.4.3
../bin/yolov5n_demo: /usr/lib/libopencv_imgproc.so.3.4.3
../bin/yolov5n_demo: /usr/lib/libopencv_core.so.3.4.3
../bin/yolov5n_demo: CMakeFiles/yolov5n_demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rpdzkj/Desktop/yolov5n_demo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable ../bin/yolov5n_demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yolov5n_demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/yolov5n_demo.dir/build: ../bin/yolov5n_demo

.PHONY : CMakeFiles/yolov5n_demo.dir/build

CMakeFiles/yolov5n_demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/yolov5n_demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/yolov5n_demo.dir/clean

CMakeFiles/yolov5n_demo.dir/depend:
	cd /home/rpdzkj/Desktop/yolov5n_demo/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rpdzkj/Desktop/yolov5n_demo /home/rpdzkj/Desktop/yolov5n_demo /home/rpdzkj/Desktop/yolov5n_demo/build /home/rpdzkj/Desktop/yolov5n_demo/build /home/rpdzkj/Desktop/yolov5n_demo/build/CMakeFiles/yolov5n_demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/yolov5n_demo.dir/depend
