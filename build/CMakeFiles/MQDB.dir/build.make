# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jio/projects/GPUcomputing/lab1/MQDB

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jio/projects/GPUcomputing/build

# Include any dependencies generated for this target.
include CMakeFiles/MQDB.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/MQDB.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/MQDB.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/MQDB.dir/flags.make

CMakeFiles/MQDB.dir/main.cpp.o: CMakeFiles/MQDB.dir/flags.make
CMakeFiles/MQDB.dir/main.cpp.o: /home/jio/projects/GPUcomputing/lab1/MQDB/main.cpp
CMakeFiles/MQDB.dir/main.cpp.o: CMakeFiles/MQDB.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jio/projects/GPUcomputing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/MQDB.dir/main.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/MQDB.dir/main.cpp.o -MF CMakeFiles/MQDB.dir/main.cpp.o.d -o CMakeFiles/MQDB.dir/main.cpp.o -c /home/jio/projects/GPUcomputing/lab1/MQDB/main.cpp

CMakeFiles/MQDB.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MQDB.dir/main.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jio/projects/GPUcomputing/lab1/MQDB/main.cpp > CMakeFiles/MQDB.dir/main.cpp.i

CMakeFiles/MQDB.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MQDB.dir/main.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jio/projects/GPUcomputing/lab1/MQDB/main.cpp -o CMakeFiles/MQDB.dir/main.cpp.s

CMakeFiles/MQDB.dir/mqdb.cpp.o: CMakeFiles/MQDB.dir/flags.make
CMakeFiles/MQDB.dir/mqdb.cpp.o: /home/jio/projects/GPUcomputing/lab1/MQDB/mqdb.cpp
CMakeFiles/MQDB.dir/mqdb.cpp.o: CMakeFiles/MQDB.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jio/projects/GPUcomputing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/MQDB.dir/mqdb.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/MQDB.dir/mqdb.cpp.o -MF CMakeFiles/MQDB.dir/mqdb.cpp.o.d -o CMakeFiles/MQDB.dir/mqdb.cpp.o -c /home/jio/projects/GPUcomputing/lab1/MQDB/mqdb.cpp

CMakeFiles/MQDB.dir/mqdb.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MQDB.dir/mqdb.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jio/projects/GPUcomputing/lab1/MQDB/mqdb.cpp > CMakeFiles/MQDB.dir/mqdb.cpp.i

CMakeFiles/MQDB.dir/mqdb.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MQDB.dir/mqdb.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jio/projects/GPUcomputing/lab1/MQDB/mqdb.cpp -o CMakeFiles/MQDB.dir/mqdb.cpp.s

CMakeFiles/MQDB.dir/prod_mqdb.cpp.o: CMakeFiles/MQDB.dir/flags.make
CMakeFiles/MQDB.dir/prod_mqdb.cpp.o: /home/jio/projects/GPUcomputing/lab1/MQDB/prod_mqdb.cpp
CMakeFiles/MQDB.dir/prod_mqdb.cpp.o: CMakeFiles/MQDB.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jio/projects/GPUcomputing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/MQDB.dir/prod_mqdb.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/MQDB.dir/prod_mqdb.cpp.o -MF CMakeFiles/MQDB.dir/prod_mqdb.cpp.o.d -o CMakeFiles/MQDB.dir/prod_mqdb.cpp.o -c /home/jio/projects/GPUcomputing/lab1/MQDB/prod_mqdb.cpp

CMakeFiles/MQDB.dir/prod_mqdb.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MQDB.dir/prod_mqdb.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jio/projects/GPUcomputing/lab1/MQDB/prod_mqdb.cpp > CMakeFiles/MQDB.dir/prod_mqdb.cpp.i

CMakeFiles/MQDB.dir/prod_mqdb.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MQDB.dir/prod_mqdb.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jio/projects/GPUcomputing/lab1/MQDB/prod_mqdb.cpp -o CMakeFiles/MQDB.dir/prod_mqdb.cpp.s

# Object files for target MQDB
MQDB_OBJECTS = \
"CMakeFiles/MQDB.dir/main.cpp.o" \
"CMakeFiles/MQDB.dir/mqdb.cpp.o" \
"CMakeFiles/MQDB.dir/prod_mqdb.cpp.o"

# External object files for target MQDB
MQDB_EXTERNAL_OBJECTS =

MQDB: CMakeFiles/MQDB.dir/main.cpp.o
MQDB: CMakeFiles/MQDB.dir/mqdb.cpp.o
MQDB: CMakeFiles/MQDB.dir/prod_mqdb.cpp.o
MQDB: CMakeFiles/MQDB.dir/build.make
MQDB: CMakeFiles/MQDB.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jio/projects/GPUcomputing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable MQDB"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MQDB.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/MQDB.dir/build: MQDB
.PHONY : CMakeFiles/MQDB.dir/build

CMakeFiles/MQDB.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/MQDB.dir/cmake_clean.cmake
.PHONY : CMakeFiles/MQDB.dir/clean

CMakeFiles/MQDB.dir/depend:
	cd /home/jio/projects/GPUcomputing/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jio/projects/GPUcomputing/lab1/MQDB /home/jio/projects/GPUcomputing/lab1/MQDB /home/jio/projects/GPUcomputing/build /home/jio/projects/GPUcomputing/build /home/jio/projects/GPUcomputing/build/CMakeFiles/MQDB.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/MQDB.dir/depend

