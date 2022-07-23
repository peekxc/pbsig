cmake_minimum_required(VERSION 3.14...3.22) # https://cliutils.gitlab.io/modern-cmake/chapters/intro/dodonot.html
project(pbsig VERSION "0.1")
# find_package(PythonExtensions REQUIRED)
# set(CMAKE_GENERATOR "Unix Makefiles" CACHE INTERNAL "" FORCE)

#find_package(pybind11 CONFIG REQUIRED)
add_subdirectory(extern/pybind11)
pybind11_add_module(boundary MODULE src/pbsig/boundary.cpp)
install(TARGETS boundary DESTINATION .)



# python_extension_module(boundary)
# find_package(Python COMPONENTS Interpreter Development)
# find_package(pybind11 CONFIG)

# pybind11_add_module(boundary src/pbsig/boundary.cpp)

# add_library(boundary MODULE src/pbsig/boundary.cpp)
# target_include_directories(boundary PRIVATE extern/pybind11/include)
# install(TARGETS boundary LIBRARY DESTINATION .)

set(ignoreMe "${SKBUILD}")