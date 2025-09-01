cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(MyTorchApp)

set(CMAKE_PREFIX_PATH "libtorch")

find_package(Torch REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)
