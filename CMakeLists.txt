cmake_minimum_required(VERSION 3.0.0)
project(diffsolvers VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(pybind11 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
set(SOURCE_DIR "src/diffqcqp")
add_subdirectory("${SOURCE_DIR}")

add_executable(diffsolvers_test "${SOURCE_DIR}/main.cpp")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

target_link_libraries(diffsolvers_test qcqplib)

pybind11_add_module(diffsolvers "${SOURCE_DIR}/diffsolvers.cpp")
target_link_libraries(diffsolvers PRIVATE Eigen3::Eigen)
target_link_libraries(diffsolvers PRIVATE qcqplib)
target_compile_options(diffsolvers PRIVATE -fPIC)
