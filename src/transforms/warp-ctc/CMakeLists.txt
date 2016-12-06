IF (APPLE)
    cmake_minimum_required(VERSION 3.4)
ELSE()
    cmake_minimum_required(VERSION 2.8)
ENDIF()

project(ctc_release)

IF (NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2 -g")
ENDIF()

IF (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -g")
    add_definitions(-DAPPLE)
ENDIF()

include_directories(include)

FIND_PACKAGE(CUDA 6.5)
MESSAGE(STATUS "cuda found ${CUDA_FOUND}")

# need to be at least 30 or __shfl_down in reduce wont compile
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -O2")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")

if (NOT APPLE)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fopenmp")
ENDIF()

IF (APPLE)
    EXEC_PROGRAM(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
    STRING(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
    MESSAGE(STATUS "DARWIN_VERSION=${DARWIN_VERSION}")

    #for el capitain have to use rpath

    IF (DARWIN_VERSION LESS 15)
        set(CMAKE_SKIP_RPATH TRUE)
    ENDIF ()

ELSE()
    #always skip for linux
    set(CMAKE_SKIP_RPATH TRUE)
ENDIF()


IF (CUDA_FOUND)

    MESSAGE(STATUS "Building shared library with GPU support")

    CUDA_ADD_LIBRARY(warpctc SHARED src/ctc_entrypoint.cu src/reduce.cu)
    TARGET_LINK_LIBRARIES(warpctc ${CUDA_curand_LIBRARY})

    add_executable(test_cpu tests/test_cpu.cpp )
    TARGET_LINK_LIBRARIES(test_cpu warpctc)
    SET_TARGET_PROPERTIES(test_cpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -g --std=c++11")

    cuda_add_executable(test_gpu tests/test_gpu.cu)
    TARGET_LINK_LIBRARIES(test_gpu warpctc ${CUDA_curand_LIBRARY})

    INSTALL(TARGETS warpctc
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib/static)

ELSE()
    MESSAGE(STATUS "Building shared library with no GPU support")

    if (NOT APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
    ENDIF()

    ADD_LIBRARY(warpctc SHARED src/ctc_entrypoint.cpp)

    add_executable(test_cpu tests/test_cpu.cpp )
    TARGET_LINK_LIBRARIES(test_cpu warpctc)
    SET_TARGET_PROPERTIES(test_cpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

    INSTALL(TARGETS warpctc
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib/static)
ENDIF()


