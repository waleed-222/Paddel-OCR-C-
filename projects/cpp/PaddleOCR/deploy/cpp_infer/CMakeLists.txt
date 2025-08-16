cmake_minimum_required(VERSION 3.10)
project(ocr_system CXX C)

option(WITH_MKL        "Compile demo with MKL/OpenBlas support, default use MKL."       ON)
option(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    OFF)
option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   ON)
option(WITH_TENSORRT   "Compile demo with TensorRT."                                     OFF)

SET(PADDLE_LIB "" CACHE PATH "Location of Paddle Inference library")
SET(OPENCV_DIR "" CACHE PATH "Location of OpenCV")
SET(CUDA_LIB "" CACHE PATH "Location of CUDA libraries")
SET(CUDNN_LIB "" CACHE PATH "Location of cuDNN libraries")
SET(TENSORRT_DIR "" CACHE PATH "Location of TensorRT")

set(DEMO_NAME "ocr_system")

# Force static linking with MT if WITH_STATIC_LIB is enabled
macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(${flag_var} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endif()
    endforeach()
endmacro()

# Basic checks
if(NOT DEFINED PADDLE_LIB)
  message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
endif()

if(NOT DEFINED OPENCV_DIR)
  message(FATAL_ERROR "please set OPENCV_DIR with -DOPENCV_DIR=/path/opencv")
endif()

# Include & link paths
if(WIN32)
  include_directories("${PADDLE_LIB}/paddle/fluid/inference")
  include_directories("${PADDLE_LIB}/paddle/include")
  link_directories("${PADDLE_LIB}/paddle/lib")
  find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/build NO_DEFAULT_PATH)
else()
  find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/share/OpenCV NO_DEFAULT_PATH)
  include_directories("${PADDLE_LIB}/paddle/include")
  link_directories("${PADDLE_LIB}/paddle/lib")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("${CMAKE_SOURCE_DIR}/")
include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
include_directories("${PADDLE_LIB}/third_party/install/glog/include")
include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
include_directories("${PADDLE_LIB}/third_party/install/xxhash/include")
include_directories("${PADDLE_LIB}/third_party/install/zlib/include")
include_directories("${PADDLE_LIB}/third_party/boost")
include_directories("${PADDLE_LIB}/third_party/eigen3")

link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")
link_directories("${PADDLE_LIB}/paddle/lib")

if (WITH_MKL)
  add_definitions(-DUSE_MKL)
  include_directories("${PADDLE_LIB}/third_party/install/mklml/include")

  if (WIN32)
    set(MATH_LIB
      ${PADDLE_LIB}/third_party/install/mklml/lib/mklml.lib
      ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5md.lib
    )
  else()
    set(MATH_LIB
      ${PADDLE_LIB}/third_party/install/mklml/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
      ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX}
    )
  endif()

  set(MKLDNN_PATH "${PADDLE_LIB}/third_party/install/mkldnn")
if(EXISTS ${MKLDNN_PATH})
  include_directories("${MKLDNN_PATH}/include")
  if (WIN32)
    if(EXISTS "${MKLDNN_PATH}/lib/mkldnn.lib")
      set(MKLDNN_LIB "${MKLDNN_PATH}/lib/mkldnn.lib")
    else()
      message(WARNING "mkldnn.lib not found — skipping linking mkldnn.lib.")
      set(MKLDNN_LIB "")  # ✅ لا نحاول ربط الملف المفقود
    endif()
  else ()
    set(MKLDNN_LIB "${MKLDNN_PATH}/lib/libmkldnn.so.0")
  endif()
endif()

else()
  if(WIN32)
    set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/openblas${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
  endif()
endif()

if(WITH_STATIC_LIB)
  if(WIN32)
    set(DEPS ${PADDLE_LIB}/paddle/lib/paddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
  endif()
else()
  if(WIN32)
    set(DEPS ${PADDLE_LIB}/paddle/lib/paddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
  else()
    set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif()
endif()

set(DEPS ${DEPS}
  ${MATH_LIB} ${MKLDNN_LIB}
  glog gflags_static libprotobuf xxhash
  ${OpenCV_LIBS}
  libcmt shlwapi
)

# Enable static runtime if needed
if(WIN32)
  add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
  set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd /openmp")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT /openmp")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd /openmp")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT /openmp")
  SET(ONNXRUNTIME_DIR "" CACHE PATH "Location of ONNX Runtime")
  if (NOT DEFINED ONNXRUNTIME_DIR)
    message(FATAL_ERROR "please set ONNXRUNTIME_DIR with -DONNXRUNTIME_DIR=/path/onnxruntime")
  endif()
  if(WITH_STATIC_LIB)
    safe_set_static_flag()
    add_definitions(-DSTATIC_LIB)
  endif()
endif()
include_directories("${ONNXRUNTIME_DIR}/include")
link_directories("${ONNXRUNTIME_DIR}/lib")

# Windows will use .lib, Linux will use .so
if(WIN32)
  set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
else()
  set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif()
# Source files
AUX_SOURCE_DIRECTORY(./src SRCS)
add_executable(${DEMO_NAME} ${SRCS})
target_link_libraries(${DEMO_NAME} ${DEPS} ${ONNXRUNTIME_LIB})

if(WIN32)
  add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
      $<TARGET_FILE_DIR:${DEMO_NAME}>
  )
endif()

# Safe DLL copy only if files exist
if (WIN32 AND WITH_MKL)
  file(GLOB DLLS_TO_COPY
    "${PADDLE_LIB}/third_party/install/mklml/lib/mklml.dll"
    "${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5md.dll"
    "${PADDLE_LIB}/third_party/install/mkldnn/lib/mkldnn.dll"
  )

  foreach(DLL_FILE IN LISTS DLLS_TO_COPY)
    if(EXISTS "${DLL_FILE}")
      add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DLL_FILE}" $<TARGET_FILE_DIR:${DEMO_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DLL_FILE}" "$<TARGET_FILE_DIR:${DEMO_NAME}>/../release"
      )
    else()
      message(WARNING "DLL not found: ${DLL_FILE}")
    endif()
  endforeach()
endif()


# Output directory (where executable is built)
set(TARGET_OUTPUT_DIR $<TARGET_FILE_DIR:${DEMO_NAME}>)

# Copy OpenCV DLL
add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "F:/projects/cpp/opencv/build/x64/vc16/bin/opencv_world4110.dll"
        ${TARGET_OUTPUT_DIR}
)

# Copy Paddle DLLs
add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "F:/projects/cpp/paddle_inference/paddle/lib/common.dll"
        ${TARGET_OUTPUT_DIR}
)

add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "F:/projects/cpp/paddle_inference/paddle/lib/paddle_inference.dll"
        ${TARGET_OUTPUT_DIR}
)
