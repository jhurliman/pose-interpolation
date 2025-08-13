cmake_minimum_required(VERSION 3.14)
project(se3_foxglove LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# Foxglove SDK version configuration (override with -DFOXGLOVE_SDK_VERSION=...)
set(FOXGLOVE_SDK_VERSION "0.10.1" CACHE STRING "Foxglove SDK version to fetch")

# Determine architecture component
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(FG_ARCH aarch64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  set(FG_ARCH x86_64)
else()
  message(FATAL_ERROR "Unsupported architecture '${CMAKE_SYSTEM_PROCESSOR}' for Foxglove SDK")
endif()

# Determine OS / toolchain component
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(FG_SYS apple-darwin)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(FG_SYS unknown-linux-gnu)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(FG_SYS pc-windows-msvc)
else()
  message(FATAL_ERROR "Unsupported system '${CMAKE_SYSTEM_NAME}' for Foxglove SDK")
endif()

set(FG_TRIPLE "${FG_ARCH}-${FG_SYS}")

# SHA256 hashes for each supported triple (v0.10.1)
set(_FG_SHA_aarch64-apple-darwin 87441008a1f1f3322e3b3eb404b3849d5ac9cc5455dc5341ba7787fa46781890)
set(_FG_SHA_aarch64-pc-windows-msvc c54591c70582ec43bcd6dc02ea1d64c89bfd4f0ff97f07c2724c20191b56b671)
set(_FG_SHA_aarch64-unknown-linux-gnu 84fb538cf7439a6c522b2c3bbb893adb4da89c1fda521a55821572e95bb7a502)
set(_FG_SHA_x86_64-apple-darwin 2145d126039256c1cce03ab04d740f61793a2851073ef16a57d63f36283beaf3)
set(_FG_SHA_x86_64-pc-windows-msvc 7c88153cb31850c8d06ed1e46289d85375a66c4a6c3d40753a68763db8b23cd6)
set(_FG_SHA_x86_64-unknown-linux-gnu 099b022d1e911fbb7af1b1a85ced5234b10c7de753b24d402afcb595bc3add01)

set(FOXGLOVE_SDK_BASE_URL "https://github.com/foxglove/foxglove-sdk/releases/download/sdk%2Fv${FOXGLOVE_SDK_VERSION}")
set(FOXGLOVE_SDK_ARCHIVE "foxglove-v${FOXGLOVE_SDK_VERSION}-cpp-${FG_TRIPLE}.zip")
set(FOXGLOVE_SDK_URL "${FOXGLOVE_SDK_BASE_URL}/${FOXGLOVE_SDK_ARCHIVE}")
set(FOXGLOVE_SDK_SHA256 ${_FG_SHA_${FG_TRIPLE}})

if(NOT FOXGLOVE_SDK_SHA256)
  message(FATAL_ERROR "No SHA256 hash known for Foxglove SDK triple '${FG_TRIPLE}'")
endif()

message(STATUS "Foxglove SDK: ${FOXGLOVE_SDK_URL}")

FetchContent_Declare(
  foxglove
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL "${FOXGLOVE_SDK_URL}"
  URL_HASH SHA256=${FOXGLOVE_SDK_SHA256}
)
FetchContent_MakeAvailable(foxglove)

add_executable(se3_foxglove main.cpp)

target_include_directories(
  se3_foxglove PRIVATE ${foxglove_SOURCE_DIR}/include
                       ${foxglove_SOURCE_DIR}/include/foxglove)

file(GLOB FOXGLOVE_SOURCES CONFIGURE_DEPENDS "${foxglove_SOURCE_DIR}/src/*.cpp"
     "${foxglove_SOURCE_DIR}/src/server/*.cpp")
target_sources(se3_foxglove PRIVATE ${FOXGLOVE_SOURCES})
if(WIN32)
  set(FOXGLOVE_LIB ${foxglove_SOURCE_DIR}/lib/foxglove.lib)
else()
  set(FOXGLOVE_LIB ${foxglove_SOURCE_DIR}/lib/libfoxglove.a)
endif()
target_link_libraries(se3_foxglove PRIVATE ${FOXGLOVE_LIB})
