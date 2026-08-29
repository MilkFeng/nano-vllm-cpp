#!/bin/bash

# If VCPKG_ROOT not set, set it to the default location
if [ -z "$VCPKG_ROOT" ]; then
  echo "VCPKG_ROOT not set, using default location: ~/vcpkg"
  VCPKG_ROOT=~/vcpkg
else
  echo "Using VCPKG_ROOT: $VCPKG_ROOT"
fi

VCPKG_CMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake

cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=${VCPKG_CMAKE_TOOLCHAIN_FILE}

cmake --build build --config Release --parallel
