if test -r "build"; then
    cd build
    cmake ..
    make -j8
else
    mkdir build
    cp cmake/config.cmake build
    cd build
    cmake ..
    make -j8
fi
