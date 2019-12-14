# StyleGAN C++

This is unofficial implementation of StyleGAN's generator in C++ using tensor4

* convertor.py - code that converts stylegan weights to a binary.
* compressor.cpp - code that compresses obtained binary using zfp.
* main.cpp - main part and entrypoint og the generator part.
* StyleGAN.cpp/.h - network blocks implementations.
* SConstruct - scons build script to build WebAsm version.
* CMakelists.txt - CMake script for native building.
* rgbs.pkl - retrained weights for all "to-RGB" layers except for the last one. (to produce "intermidiate" outputs)
