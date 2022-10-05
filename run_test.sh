#!/bin/bash

make -j$(nproc) && ./build/OptimizationTimeTest
