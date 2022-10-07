#!/bin/bash

clear && make -j$(nproc) && clear && ./build/OptimizationTimeTest
