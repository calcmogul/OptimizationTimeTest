#!/bin/bash

clear && make -j$(nproc) && clear && gdb ./build/OptimizationTimeTest --eval-command="run"
