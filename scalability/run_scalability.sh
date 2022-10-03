#!/bin/bash

make -j$(nproc) && ./build/ScalabilityTest
