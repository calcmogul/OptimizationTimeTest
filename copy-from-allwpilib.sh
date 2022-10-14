#!/bin/bash

rm -rf src/frc/autodiff src/frc/optimization/ src/wpi/IntrusiveSharedPtr.h
cp -r ~/frc/wpilib/allwpilib/wpimath/src/main/native/include/frc/autodiff/ src/frc/
cp -r ~/frc/wpilib/allwpilib/wpimath/src/main/native/include/frc/optimization/ src/frc/
cp -r ~/frc/wpilib/allwpilib/wpimath/src/main/native/cpp/autodiff/ src/frc/
cp -r ~/frc/wpilib/allwpilib/wpimath/src/main/native/cpp/optimization/ src/frc/
cp ~/frc/wpilib/allwpilib/wpiutil/src/main/native/include/wpi/IntrusiveSharedPtr.h src/wpi/
cp ~/frc/wpilib/allwpilib/wpiutil/src/main/native/include/wpi/scope src/wpi/

find src/frc/autodiff -type f -name \*\.cpp -exec sed -i 's!wpi/DenseMap.h!unordered_map!' {} \;
find src/frc/autodiff -type f -name \*\.cpp -exec sed -i 's!wpi::DenseMap!std::unordered_map!' {} \;
wpiformat
