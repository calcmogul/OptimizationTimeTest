#!/bin/bash

cd problem
rm -rf src/frc/autodiff src/frc/optimization/ src/wpi/IntrusiveSharedPtr.h
cp -r ~/frc/wpilib/allwpilib/wpimath/src/main/native/include/frc/autodiff/ src/frc/
cp -r ~/frc/wpilib/allwpilib/wpimath/src/main/native/include/frc/optimization/ src/frc/
cp -r ~/frc/wpilib/allwpilib/wpimath/src/main/native/cpp/autodiff/ src/frc/
cp -r ~/frc/wpilib/allwpilib/wpimath/src/main/native/cpp/optimization/ src/frc/
cp ~/frc/wpilib/allwpilib/wpiutil/src/main/native/include/wpi/IntrusiveSharedPtr.h src/wpi/
cp ~/frc/wpilib/allwpilib/wpiutil/src/main/native/include/wpi/scope src/wpi/
cd ..

cd scalability
rm -rf src/frc/autodiff src/frc/optimization/ src/wpi/IntrusiveSharedPtr.h
cp -r ~/frc/wpilib/allwpilib/wpimath/src/main/native/include/frc/autodiff/ src/frc/
cp -r ~/frc/wpilib/allwpilib/wpimath/src/main/native/include/frc/optimization/ src/frc/
cp -r ~/frc/wpilib/allwpilib/wpimath/src/main/native/cpp/autodiff/ src/frc/
cp -r ~/frc/wpilib/allwpilib/wpimath/src/main/native/cpp/optimization/ src/frc/
cp ~/frc/wpilib/allwpilib/wpiutil/src/main/native/include/wpi/IntrusiveSharedPtr.h src/wpi/
cp ~/frc/wpilib/allwpilib/wpiutil/src/main/native/include/wpi/scope src/wpi/
cd ..
