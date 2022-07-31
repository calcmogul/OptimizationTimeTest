// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "frc/optimization/InequalityConstraint.h"

#include <utility>

using namespace frc;

InequalityConstraint::InequalityConstraint(AutodiffWrapper variable)
    : variable{std::move(variable)} {}
