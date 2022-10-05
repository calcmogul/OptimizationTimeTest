// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#pragma once

#include <frc/optimization/OptimizationProblem.h>
#include <units/time.h>

/**
 * Creates a flywheel quadratic optimization problem with OptimizationProblem.
 *
 * @param dt Timestep duration.
 * @param N Number of samples in the problem.
 */
frc::OptimizationProblem FlywheelOptimizationProblem(units::second_t dt, int N);

/**
 * Creates a cartpole nonlinear optimization problem with OptimizationProblem.
 *
 * @param dt Timestep duration.
 * @param N Number of samples in the problem.
 */
frc::OptimizationProblem CartpoleOptimizationProblem(units::second_t dt, int N);
