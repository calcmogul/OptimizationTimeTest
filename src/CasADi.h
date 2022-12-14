// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#pragma once

#include <casadi/casadi.hpp>
#include <units/time.h>

/**
 * Creates a flywheel quadratic optimization problem with CasADi.
 *
 * @param dt Timestep duration.
 * @param N Number of samples in the problem.
 * @param diagnostics True if diagnostic prints should be enabled.
 */
casadi::Opti FlywheelCasADi(units::second_t dt, int N,
                            bool diagnostics = false);

/**
 * Creates a cart-pole nonlinear optimization problem with CasADi.
 *
 * @param dt Timestep duration.
 * @param N Number of samples in the problem.
 * @param diagnostics True if diagnostic prints should be enabled.
 */
casadi::Opti CartPoleCasADi(units::second_t dt, int N,
                            bool diagnostics = false);
