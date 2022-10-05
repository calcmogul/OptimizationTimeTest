// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "OptimizationProblem.h"

#include <frc/EigenCore.h>
#include <frc/system/Discretization.h>
#include <frc/system/plant/LinearSystemId.h>
#include <units/angle.h>
#include <units/angular_acceleration.h>
#include <units/angular_velocity.h>
#include <units/voltage.h>

frc::OptimizationProblem FlywheelOptimizationProblem(units::second_t dt,
                                                     int N) {
  // Flywheel model:
  // States: [velocity]
  // Inputs: [voltage]
  auto system = frc::LinearSystemId::IdentifyVelocitySystem<units::radians>(
      1_V / 1_rad_per_s, 1_V / 1_rad_per_s_sq);
  frc::Matrixd<1, 1> A;
  frc::Matrixd<1, 1> B;
  frc::DiscretizeAB<1, 1>(system.A(), system.B(), dt, &A, &B);

  frc::OptimizationProblem problem;
  auto X = problem.DecisionVariable(1, N + 1);
  auto U = problem.DecisionVariable(1, N);

  // Dynamics constraint
  for (int k = 0; k < N; ++k) {
    problem.SubjectTo(X.Col(k + 1) == A * X.Col(k) + B * U.Col(k));
  }

  // State and input constraints
  problem.SubjectTo(X.Col(0) == 0.0);
  problem.SubjectTo(-12 <= U);
  problem.SubjectTo(U <= 12);

  // Cost function - minimize error
  frc::Matrixd<1, 1> r{10.0};
  frc::VariableMatrix J = 0.0;
  for (int k = 0; k < N + 1; ++k) {
    J += ((r - X.Col(k)).Transpose() * (r - X.Col(k)));
  }
  problem.Minimize(J);

  return problem;
}
