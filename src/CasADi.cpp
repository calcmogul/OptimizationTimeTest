// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "CasADi.h"

#include <frc/system/Discretization.h>
#include <frc/system/plant/LinearSystemId.h>
#include <units/angle.h>
#include <units/angular_acceleration.h>
#include <units/angular_velocity.h>
#include <units/time.h>
#include <units/voltage.h>

casadi::Opti CasADiFlywheel(units::second_t dt, int N) {
  // Flywheel model:
  // States: [velocity]
  // Inputs: [voltage]
  auto system = frc::LinearSystemId::IdentifyVelocitySystem<units::radians>(
      1_V / 1_rad_per_s, 1_V / 1_rad_per_s_sq);
  frc::Matrixd<1, 1> A;
  frc::Matrixd<1, 1> B;
  frc::DiscretizeAB<1, 1>(system.A(), system.B(), dt, &A, &B);

  casadi::MX caA = A(0, 0);
  casadi::MX caB = B(0, 0);

  casadi::Opti opti;
  casadi::Slice all;
  auto X = opti.variable(1, N + 1);
  auto U = opti.variable(1, N);

  // Dynamics constraint
  for (int k = 0; k < N; ++k) {
    opti.subject_to(X(all, k + 1) == caA * X(all, k) + caB * U(all, k));
  }

  // State and input constraints
  opti.subject_to(X(all, 0) == 0.0);
  opti.subject_to(-12 <= U);
  opti.subject_to(U <= 12);

  // Cost function - minimize error
  casadi::MX J = 0.0;
  for (int k = 0; k < N + 1; ++k) {
    J += ((10.0 - X(all, k)).T() * (10.0 - X(all, k)));
  }
  opti.minimize(J);

  opti.solver("ipopt", {{"print_time", 0}},
              {{"print_level", 0}, {"sb", "yes"}});

  return opti;
}
