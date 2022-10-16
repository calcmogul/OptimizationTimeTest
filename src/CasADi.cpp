// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "CasADi.h"

#include <numbers>

#include <frc/EigenCore.h>
#include <frc/system/Discretization.h>
#include <frc/system/NumericalIntegration.h>
#include <frc/system/plant/LinearSystemId.h>
#include <units/angle.h>
#include <units/angular_acceleration.h>
#include <units/angular_velocity.h>
#include <units/force.h>
#include <units/length.h>
#include <units/voltage.h>

casadi::Opti FlywheelCasADi(units::second_t dt, int N) {
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

casadi::MX CartPoleDynamics(const casadi::MX& x, const casadi::MX& u) {
  // https://underactuated.mit.edu/acrobot.html#cart_pole
  //
  // q = [x, θ]ᵀ
  // q̇ = [ẋ, θ̇]ᵀ
  // u = f_x
  //
  // M(q)q̈ + C(q, q̇)q̇ = τ_g(q) + Bu
  // M(q)q̈ = τ_g(q) − C(q, q̇)q̇ + Bu
  // q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
  //
  //        [ m_c + m_p  m_p l cosθ]
  // M(q) = [m_p l cosθ    m_p l²  ]
  //
  //           [0  −m_p lθ̇ sinθ]
  // C(q, q̇) = [0       0      ]
  //
  //          [     0      ]
  // τ_g(q) = [−m_p gl sinθ]
  //
  //     [1]
  // B = [0]
  constexpr double m_c = (5_kg).value();        // Cart mass
  constexpr double m_p = (0.5_kg).value();      // Pole mass
  constexpr double l = (0.5_m).value();         // Pole length
  constexpr double g = (9.806_mps_sq).value();  // Acceleration due to gravity

  auto q = x(casadi::Slice{0, 2});
  auto qdot = x(casadi::Slice{2, 4});
  auto theta = q(1);
  auto thetadot = qdot(1);

  //        [ m_c + m_p  m_p l cosθ]
  // M(q) = [m_p l cosθ    m_p l²  ]
  casadi::MX M{2, 2};
  M(0, 0) = m_c + m_p;
  M(0, 1) = m_p * l * cos(theta);  // NOLINT
  M(1, 0) = m_p * l * cos(theta);  // NOLINT
  M(1, 1) = m_p * std::pow(l, 2);

  casadi::MX Minv{2, 2};
  Minv(0, 0) = M(1, 1);
  Minv(0, 1) = -M(0, 1);
  Minv(1, 0) = -M(1, 0);
  Minv(1, 1) = M(0, 0);
  auto detM = M(0, 0) * M(1, 1) - M(0, 1) * M(1, 0);
  Minv /= detM;

  //           [0  −m_p lθ̇ sinθ]
  // C(q, q̇) = [0       0      ]
  casadi::MX C{2, 2};
  C(0, 0) = 0;
  C(0, 1) = -m_p * l * thetadot * sin(theta);  // NOLINT
  C(1, 0) = 0;
  C(1, 1) = 0;

  //          [     0      ]
  // τ_g(q) = [-m_p gl sinθ]
  casadi::MX tau_g{2, 1};
  tau_g(0) = 0;
  tau_g(1) = -m_p * g * l * sin(theta);  // NOLINT

  //     [1]
  // B = [0]
  casadi::MX B{2, 1};
  B(0) = 1.0;
  B(1) = 0.0;

  // q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)
  casadi::MX qddot{4, 1};
  qddot(casadi::Slice{0, 2}) = qdot;
  qddot(casadi::Slice{2, 4}) =
      mtimes(Minv, tau_g - mtimes(C, qdot) + mtimes(B, u));
  return qddot;
}

casadi::Opti CartPoleCasADi(units::second_t dt, int N) {
  casadi::Opti opti;
  casadi::Slice all;

  constexpr auto u_max = 20_N;
  constexpr auto d = 1_m;
  constexpr auto d_max = 2_m;

  // x = [q, q̇]ᵀ = [x, θ, ẋ, θ̇]ᵀ
  auto X = opti.variable(4, N + 1);

  // Initial guess
  for (int k = 0; k < N; ++k) {
    opti.set_initial(X(0, k), static_cast<double>(k) / N * d.value());
    opti.set_initial(X(1, k), static_cast<double>(k) / N * std::numbers::pi);
  }

  // u = f_x
  auto U = opti.variable(1, N);

  // Initial conditions
  opti.subject_to(X(all, 0) == 0.0);

  // Final conditions
  opti.subject_to(X(0, N) == 1.0);
  opti.subject_to(X(1, N) == std::numbers::pi);
  opti.subject_to(X(2, N) == 0.0);
  opti.subject_to(X(3, N) == 0.0);

  // Cart position constraints
  opti.subject_to(X(0, all) >= 0.0);
  opti.subject_to(X(0, all) <= d_max.value());

  // Input constraints
  opti.subject_to(U >= -u_max.value());
  opti.subject_to(U <= u_max.value());

  // Dynamics constraints - RK4 integration
  for (int k = 0; k < N; ++k) {
    opti.subject_to(
        X(all, k + 1) ==
        frc::RK4<decltype(CartPoleDynamics), casadi::MX, casadi::MX>(
            CartPoleDynamics, X(all, k), U(all, k), dt));
  }

  // Minimize sum squared inputs
  casadi::MX J = 0.0;
  for (int k = 0; k < N; ++k) {
    J += U(all, k).T() * U(all, k);
  }
  opti.minimize(J);

  opti.solver("ipopt", {{"print_time", 0}},
              {{"print_level", 0}, {"sb", "yes"}});

  return opti;
}
