#include <chrono>
#include <fmt/core.h>

#include <casadi/casadi.hpp>
#include <frc/EigenCore.h>
#include <frc/system/Discretization.h>
#include <frc/system/plant/LinearSystemId.h>
#include <units/angle.h>
#include <units/angular_acceleration.h>
#include <units/angular_velocity.h>
#include <units/time.h>
#include <units/voltage.h>

int main() {
  auto start = std::chrono::system_clock::now();

  constexpr auto T = 5_s;
  constexpr auto dt = 5_ms;
  constexpr int N = T / dt;

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

  opti.solver("ipopt");
  auto end1 = std::chrono::system_clock::now();

  auto sol = opti.solve();
  auto end2 = std::chrono::system_clock::now();

  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  fmt::print("Setup time: {} ms\n",
      duration_cast<microseconds>(end1 - start).count() / 1000.0);
  fmt::print("Solve time: {} ms\n",
      duration_cast<microseconds>(end2 - end1).count() / 1000.0);
}
