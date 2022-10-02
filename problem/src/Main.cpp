#include <chrono>

#include <fmt/core.h>
#include <frc/EigenCore.h>
#include <frc/optimization/Problem.h>
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

  frc::Problem problem;
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
  frc::VariableMatrix J = 0.0;
  for (int k = 0; k < N + 1; ++k) {
    J += ((10.0 - X.Col(k)).Transpose() * (10.0 - X.Col(k)));
  }
  problem.Minimize(J);

  auto end1 = std::chrono::system_clock::now();
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  fmt::print("Setup time: {} ms\n\n",
             duration_cast<microseconds>(end1 - start).count() / 1000.0);

  frc::SolverConfig config;
  config.diagnostics = true;
  problem.Solve(config);
}
