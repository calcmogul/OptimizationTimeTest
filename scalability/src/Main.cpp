#include <chrono>
#include <fstream>

#include <fmt/core.h>

#include <casadi/casadi.hpp>
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
  constexpr auto T = 5_s;

  std::ofstream scalability{"Flywheel problem scalability.csv"};
  if (!scalability.is_open()) {
    return 1;
  }

  scalability << "Flywheel samples,CasADi solve time (ms),Problem solve time (ms)\n";

  fmt::print(
      "Solving flywheel direct transcription from N = 100 to N = 1000.\n");
  for (int N = 100; N <= 2000; N += 100) {
    scalability << N << ",";

    units::second_t dt = T / N;

    // Flywheel model:
    // States: [velocity]
    // Inputs: [voltage]
    auto system = frc::LinearSystemId::IdentifyVelocitySystem<units::radians>(
        1_V / 1_rad_per_s, 1_V / 1_rad_per_s_sq);
    frc::Matrixd<1, 1> A;
    frc::Matrixd<1, 1> B;
    frc::DiscretizeAB<1, 1>(system.A(), system.B(), dt, &A, &B);

    {
      fmt::print("CasADi (N = {})...", N);

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

      auto start = std::chrono::system_clock::now();
      opti.solve();
      auto end = std::chrono::system_clock::now();

      using std::chrono::duration_cast;
      using std::chrono::microseconds;
      double solveTime =
          duration_cast<microseconds>(end - start).count() / 1000.0;
      scalability << solveTime;
    }

    scalability << ",";

    {
      fmt::print("Problem (N = {})...", N);

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
      frc::Matrixd<1, 1> r{10.0};
      frc::VariableMatrix J = 0.0;
      for (int k = 0; k < N + 1; ++k) {
        J += ((r - X.Col(k)).Transpose() * (r - X.Col(k)));
      }
      problem.Minimize(J);

      auto start = std::chrono::system_clock::now();
      problem.Solve();
      auto end = std::chrono::system_clock::now();

      fmt::print(" done.\n");

      using std::chrono::duration_cast;
      using std::chrono::microseconds;
      double solveTime =
          duration_cast<microseconds>(end - start).count() / 1000.0;
      scalability << solveTime;
    }

    scalability << "\n";
  }
}
