// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include <chrono>
#include <fstream>
#include <vector>

#include <casadi/casadi.hpp>
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

/**
 * Converts std::chrono::duration to a number of milliseconds rounded to three
 * decimals.
 */
template <typename Rep, typename Period = std::ratio<1>>
double ToMilliseconds(const std::chrono::duration<Rep, Period>& duration) {
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  return duration_cast<microseconds>(duration).count() / 1000.0;
}

int main() {
  constexpr auto T = 5_s;

  std::ofstream results{"results.csv"};
  if (!results.is_open()) {
    return 1;
  }

  results << "Flywheel samples,"
          << "CasADi setup time (ms),CasADi solve time (ms),"
          << "Problem setup time (ms),Problem solve time (ms)\n";

  std::vector<int> Ns;
  for (int N = 1; N < 10; ++N) {
    Ns.emplace_back(N);
  }
  for (int N = 10; N < 100; N += 10) {
    Ns.emplace_back(N);
  }
  for (int N = 100; N < 4000; N += 100) {
    Ns.emplace_back(N);
  }
  Ns.emplace_back(4000);

  fmt::print("Solving flywheel direct transcription from N = {} to N = {}.\n",
             Ns.front(), Ns.back());
  for (int N : Ns) {
    results << N << ",";

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
      fmt::print(stderr, "CasADi (N = {})...", N);

      auto setupStartTime = std::chrono::system_clock::now();

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

      auto setupEndTime = std::chrono::system_clock::now();

      auto solveStartTime = std::chrono::system_clock::now();
      opti.solve();
      auto solveEndTime = std::chrono::system_clock::now();

      results << ToMilliseconds(setupEndTime - setupStartTime) << ","
              << ToMilliseconds(solveEndTime - solveStartTime);

      fmt::print(stderr, " done.\n");
    }

    results << ",";

    {
      fmt::print(stderr, "Problem (N = {})...", N);

      auto setupStartTime = std::chrono::system_clock::now();

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

      auto setupEndTime = std::chrono::system_clock::now();

      auto solveStartTime = std::chrono::system_clock::now();
      problem.Solve();
      auto solveEndTime = std::chrono::system_clock::now();

      results << ToMilliseconds(setupEndTime - setupStartTime) << ","
              << ToMilliseconds(solveEndTime - solveStartTime);

      fmt::print(stderr, " done.\n");
    }

    results << "\n";
  }
}
