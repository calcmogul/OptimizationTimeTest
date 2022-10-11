// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <vector>

#include <fmt/core.h>

#include "CasADi.h"
#include "OptimizationProblem.h"

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

template <typename Problem>
void RunTest(std::ofstream& results, std::function<Problem()> setup,
             std::function<void(Problem&)> solve) {
  // Record setup time
  auto setupStartTime = std::chrono::system_clock::now();
  auto problem = setup();
  auto setupEndTime = std::chrono::system_clock::now();

  results << ToMilliseconds(setupEndTime - setupStartTime);
  std::flush(results);

  results << ",";
  std::flush(results);

  // Record solve time
  auto solveStartTime = std::chrono::system_clock::now();
  solve(problem);
  auto solveEndTime = std::chrono::system_clock::now();

  results << ToMilliseconds(solveEndTime - solveStartTime);
  std::flush(results);
}

int main() {
  constexpr auto T = 5_s;
  constexpr int kMaxPower = 3;

  std::ofstream results{"results.csv"};
  if (!results.is_open()) {
    return 1;
  }

  results << "Samples,"
          << "CasADi setup time (ms),CasADi solve time (ms),"
          << "Problem setup time (ms),Problem solve time (ms)\n";
  std::flush(results);

  std::vector<int> Ns;
  for (int power = 2; power < kMaxPower; ++power) {
    for (int N = std::pow(10, power); N < std::pow(10, power + 1);
         N += std::pow(10, power)) {
      Ns.emplace_back(N);
    }
  }
  Ns.emplace_back(std::pow(10, kMaxPower));

  fmt::print("Solving from N = {} to N = {}.\n", Ns.front(), Ns.back());
  for (int N : Ns) {
    results << N << ",";
    std::flush(results);

    units::second_t dt = T / N;

    fmt::print(stderr, "CasADi (N = {})...", N);
    RunTest<casadi::Opti>(
        results, [=] { return CartPoleCasADi(dt, N); },
        [](casadi::Opti& opti) { opti.solve(); });
    fmt::print(stderr, " done.\n");

    results << ",";
    std::flush(results);

    fmt::print(stderr, "Problem (N = {})...", N);
    RunTest<frc::OptimizationProblem>(
        results, [=] { return CartPoleOptimizationProblem(dt, N); },
        [](frc::OptimizationProblem& problem) { problem.Solve(); });
    fmt::print(stderr, " done.\n");

    results << "\n";
    std::flush(results);
  }
}
