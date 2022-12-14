// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "frc/optimization/OptimizationProblem.h"

#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <wpi/scope>

#include "Eigen/IterativeLinearSolvers"
#include "Eigen/SparseCore"
#include "SimplicialLDLTPreconditioner.h"
#include "frc/autodiff/Expression.h"
#include "frc/autodiff/Gradient.h"
#include "frc/autodiff/Hessian.h"
#include "frc/autodiff/Jacobian.h"
#include "frc/autodiff/Variable.h"
#include "units/time.h"

using namespace frc;

namespace {
/**
 * Assigns the contents of a double vector to an autodiff vector.
 *
 * @param dest The autodiff vector.
 * @param src The double vector.
 */
void SetAD(std::vector<autodiff::Variable>& dest,
           const Eigen::Ref<const Eigen::VectorXd>& src) {
  assert(dest.size() == static_cast<size_t>(src.rows()));

  for (size_t row = 0; row < dest.size(); ++row) {
    dest[row] = src(row);
  }
}

/**
 * Assigns the contents of a double vector to an autodiff vector.
 *
 * @param dest The autodiff vector.
 * @param src The double vector.
 */
void SetAD(Eigen::Ref<autodiff::VectorXvar> dest,
           const Eigen::Ref<const Eigen::VectorXd>& src) {
  assert(dest.rows() == src.rows());

  for (int row = 0; row < dest.rows(); ++row) {
    dest(row) = src(row);
  }
}

/**
 * Applies fraction-to-the-boundary rule to a variable and its iterate, then
 * returns a fraction of the iterate step size within (0, 1].
 *
 * @param x The variable.
 * @param p The iterate on the variable.
 * @param tau Fraction-to-the-boundary rule scaling factor.
 * @return Fraction of the iterate step size within (0, 1].
 */
double FractionToTheBoundaryRule(const Eigen::Ref<const Eigen::VectorXd>& x,
                                 const Eigen::Ref<const Eigen::VectorXd>& p,
                                 double tau) {
  // ?????????? = max(?? ??? (0, 1] : x + ??p ??? (1?????)x)
  double alpha = 1;
  for (int i = 0; i < x.rows(); ++i) {
    if (p(i) != 0.0) {
      while (alpha * p(i) < -tau * x(i)) {
        alpha *= 0.999;
      }
    }
  }

  return alpha;
}

/**
 * Adds a sparse matrix to the list of triplets with the given row and column
 * offset.
 *
 * @param[out] triplets The triplet storage.
 * @param[in] rowOffset The row offset for each triplet.
 * @param[in] colOffset The column offset for each triplet.
 * @param[in] mat The matrix to iterate over.
 * @param[in] transpose Whether to transpose mat.
 */
void AssignSparseBlock(std::vector<Eigen::Triplet<double>>& triplets,
                       int rowOffset, int colOffset,
                       const Eigen::SparseMatrix<double>& mat,
                       bool transpose = false) {
  for (int k = 0; k < mat.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it{mat, k}; it; ++it) {
      if (transpose) {
        triplets.emplace_back(rowOffset + it.col(), colOffset + it.row(),
                              it.value());
      } else {
        triplets.emplace_back(rowOffset + it.row(), colOffset + it.col(),
                              it.value());
      }
    }
  }
}

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
}  // namespace

OptimizationProblem::OptimizationProblem() noexcept {
  m_decisionVariables.reserve(1024);
  m_equalityConstraints.reserve(1024);
  m_inequalityConstraints.reserve(1024);
}

VariableMatrix OptimizationProblem::DecisionVariable(int rows, int cols) {
  VariableMatrix vars{rows, cols};
  int oldSize = m_decisionVariables.size();

  m_decisionVariables.reserve(m_decisionVariables.size() + rows * cols);

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      m_decisionVariables.emplace_back(0.0);
      vars.Autodiff(row, col) = m_decisionVariables[oldSize + row * cols + col];
    }
  }

  return vars;
}

void OptimizationProblem::Minimize(const VariableMatrix& cost) {
  assert(cost.Rows() == 1 && cost.Cols() == 1);
  m_f = cost.Autodiff(0, 0);
}

void OptimizationProblem::Minimize(VariableMatrix&& cost) {
  assert(cost.Rows() == 1 && cost.Cols() == 1);
  m_f = std::move(cost.Autodiff(0, 0));
}

void OptimizationProblem::Maximize(const VariableMatrix& objective) {
  assert(objective.Rows() == 1 && objective.Cols() == 1);

  // Maximizing an objective function is the same as minimizing its negative
  m_f = -objective.Autodiff(0, 0);
}

void OptimizationProblem::Maximize(VariableMatrix&& objective) {
  assert(objective.Rows() == 1 && objective.Cols() == 1);

  // Maximizing an objective function is the same as minimizing its negative
  m_f = -std::move(objective.Autodiff(0, 0));
}

void OptimizationProblem::SubjectTo(EqualityConstraints&& constraint) {
  auto& storage = constraint.constraints;

  m_equalityConstraints.reserve(m_equalityConstraints.size() + storage.size());

  for (size_t i = 0; i < storage.size(); ++i) {
    m_equalityConstraints.emplace_back(std::move(storage[i]));
  }
}

void OptimizationProblem::SubjectTo(InequalityConstraints&& constraint) {
  auto& storage = constraint.constraints;

  m_inequalityConstraints.reserve(m_inequalityConstraints.size() +
                                  storage.size());

  for (size_t i = 0; i < storage.size(); ++i) {
    m_inequalityConstraints.emplace_back(std::move(storage[i]));
  }
}

SolverStatus OptimizationProblem::Solve(const SolverConfig& config) {
  constexpr std::array<const char*, 5> kExprTypeToName = {
      "empty", "constant", "linear", "quadratic", "nonlinear"};

  m_config = config;

  // Create the initial value column vector
  Eigen::VectorXd x{m_decisionVariables.size(), 1};
  for (size_t i = 0; i < m_decisionVariables.size(); ++i) {
    x(i) = m_decisionVariables[i].Value();
  }

  SolverStatus status;

  // Get f's expression type
  if (m_f.has_value()) {
    status.costFunctionType = m_f.value().Type();
  }

  // Get the highest order equality constraint expression type
  for (const auto& constraint : m_equalityConstraints) {
    auto constraintType = constraint.Type();
    if (status.equalityConstraintType < constraintType) {
      status.equalityConstraintType = constraintType;
    }
  }

  // Get the highest order inequality constraint expression type
  for (const auto& constraint : m_inequalityConstraints) {
    auto constraintType = constraint.Type();
    if (status.inequalityConstraintType < constraintType) {
      status.inequalityConstraintType = constraintType;
    }
  }

  if (m_config.diagnostics) {
    fmt::print("The cost function is {}.\n",
               kExprTypeToName[static_cast<int>(status.costFunctionType)]);
    fmt::print(
        "The equality constraints are {}.\n",
        kExprTypeToName[static_cast<int>(status.equalityConstraintType)]);
    fmt::print(
        "The inequality constraints are {}.\n",
        kExprTypeToName[static_cast<int>(status.inequalityConstraintType)]);
    fmt::print("\n");
  }

  // If the problem is empty or constant, there's nothing to do
  if (status.costFunctionType <= autodiff::ExpressionType::kConstant &&
      status.equalityConstraintType <= autodiff::ExpressionType::kConstant &&
      status.inequalityConstraintType <= autodiff::ExpressionType::kConstant) {
    return status;
  }

  // If there's no cost function, make it zero and continue
  if (!m_f.has_value()) {
    m_f = 0.0;
  }

  // Solve the optimization problem
  Eigen::VectorXd solution = InteriorPoint(x, &status);

  if (m_config.diagnostics) {
    fmt::print("Exit condition: ");
    if (status.exitCondition == SolverExitCondition::kOk) {
      fmt::print("optimal solution found");
    } else if (status.exitCondition == SolverExitCondition::kTooFewDOFs) {
      fmt::print("problem has too few degrees of freedom");
    } else if (status.exitCondition ==
               SolverExitCondition::kLocallyInfeasible) {
      fmt::print("problem is locally infeasible");
    } else if (status.exitCondition == SolverExitCondition::kMaxIterations) {
      fmt::print("maximum iterations exceeded");
    } else if (status.exitCondition == SolverExitCondition::kTimeout) {
      fmt::print("solution returned after timeout");
    }
    fmt::print("\n");
  }

  // Assign the solution to the original Variable instances
  SetAD(m_decisionVariables, solution);

  return status;
}

Eigen::VectorXd OptimizationProblem::InteriorPoint(
    const Eigen::Ref<const Eigen::VectorXd>& initialGuess,
    SolverStatus* status) {
  // Let f(x)??? be the cost function, c???(x)??? be the equality constraints, and
  // c???(x)??? be the inequality constraints. The Lagrangian of the optimization
  // problem is
  //
  //   L(x, s, y, z)??? = f(x)??? ??? y??????c???(x)??? ??? z??????(c???(x)??? ??? s???)
  //
  // The Hessian of the Lagrangian is
  //
  //   H(x)??? = ???????????L(x, s, y, z)???
  //
  // The primal-dual barrier term Hessian ?? is defined as
  //
  //   ?? = S?????Z
  //
  // where
  //
  //       [s??? 0 ??? 0 ]
  //   S = [0  ???   ??? ]
  //       [???    ??? 0 ]
  //       [0  ??? 0 s???]
  //
  //       [z??? 0 ??? 0 ]
  //   Z = [0  ???   ??? ]
  //       [???    ??? 0 ]
  //       [0  ??? 0 z???]
  //
  // and where m is the number of inequality constraints.
  //
  // Let f(x) = f(x)???, H = H(x)???, A??? = A???(x)???, and A??? = A???(x)??? for clarity. We
  // want to solve the following Newton-KKT system shown in equation (19.12) of
  // [1].
  //
  //   [H    0  A??????  A??????][ p?????]    [???f(x) ??? A??????y ??? A??????z]
  //   [0    ??   0   ???I ][ p?????] = ???[     z ??? ??S?????e     ]
  //   [A???   0   0    0 ][???p?????]    [        c???         ]
  //   [A???  ???I   0    0 ][???p??????]    [      c??? ??? s       ]
  //
  // where e is a column vector of ones with a number of rows equal to the
  // number of inequality constraints.
  //
  // Solve the second row for p?????.
  //
  //   ??p????? + p?????? = ??S?????e ??? z
  //   ??p????? = ??S?????e ??? z ??? p??????
  //   p????? = ?????????S?????e ??? ???????z ??? ???????p??????
  //
  // Substitute ?? = S?????Z into the first two terms.
  //
  //   p????? = ??(S?????Z)?????S?????e ??? (S?????Z)?????z ??? ???????p??????
  //   p????? = ??Z?????SS?????e ??? Z?????Sz ??? ???????p??????
  //   p????? = ??Z?????e ??? s ??? ???????p??????
  //
  // Substitute the explicit formula for p????? into the fourth row and simplify.
  //
  //   A???p????? ??? p????? = s ??? c???
  //   A???p????? ??? (??Z?????e ??? s ??? ???????p??????) = s ??? c???
  //   A???p????? ??? ??Z?????e + s + ???????p?????? = s ??? c???
  //   A???p????? + ???????p?????? = ???c??? + ??Z?????e
  //
  // Substitute the new second and fourth rows into the system.
  //
  //   [H   0  A??????  A?????? ][ p?????]    [???f(x) ??? A??????y ??? A??????z]
  //   [0   I   0    0  ][ p?????] = ???[?????Z?????e + s + ???????p??????]
  //   [A???  0   0    0  ][???p?????]    [        c???         ]
  //   [A???  0   0   ??????????][???p??????]    [     c??? ??? ??Z?????e    ]
  //
  // Eliminate the second row and column.
  //
  //   [H   A??????   A??? ][ p?????]    [???f(x) ??? A??????y ??? A??????z]
  //   [A???   0    0  ][???p?????] = ???[        c???         ]
  //   [A???   0   ??????????][???p??????]    [    c??? ??? ??Z?????e     ]
  //
  // Solve the third row for p??????.
  //
  //   A???p????? + ???????p?????? = ???c??? + ??Z?????e
  //   p?????? = ?????c??? + ??S?????e ??? ??A???p?????
  //
  // Substitute the explicit formula for p?????? into the first row.
  //
  //   Hp????? ??? A??????p????? ??? A??????p?????? = ??????f(x) + A??????y + A??????z
  //   Hp????? ??? A??????p????? ??? A??????(?????c??? + ??S?????e ??? ??A???p?????) = ??????f(x) + A??????y + A??????z
  //
  // Expand and simplify.
  //
  //   Hp????? ??? A??????p????? + A????????c??? ??? A????????S?????e + A????????A???p????? = ??????f(x) + A??????y + A??????z
  //   Hp????? + A????????A???p????? ??? A??????p?????  = ??????f(x) + A??????y + A????????c??? + A????????S?????e + A??????z
  //   (H + A????????A???)p????? ??? A??????p????? = ??????f(x) + A??????y ??? A??????(??c??? ??? ??S?????e ??? z)
  //   (H + A????????A???)p????? ??? A??????p????? = ???(???f(x) ??? A??????y + A??????(??c??? ??? ??S?????e ??? z))
  //
  // Substitute the new first and third rows into the system.
  //
  //   [H + A????????A???   A??????  0][ p?????]    [???f(x) ??? A??????y + A??????(??c??? ??? ??S?????e ??? z)]
  //   [    A???        0   0][???p?????] = ???[                c???                 ]
  //   [    0         0   I][???p??????]    [       ?????c??? + ??S?????e ??? ??A???p?????       ]
  //
  // Eliminate the third row and column.
  //
  //   [H + A????????A???  A??????][ p?????] = ???[???f(x) ??? A??????y + A??????(??c??? ??? ??S?????e ??? z)]
  //   [    A???       0 ][???p?????]    [                c???                 ]
  //
  // This reduced 2x2 block system gives the iterates p????? and p????? with the
  // iterates p?????? and p????? given by
  //
  //   p?????? = ?????c??? + ??S?????e ??? ??A???p?????
  //   p????? = ??Z?????e ??? s ??? ???????p??????
  //
  // The iterates are applied like so
  //
  //   x????????? = x??? + ?????????????p?????
  //   s????????? = x??? + ?????????????p?????
  //   y????????? = x??? + ????????p?????
  //   z????????? = x??? + ????????p??????
  //
  // where ????????????? and ???????? are computed via the fraction-to-the-boundary rule
  // shown in equations (15a) and (15b) of [2].
  //
  //   ????????????? = max(?? ??? (0, 1] : s??? + ??p????? ??? (1????????)s???)
  //   ???????? = max(?? ??? (0, 1] : z??? + ??p?????? ??? (1????????)z???)
  //
  // [1] Nocedal, J. and Wright, S. "Numerical Optimization", 2nd. ed., Ch. 19.
  //     Springer, 2006.
  // [2] Wa??chter, A. and Biegler, L. "On the implementation of an interior-point
  //     filter line-search algorithm for large-scale nonlinear programming",
  //     2005. http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf
  // [3] Byrd, R. and Nocedal J. and Waltz R. "KNITRO: An Integrated Package for
  //     Nonlinear Optimization", 2005.
  //     https://users.iems.northwestern.edu/~nocedal/PDFfiles/integrated.pdf

  auto solveStartTime = std::chrono::system_clock::now();

  if (m_config.diagnostics) {
    fmt::print("Number of equality constraints: {}\n",
               m_equalityConstraints.size());
    fmt::print("Number of inequality constraints: {}\n\n",
               m_inequalityConstraints.size());
  }

  // Barrier parameter scale factor ??_?? for tolerance checks
  constexpr double kappa_epsilon = 10.0;

  // Barrier parameter scale factor ??_?? for inequality constraint Lagrange
  // multiplier safeguard
  constexpr double kappa_sigma = 1e10;

  // Fraction-to-the-boundary rule scale factor minimum
  constexpr double tau_min = 0.99;

  // Tuning parameters for ?? update
  constexpr double kappa_mu = 0.2;  // (0, 1)
  constexpr double theta_mu = 1.5;  // (1, 2)

  // Barrier parameter ??
  double mu = 0.1;

  // Fraction-to-the-boundary rule scale factor ??
  double tau = tau_min;

  std::vector<Eigen::Triplet<double>> triplets;

  Eigen::VectorXd x = initialGuess;
  autodiff::MapVectorXvar xAD(m_decisionVariables.data(),
                              m_decisionVariables.size());

  Eigen::VectorXd s = Eigen::VectorXd::Ones(m_inequalityConstraints.size());
  autodiff::VectorXvar sAD =
      autodiff::VectorXvar::Ones(m_inequalityConstraints.size());

  Eigen::VectorXd y = Eigen::VectorXd::Zero(m_equalityConstraints.size());
  autodiff::VectorXvar yAD =
      autodiff::VectorXvar::Zero(m_equalityConstraints.size());

  Eigen::VectorXd z = Eigen::VectorXd::Ones(m_inequalityConstraints.size());
  autodiff::VectorXvar zAD =
      autodiff::VectorXvar::Ones(m_inequalityConstraints.size());

  autodiff::MapVectorXvar c_eAD(m_equalityConstraints.data(),
                                m_equalityConstraints.size());
  autodiff::MapVectorXvar c_iAD(m_inequalityConstraints.data(),
                                m_inequalityConstraints.size());

  const Eigen::MatrixXd e = Eigen::VectorXd::Ones(s.rows());

  // L(x, s, y, z)??? = f(x)??? ??? y??????c???(x)??? ??? z??????(c???(x)??? ??? s???)
  autodiff::Variable L = m_f.value();
  if (m_equalityConstraints.size() > 0) {
    L -= yAD.transpose() * c_eAD;
  }
  if (m_inequalityConstraints.size() > 0) {
    L -= zAD.transpose() * (c_iAD - sAD);
  }

  Eigen::VectorXd step = Eigen::VectorXd::Zero(x.rows());

  SetAD(xAD, x);
  L.Update();

  autodiff::Gradient gradientF{m_f.value(), xAD};
  autodiff::Hessian hessianL{L, xAD};
  autodiff::Jacobian jacobianCe{c_eAD, xAD};
  autodiff::Jacobian jacobianCi{c_iAD, xAD};

  // Error estimate E_??
  double E_mu = std::numeric_limits<double>::infinity();

  // Gradient of f ???f
  Eigen::SparseVector<double> g = gradientF.Calculate();

  // Hessian of the Lagrangian H
  //
  // H??? = ???????????L(x, s, y, z)???
  Eigen::SparseMatrix<double> H = hessianL.Calculate();

  // Equality constraints c???
  Eigen::VectorXd c_e{m_equalityConstraints.size()};

  // Equality constraint Jacobian A???
  //
  //         [??????c??????(x)???]
  // A???(x) = [??????c??????(x)???]
  //         [    ???    ]
  //         [??????c??????(x)???]
  Eigen::SparseMatrix<double> A_e = jacobianCe.Calculate();

  // Inequality constraints c???
  Eigen::VectorXd c_i{m_inequalityConstraints.size()};

  // Inequality constraint Jacobian A???
  //
  //         [??????c??????(x)???]
  // A???(x) = [??????c??????(x)???]
  //         [    ???    ]
  //         [??????c??????(x)???]
  Eigen::SparseMatrix<double> A_i = jacobianCi.Calculate();

  auto iterationsStartTime = std::chrono::system_clock::now();

  if (m_config.diagnostics) {
    // Print number of nonzeros in Lagrangian Hessian and constraint Jacobians
    std::string prints;

    if (status->costFunctionType <= autodiff::ExpressionType::kQuadratic &&
        status->equalityConstraintType <=
            autodiff::ExpressionType::kQuadratic &&
        status->inequalityConstraintType <=
            autodiff::ExpressionType::kQuadratic) {
      prints += fmt::format("Number of nonzeros in Lagrangian Hessian: {}\n",
                            H.nonZeros());
    }
    if (status->equalityConstraintType <= autodiff::ExpressionType::kLinear) {
      prints += fmt::format(
          "Number of nonzeros in equality constraint Jacobian: {}\n",
          A_e.nonZeros());
    }
    if (status->inequalityConstraintType <= autodiff::ExpressionType::kLinear) {
      prints += fmt::format(
          "Number of nonzeros in inequality constraint Jacobian: {}\n",
          A_i.nonZeros());
    }

    if (prints.length() > 0) {
      fmt::print("{}\n", prints);
    }

    fmt::print("Error tolerance: {}\n\n", m_config.tolerance);
  }

  int iterations = 0;

  wpi::scope_exit exit{[&] {
    if (m_config.diagnostics) {
      auto solveEndTime = std::chrono::system_clock::now();

      fmt::print("\nSolve time: {} ms\n",
                 ToMilliseconds(solveEndTime - solveStartTime));
      fmt::print("  ??? {} ms (IPM setup)\n",
                 ToMilliseconds(iterationsStartTime - solveStartTime));
      if (iterations > 0) {
        fmt::print(
            "  ??? {} ms ({} IPM iterations; {} ms average)\n",
            ToMilliseconds(solveEndTime - iterationsStartTime), iterations,
            ToMilliseconds((solveEndTime - iterationsStartTime) / iterations));
      }
      fmt::print("\n");

      fmt::print("autodiff  setup (ms)  avg solve (ms)  solves\n");
      fmt::print("============================================\n");
      fmt::print("   ???f(x)  {:>10}  {:>14}  {:>6}\n",
                 gradientF.GetProfiler().SetupDuration(),
                 gradientF.GetProfiler().AverageSolveDuration(),
                 gradientF.GetProfiler().SolveMeasurements());
      fmt::print("   ???????????L  {:>10}  {:>14}  {:>6}\n",
                 hessianL.GetProfiler().SetupDuration(),
                 hessianL.GetProfiler().AverageSolveDuration(),
                 hessianL.GetProfiler().SolveMeasurements());
      fmt::print("  ???c???/???x  {:>10}  {:>14}  {:>6}\n",
                 jacobianCe.GetProfiler().SetupDuration(),
                 jacobianCe.GetProfiler().AverageSolveDuration(),
                 jacobianCe.GetProfiler().SolveMeasurements());
      fmt::print("  ???c???/???x  {:>10}  {:>14}  {:>6}\n",
                 jacobianCi.GetProfiler().SetupDuration(),
                 jacobianCi.GetProfiler().AverageSolveDuration(),
                 jacobianCi.GetProfiler().SolveMeasurements());
      fmt::print("\n");
    }
  }};

  while (E_mu > m_config.tolerance) {
    while (E_mu > kappa_epsilon * mu) {
      auto innerIterStartTime = std::chrono::system_clock::now();

      //     [s??? 0 ??? 0 ]
      // S = [0  ???   ??? ]
      //     [???    ??? 0 ]
      //     [0  ??? 0 s???]
      triplets.clear();
      for (int k = 0; k < s.rows(); k++) {
        triplets.emplace_back(k, k, s[k]);
      }
      Eigen::SparseMatrix<double> S{s.rows(), s.rows()};
      S.setFromTriplets(triplets.begin(), triplets.end());

      //     [z??? 0 ??? 0 ]
      // Z = [0  ???   ??? ]
      //     [???    ??? 0 ]
      //     [0  ??? 0 z???]
      triplets.clear();
      for (int k = 0; k < s.rows(); k++) {
        triplets.emplace_back(k, k, z[k]);
      }
      Eigen::SparseMatrix<double> Z{z.rows(), z.rows()};
      Z.setFromTriplets(triplets.begin(), triplets.end());

      // S?????
      Eigen::SparseMatrix<double> inverseS = S.cwiseInverse();

      // Z?????
      Eigen::SparseMatrix<double> inverseZ = Z.cwiseInverse();

      // ?? = S?????Z
      Eigen::SparseMatrix<double> sigma = inverseS * Z;

      // ??????? = SZ?????
      Eigen::SparseMatrix<double> inverseSigma = S * inverseZ;

      // H??? = ???????????L(x, s, y, z)???
      H = hessianL.Calculate();

      //         [??????c??????(x)???]
      // A???(x) = [??????c??????(x)???]
      //         [    ???    ]
      //         [??????c??????(x)???]
      A_e = jacobianCe.Calculate();

      //         [??????c??????(x)???]
      // A???(x) = [??????c??????(x)???]
      //         [    ???    ]
      //         [??????c??????(x)???]
      A_i = jacobianCi.Calculate();

      // Update c??? and c???
      for (size_t row = 0; row < m_equalityConstraints.size(); row++) {
        c_e[row] = m_equalityConstraints[row].Value();
      }
      for (size_t row = 0; row < m_inequalityConstraints.size(); row++) {
        c_i[row] = m_inequalityConstraints[row].Value();
      }

      // lhs = [H + A????????A???  A??????]
      //       [    A???       0 ]
      triplets.clear();
      Eigen::SparseMatrix<double> tmp = H;
      if (m_inequalityConstraints.size() > 0) {
        tmp += A_i.transpose() * sigma * A_i;
      }
      // Assign top-left quadrant
      AssignSparseBlock(triplets, 0, 0, tmp);
      if (m_equalityConstraints.size() > 0) {
        // Assign bottom-left quadrant
        AssignSparseBlock(triplets, tmp.rows(), 0, A_e);

        // Assign top-right quadrant
        AssignSparseBlock(triplets, 0, tmp.rows(), A_e, true);
      }
      Eigen::SparseMatrix<double> lhs{H.rows() + A_e.rows(),
                                      H.cols() + A_e.rows()};
      lhs.setFromTriplets(triplets.begin(), triplets.end());

      g = gradientF.Calculate();

      // rhs = ???[???f ??? A??????y + A??????(??c??? ??? ??S?????e ??? z)]
      //        [               c???               ]
      //
      // The outer negative sign is applied in the solve() call.
      Eigen::VectorXd rhs{x.rows() + y.rows()};
      rhs.topRows(x.rows()) = g;
      if (m_equalityConstraints.size() > 0) {
        rhs.topRows(x.rows()) -= A_e.transpose() * y;
      }
      if (m_inequalityConstraints.size() > 0) {
        rhs.topRows(x.rows()) +=
            A_i.transpose() * (sigma * c_i - mu * inverseS * e - z);
      }
      rhs.bottomRows(y.rows()) = c_e;

      // Regularize lhs by adding a multiple of the identity matrix
      //
      // lhs = [H + A????????A??? + ??I   A??????]
      //       [       A???        ?????I ]
      Eigen::SparseMatrix<double> regularization{lhs.rows(), lhs.cols()};
      triplets.clear();
      for (int row = 0; row < H.rows(); ++row) {
        triplets.emplace_back(row, row, 1e-9);
      }
      for (int row = H.rows(); row < lhs.rows(); ++row) {
        triplets.emplace_back(row, row, -1e-9);
      }
      regularization.setFromTriplets(triplets.begin(), triplets.end());
      lhs += regularization;

      // Solve the Newton-KKT system
      //
      // https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
      Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
                               Eigen::Lower | Eigen::Upper,
                               SimplicialLDLTPreconditioner<double>>
          solver{lhs};
      step = solver.solve(-rhs);

      // step = [ p?????]
      //        [???p?????]
      Eigen::VectorXd p_x = step.segment(0, x.rows());
      Eigen::VectorXd p_y = -step.segment(x.rows(), y.rows());

      // p?????? = ?????c??? + ??S?????e ??? ??A???p?????
      Eigen::VectorXd p_z =
          -sigma * c_i + mu * inverseS * e - sigma * A_i * p_x;

      // p????? = ??Z?????e ??? s ??? ???????p??????
      Eigen::VectorXd p_s = mu * inverseZ * e - s - inverseSigma * p_z;

      // ????????????? = max(?? ??? (0, 1] : s??? + ??p????? ??? (1????????)s???)
      double alpha_max = FractionToTheBoundaryRule(s, p_s, tau);

      // ???????? = max(?? ??? (0, 1] : z??? + ??p?????? ??? (1????????)z???)
      double alpha_z = FractionToTheBoundaryRule(z, p_z, tau);

      // x????????? = x??? + ?????????????p?????
      // s????????? = x??? + ?????????????p?????
      // y????????? = x??? + ????????p?????
      // z????????? = x??? + ????????p??????
      x += alpha_max * p_x;
      s += alpha_max * p_s;
      y += alpha_z * p_y;
      z += alpha_z * p_z;

      // A requirement for the convergence proof is that the "primal-dual
      // barrier term Hessian" ????? does not deviate arbitrarily much from the
      // "primal Hessian" ?????S????????. We ensure this by resetting
      //
      //   z?????????????????? = max(min(z??????????????????, ??_?? ?????/s??????????????????), ?????/(??_?? s??????????????????))
      //
      // for some fixed ??_?? ??? 1 after each step. See equation (16) in [2].
      for (int row = 0; row < z.rows(); ++row) {
        z(row) = std::max(std::min(z(row), kappa_sigma * mu / s(row)),
                          mu / (kappa_sigma * s(row)));
      }

      SetAD(xAD, x);
      SetAD(sAD, s);
      SetAD(yAD, y);
      SetAD(zAD, z);
      L.Update();

      // s_d = max(s?????????, (||y||??? + ||z||???) / (m + n)) / s?????????
      constexpr double s_max = 100.0;
      double s_d = std::max(s_max, (y.lpNorm<1>() + z.lpNorm<1>()) /
                                       (m_equalityConstraints.size() +
                                        m_inequalityConstraints.size())) /
                   s_max;

      // s_c = max(s?????????, ||z||??? / n) / s?????????
      double s_c =
          std::max(s_max, z.lpNorm<1>() / m_inequalityConstraints.size()) /
          s_max;

      // Update variables needed in error estimate
      g = gradientF.Calculate();
      A_e = jacobianCe.Calculate();
      A_i = jacobianCi.Calculate();
      for (size_t row = 0; row < m_equalityConstraints.size(); row++) {
        c_e[row] = m_equalityConstraints[row].Value();
      }
      for (size_t row = 0; row < m_inequalityConstraints.size(); row++) {
        c_i[row] = m_inequalityConstraints[row].Value();
      }
      triplets.clear();
      for (int k = 0; k < s.rows(); k++) {
        triplets.emplace_back(k, k, s[k]);
      }
      S.setFromTriplets(triplets.begin(), triplets.end());

      // Check for overconstrained problem
      if (m_equalityConstraints.size() > m_decisionVariables.size()) {
        fmt::print("The problem has too few degrees of freedom.\n");
        fmt::print(
            "Violated constraints (c???(x) = 0) in order of declaration:\n");
        for (int row = 0; row < c_e.rows(); ++row) {
          if (c_e(row) < 0.0) {
            fmt::print("  {}/{}: {} = 0\n", row + 1, c_e.rows(), c_e(row));
          }
        }

        status->exitCondition = SolverExitCondition::kTooFewDOFs;
        return x;
      }

      // Check for problem local infeasibility. The problem is locally
      // infeasible if
      //
      //   A??????c??? ??? 0
      //   A??????c?????? ??? 0
      //   ||(c???, c??????)|| > ??
      //
      // where c?????? = min(c???, 0).
      //
      // See "Infeasibility detection" in section 6 of [3].
      //
      // c?????? is used instead of c?????? from the paper to follow the convention that
      // feasible inequality constraints are ??? 0.
      if (m_equalityConstraints.size() > 0 &&
          (A_e.transpose() * c_e).norm() < 1e-6 && c_e.norm() > 1e-2) {
        if (m_config.diagnostics) {
          fmt::print(
              "The problem is locally infeasible due to violated equality "
              "constraints.\n");
          fmt::print(
              "Violated constraints (c???(x) = 0) in order of declaration:\n");
          for (int row = 0; row < c_e.rows(); ++row) {
            if (c_e(row) < 0.0) {
              fmt::print("  {}/{}: {} = 0\n", row + 1, c_e.rows(), c_e(row));
            }
          }
        }

        status->exitCondition = SolverExitCondition::kLocallyInfeasible;
        return x;
      }
      if (m_inequalityConstraints.size() > 0) {
        Eigen::VectorXd c_i_plus = c_i.cwiseMin(0.0);
        if ((A_i.transpose() * c_i_plus).norm() < 1e-6 &&
            c_i_plus.norm() > 1e-6) {
          if (m_config.diagnostics) {
            fmt::print(
                "The problem is infeasible due to violated inequality "
                "constraints.\n");
            fmt::print(
                "Violated constraints (c???(x) ??? 0) in order of declaration:\n");
            for (int row = 0; row < c_i.rows(); ++row) {
              if (c_i(row) < 0.0) {
                fmt::print("  {}/{}: {} ??? 0\n", row + 1, c_i.rows(), c_i(row));
              }
            }
          }

          status->exitCondition = SolverExitCondition::kLocallyInfeasible;
          return x;
        }
      }

      // Update the error estimate using the KKT conditions from equations
      // (19.5a) through (19.5d) in [1].
      //
      //   ???f ??? A??????y ??? A??????z = 0
      //   Sz ??? ??e = 0
      //   c??? = 0
      //   c??? ??? s = 0
      //
      // The error tolerance is the max of the following infinity norms scaled
      // by s_d and s_c (see equation (5) in [2]).
      //
      //   ||???f ??? A??????y ??? A??????z||_??? / s_d
      //   ||Sz ??? ??e||_??? / s_c
      //   ||c???||_???
      //   ||c??? ??? s||_???
      Eigen::VectorXd eq1 = g;
      if (m_equalityConstraints.size() > 0) {
        eq1 -= A_e.transpose() * y;
      }
      if (m_inequalityConstraints.size() > 0) {
        eq1 -= A_i.transpose() * z;
      }
      E_mu = std::max(eq1.lpNorm<Eigen::Infinity>() / s_d,
                      (S * z - mu * e).lpNorm<Eigen::Infinity>() / s_c);
      if (m_equalityConstraints.size() > 0) {
        E_mu = std::max(E_mu, c_e.lpNorm<Eigen::Infinity>());
      }
      if (m_inequalityConstraints.size() > 0) {
        E_mu = std::max(E_mu, (c_i - s).lpNorm<Eigen::Infinity>());
      }

      auto innerIterEndTime = std::chrono::system_clock::now();

      if (m_config.diagnostics) {
        if (iterations % 20 == 0) {
          fmt::print("iter  duration (ms)    error\n");
          fmt::print("==============================\n");
        }
        fmt::print("{:>4}  {:>10}     {:>9.3e}\n", iterations,
                   ToMilliseconds(innerIterEndTime - innerIterStartTime), E_mu);
      }

      ++iterations;
      if (iterations >= m_config.maxIterations) {
        status->exitCondition = SolverExitCondition::kMaxIterations;
        return x;
      }

      if (units::second_t{innerIterEndTime - solveStartTime} >
          m_config.timeout) {
        status->exitCondition = SolverExitCondition::kTimeout;
        return x;
      }
    }

    // Update the barrier parameter.
    //
    //   ??????????? = max(???????????/10, min(??_?? ?????, ?????^??_??))
    //
    // See equation (7) in [2].
    mu = std::max(m_config.tolerance / 10.0,
                  std::min(kappa_mu * mu, std::pow(mu, theta_mu)));

    // Update the fraction-to-the-boundary rule scaling factor.
    //
    //   ????? = max(???????????, 1 ??? ?????)
    //
    // See equation (8) in [2].
    tau = std::max(tau_min, 1.0 - mu);
  }

  return x;
}
