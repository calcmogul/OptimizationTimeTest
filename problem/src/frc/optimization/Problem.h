// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#pragma once

#include <utility>

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <wpi/SymbolExports.h>

#include "Eigen/Core"
#include "frc/optimization/AutodiffWrapper.h"
#include "frc/optimization/EqualityConstraint.h"
#include "frc/optimization/InequalityConstraint.h"

namespace frc {

/**
 * Allows the user to pose a constrained nonlinear optimization problem in
 * natural mathematical notation and solve it.
 *
 * To motivate this class, we'll use the hypothetical problem of making a double
 * integrator (a system with position and velocity states and an acceleration
 * input) move from x=0 to x=10 in the minimum time with some velocity and
 * acceleration limits.
 *
 * This class supports problems of the
 * form:
 * @verbatim
 *      min_x f(x)
 * subject to b(x) ≥ 0
 *            c(x) = 0
 * @endverbatim
 *
 * where f(x) is the scalar cost function, x is the vector of decision variables
 * (variables the solver can tweak to minimize the cost function), b(x) are the
 * inequality constraints, and c(x) are the equality constraints. Constraints
 * are equations or inequalities of the decision variables that constrain what
 * values the solver is allowed to use when searching for an optimal solution.
 *
 * The nice thing about this class is users don't have to put their system in
 * the form shown above manually; they can write it in natural mathematical form
 * and it'll be converted for them.
 *
 * The model for our double integrator is ẍ=u where x is the vector [position;
 * velocity] and u is the scalar acceleration. We want to go from 0 m at rest to
 * 10 m at rest while obeying the velocity limit -1 ≤ x(1) ≤ 1 and the
 * acceleration limit -1 ≤ u ≤ 1.
 *
 * First, we need to make decision variables for our state and input.
 * @code{.cpp}
 * #include <frc/EigenCore.h>
 * #include <frc/optimization/Problem.h>
 * #include <units/time.h>
 *
 * constexpr auto T = 5_s;
 * constexpr auto dt = 5_ms;
 * constexpr int N = T / dt;
 *
 * frc::Problem problem;
 *
 * // 2x1 state vector with N + 1 timesteps (includes last state)
 * auto X = problem.Var<2, N + 1>();  // 2x1 state vector with N+1 timesteps
 *
 * // 1x1 input vector with N timesteps (input at last state doesn't matter)
 * auto U = problem.Var<1, N>();
 * @endcode
 * By convention, we use capital letters for the variables to designate
 * matrices.
 *
 * Now, we need to apply dynamics constraints between timesteps.
 * @code{.cpp}
 * // Kinematics constraint assuming constant acceleration between timesteps
 * for (int k = 0; k < N; ++k) {
 *   constexpr double t = dt.value();
 *   auto p_k1 = X(0, k + 1);
 *   auto v_k = X(1, k);
 *   auto a_k = U(0, k);
 *
 *   // pₖ₊₁ = 1/2aₖt² + vₖt
 *   problem.SubjectTo(p_k1 == 0.5 * pow(t, 2) * a_k + t * v_k);  // NOLINT
 * }
 * @endcode
 *
 * Next, we'll apply the state and input constraints.
 * @code{.cpp}
 * // Start and end at rest
 * problem.SubjectTo(X.Col(0) == frc::Matrixd<2, 1>{{0.0}, {0.0}});
 * problem.SubjectTo(X.Col(N + 1) == frc::Matrixd<2, 1>{{10.0}, {0.0}});

 * // Limit velocity
 * problem.SubjectTo(X.Row(1) >= -1);
 * problem.SubjectTo(X.Row(1) <= 1);

 * // Limit acceleration
 * problem.SubjectTo(U >= -1);
 * problem.SubjectTo(U <= 1);
 * @endcode
 *
 * Next, we'll create a cost function for minimizing position error.
 * @code{.cpp}
 * // Cost function - minimize position error
 * frc::Variable<1, 1> J = 0.0;
 * for (int k = 0; k < N + 1; ++k) {
 *   J += std::pow(10.0 - X(0, k), 2);
 * }
 * problem.Minimize(J);
 * @endcode
 * The cost function passed to Minimize() should produce a scalar output.
 *
 * Now we can solve the problem.
 * @code{.cpp}
 * problem.Solve();
 * @endcode
 *
 * The solver will find the decision variable values that minimize the cost
 * function while obeying the constraints. You can obtain the solution by
 * querying the values of the variables like so.
 * @code{.cpp}
 * double input = U.Value(0, 0);
 * @endcode
 *
 * In retrospect, the solution here seems obvious: if you want to reach the
 * desired position in minimal time, you just apply max input to move toward it,
 * then stop applying input once you get there. Problems can get more complex
 * than this though. In fact, we can use this same framework to design optimal
 * trajectories for a drivetrain while obeying dynamics constraints, avoiding
 * obstacles, and driving through points of interest.
 */
class WPILIB_DLLEXPORT Problem {
 public:
  Problem() = default;

  /**
   * Create a matrix of decision variables in the optimization problem.
   */
  template <int Rows = 1, int Cols = 1>
  Variable<Rows, Cols> Var() {
    Variable<Rows, Cols> vars;

    int oldSize = m_leaves.rows();
    m_leaves.resize(oldSize + Rows * Cols);
    for (int row = 0; row < Rows; ++row) {
      for (int col = 0; col < Cols; ++col) {
        vars.GetStorage()(row, col) =
            AutodiffWrapper{this, oldSize + row * Cols + col};
      }
    }

    return vars;
  }

  /**
   * Tells the solver to minimize the output of the given cost function.
   *
   * Note that this is optional. If only constraints are specified, the solver
   * will find the closest solution to the initial conditions that's in the
   * feasible set.
   *
   * @param cost The cost function to minimize.
   */
  void Minimize(const Variable<1, 1>& cost);

  /**
   * Tells the solver to minimize the output of the given cost function.
   *
   * Note that this is optional. If only constraints are specified, the solver
   * will find the closest solution to the initial conditions that's in the
   * feasible set.
   *
   * @param cost The cost function to minimize.
   */
  void Minimize(Variable<1, 1>&& cost);

  /**
   * Tells the solver to solve the problem while obeying the given equality
   * constraint.
   *
   * @param constraint The constraint to satisfy.
   */
  template <int Rows, int Cols, int... Args>
  void SubjectTo(
      Eigen::Matrix<EqualityConstraint, Rows, Cols, Args...>&& constraint) {
    int oldSize = m_equalityConstraints.rows();
    m_equalityConstraints.resize(oldSize + Rows * Cols);

    for (int row = 0; row < Rows; ++row) {
      for (int col = 0; col < Cols; ++col) {
        m_equalityConstraints(oldSize + row * Cols + col) =
            std::move(constraint(row, col).variable.GetAutodiff());
      }
    }
  }

  /**
   * Tells the solver to solve the problem while obeying the given inequality
   * constraint.
   *
   * @param constraint The constraint to satisfy.
   */
  template <int Rows, int Cols, int... Args>
  void SubjectTo(
      Eigen::Matrix<InequalityConstraint, Rows, Cols, Args...>&& constraint) {
    int oldSize = m_inequalityConstraints.rows();
    m_inequalityConstraints.resize(oldSize + Rows * Cols);

    for (int row = 0; row < Rows; ++row) {
      for (int col = 0; col < Cols; ++col) {
        m_inequalityConstraints(oldSize + row * Cols + col) =
            std::move(constraint(row, col).variable.GetAutodiff());
      }
    }
  }

  /**
   * Solve the optimization problem. The solution will be stored in the original
   * variables used to construct the problem.
   */
  void Solve();

 private:
  // Leaves of the problem's expression tree
  autodiff::VectorXvar m_leaves;

  // Cost function: f(x)
  autodiff::var m_f;

  // Inequality constraints: b(x) ≥ 0
  autodiff::VectorXvar m_inequalityConstraints;

  // Equality constraints: c(x) = 0
  autodiff::VectorXvar m_equalityConstraints;

  /**
   * Initialize leaves with the given vector.
   *
   * @param x The input vector.
   */
  void SetLeaves(const Eigen::Ref<const Eigen::VectorXd>& x);

  /**
   * The cost function f(x).
   *
   * @param x The input of f(x).
   */
  double f(const Eigen::Ref<const Eigen::VectorXd>& x);

  /**
   * Return the optimal step size alpha using backtracking line search.
   *
   * @param x The initial guess.
   * @param gradient The gradient at x.
   */
  double BacktrackingLineSearch(
      const Eigen::Ref<const Eigen::VectorXd>& x,
      const Eigen::Ref<const Eigen::VectorXd>& gradient);

  /**
   * Find the optimal solution using gradient descent.
   *
   * @param x The initial guess.
   */
  Eigen::VectorXd GradientDescent(const Eigen::Ref<const Eigen::VectorXd>& x);

  /**
  Find the optimal solution using a sequential quadratic programming solver.

  A sequential quadratic programming (SQP) problem has the form:

  @verbatim
       min_x f(x)
  @endverbatim

  where f(x) is the cost function.

  @param x The initial guess.
  */
  Eigen::VectorXd UnconstrainedSQP(const Eigen::Ref<const Eigen::VectorXd>& x);

  /**
  Find the optimal solution using a sequential quadratic programming solver.

  A sequential quadratic programming (SQP) problem has the form:

  @verbatim
       min_x f(x)
  subject to b(x) ≥ 0
             c(x) = 0
  @endverbatim

  where f(x) is the cost function, b(x) are the inequality constraints, and c(x)
  are the equality constraints.

  @param x The initial guess.
  */
  Eigen::VectorXd SQP(const Eigen::Ref<const Eigen::VectorXd>& x);

  friend class AutodiffWrapper;
};

}  // namespace frc
