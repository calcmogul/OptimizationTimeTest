// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "frc/optimization/Problem.h"

#include <vector>

#include "Eigen/SparseCore"
#include "Eigen/SparseQR"

using namespace frc;

void Problem::Minimize(const Variable<1, 1>& cost) {
  m_f = cost.GetStorage()(0, 0).GetAutodiff();
}

void Problem::Minimize(Variable<1, 1>&& cost) {
  m_f = std::move(cost.GetStorage()(0, 0).GetAutodiff());
}

void Problem::Solve() {
  // If there's no cost function or constraints, do nothing
  if (m_f == 0.0 && m_inequalityConstraints.rows() == 0 &&
      m_equalityConstraints.rows() == 0) {
    return;
  }

  // Create the initial value column vector
  Eigen::VectorXd x{m_leaves.size(), 1};
  for (int i = 0; i < m_leaves.rows(); ++i) {
    x(i) = val(m_leaves(i));
  }

  // Solve the optimization problem
  Eigen::VectorXd solution;
  if (m_inequalityConstraints.rows() == 0 &&
      m_equalityConstraints.rows() == 0) {
    // TODO: Implement faster unconstrained solver than gradient descent. SQP
    // beats gradient descent at the moment.
    solution = UnconstrainedSQP(x);
  } else {
    solution = SQP(x);
  }

  // Assign solution to the original AutodiffWrapper instances
  SetLeaves(solution);
}

void Problem::SetLeaves(const Eigen::Ref<const Eigen::VectorXd>& x) {
  for (int i = 0; i < m_leaves.rows(); ++i) {
    m_leaves(i).update(x(i));
  }
}

double Problem::f(const Eigen::Ref<const Eigen::VectorXd>& x) {
  SetLeaves(x);
  return val(m_f);
}

double Problem::BacktrackingLineSearch(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& gradient) {
  // [1] https://en.wikipedia.org/wiki/Backtracking_line_search#Algorithm

  double m = gradient.dot(gradient);  // gradiental derivative

  constexpr double c = 0.1;    // [0, 1]
  constexpr double tau = 0.9;  // [0, 1]
  double alpha = 0.01;         // > 0

  double t = -c * m;
  double f_x = f(x);
  while (f_x - f(x + alpha * gradient) < alpha * t) {
    alpha *= tau;
  }

  // Perform gradient descent with the step size alpha
  return alpha;
}

Eigen::VectorXd Problem::GradientDescent(
    const Eigen::Ref<const Eigen::VectorXd>& x) {
  constexpr double kConvergenceTolerance = 1E-4;

  Eigen::VectorXd lastX = x;
  Eigen::VectorXd currentX = x;
  while (true) {
    SetLeaves(lastX);
    m_f.update();
    Eigen::VectorXd g = gradient(m_f, m_leaves);

    currentX = lastX - g * BacktrackingLineSearch(lastX, g);

    if ((currentX - lastX).norm() < kConvergenceTolerance) {
      return currentX;
    }
    lastX = currentX;
  }
}

Eigen::VectorXd Problem::UnconstrainedSQP(
    const Eigen::Ref<const Eigen::VectorXd>& x) {
  // The equality-constrained quadratic programming problem is defined as
  //
  //        min f(x)
  //         x
  //
  // The Lagrangian for this problem is
  //
  // L(x, λ) = f(x)
  //
  // The first-order KKT conditions of the problem are
  //
  // F(x, λ) = ∇f(x) = 0
  //
  // The Jacobian of the KKT conditions with respect to x is given by
  //
  // F'(x, λ) = ∇²ₓₓL(x, λ)
  //
  // Let H(x) = ∇²ₓₓL(x, λ).
  //
  // F'(x, λ) = H(x)
  //
  // The Newton step from the iterate xₖ is given by
  //
  // xₖ₊₁ = xₖ + p_k
  //
  // where p_k solves the Newton-KKT system
  //
  // H(x)ₖpₖ = −∇f(x)ₖ
  //
  // [1] Nocedal, J. and Wright, S. Numerical Optimization, 2nd. ed., Ch. 18.
  //     Springer, 2006.

  using namespace autodiff;

  constexpr double kConvergenceTolerance = 1E-4;

  // L(x, λ)ₖ = f(x)ₖ
  autodiff::var L = m_f;

  Eigen::VectorXd lastX = x;
  Eigen::VectorXd currentX = x;
  std::vector<Eigen::Triplet<double>> triplets;
  while (true) {
    SetLeaves(lastX);
    L.update();

    // Hₖ = ∇²ₓₓL(x, λ)ₖ
    Eigen::MatrixXd H = hessian(L, m_leaves);

    // F'(x, λ) = H(x)ₖ
    Eigen::SparseMatrix<double, Eigen::RowMajor> Fprime{H.rows(), H.cols()};
    Fprime = H.sparseView();

    // rhs = −∇f(x)ₖ
    triplets.clear();
    // Add triplets for cost function gradient
    Eigen::VectorXd g = -gradient(m_f, m_leaves);
    for (int row = 0; row < g.rows(); ++row) {
      if (g(row) != 0.0) {
        triplets.emplace_back(row, 0, g(row));
      }
    }
    // Construct sparse vector
    Eigen::SparseMatrix<double> rhs{x.rows(), 1};
    rhs.setFromTriplets(triplets.begin(), triplets.end());

    // Solve the Newton-KKT system
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::AMDOrdering<int>>
        solver;
    solver.compute(Fprime);
    Eigen::VectorXd step = solver.solve(rhs);

    currentX = lastX + step;

    if ((currentX - lastX).norm() < kConvergenceTolerance) {
      return currentX;
    }
    lastX = currentX;
  }
}

Eigen::VectorXd Problem::SQP(const Eigen::Ref<const Eigen::VectorXd>& x) {
  // The equality-constrained quadratic programming problem is defined as
  //
  //        min f(x)
  //         x
  // subject to c(x) = 0
  //
  // The Lagrangian for this problem is
  //
  // L(x, λ) = f(x) − λᵀc(x)
  //
  // The Jacobian of the equality constraints is
  //
  //         [∇ᵀc₁(x)ₖ]
  // A(x)ₖ = [∇ᵀc₂(x)ₖ]
  //         [   ⋮    ]
  //         [∇ᵀcₘ(x)ₖ]
  //
  // The first-order KKT conditions of the equality-constrained problem are
  //
  // F(x, λ) = [∇f(x) − A(x)ᵀλ] = 0
  //           [     c(x)     ]
  //
  // The Jacobian of the KKT conditions with respect to x and λ is given by
  //
  // F'(x, λ) = [∇²ₓₓL(x, λ)  −A(x)ᵀ]
  //            [   A(x)        0   ]
  //
  // Let H(x) = ∇²ₓₓL(x, λ).
  //
  // F'(x, λ) = [H(x)  −A(x)ᵀ]
  //            [A(x)    0   ]
  //
  // The Newton step from the iterate (xₖ, λₖ) is given by
  //
  // [xₖ₊₁] = [xₖ] + [p_k]
  // [λₖ₊₁]   [λₖ]   [p_λ]
  //
  // where p_k and p_λ solve the Newton-KKT system
  //
  // [H(x)ₖ  −A(x)ₖᵀ][pₖ ] = [−∇f(x)ₖ + A(x)ₖᵀλₖ]
  // [A(x)ₖ     0   ][p_λ]   [      −c(x)ₖ      ]
  //
  // Subtracting A(x)ₖᵀλₖ from both sides of the first equation, we get
  //
  // [H(x)ₖ  −A(x)ₖᵀ][pₖ  ] = [−∇f(x)ₖ]
  // [A(x)ₖ     0   ][λₖ₊₁]   [ −c(x)ₖ]
  //
  // [1] Nocedal, J. and Wright, S. Numerical Optimization, 2nd. ed., Ch. 18.
  //     Springer, 2006.

  using namespace autodiff;

  constexpr double kConvergenceTolerance = 1E-4;

  // autodiff vector of the equality constraints c(x)
  auto& c = m_equalityConstraints;

  // Lagrange multipliers for the equality constraints
  autodiff::VectorXvar lambda{m_equalityConstraints.size(), 1};

  // L(x, λ)ₖ = f(x)ₖ − λₖᵀc(x)ₖ
  autodiff::var L = m_f - (lambda.transpose() * c)(0);

  Eigen::VectorXd lastX = x;
  Eigen::VectorXd currentX = x;
  std::vector<Eigen::Triplet<double>> triplets;
  while (true) {
    SetLeaves(lastX);
    L.update();

    // Hₖ = ∇²ₓₓL(x, λ)ₖ
    Eigen::MatrixXd H = hessian(L, m_leaves);

    //         [∇ᵀc₁(x)ₖ]
    // A(x)ₖ = [∇ᵀc₂(x)ₖ]
    //         [   ⋮    ]
    //         [∇ᵀcₘ(x)ₖ]
    triplets.clear();
    for (int row = 0; row < m_equalityConstraints.size(); ++row) {
      Eigen::RowVectorXd g = gradient(c(row), m_leaves).transpose();
      for (int col = 0; col < g.cols(); ++col) {
        if (g(col) != 0.0) {
          triplets.emplace_back(row, col, g(col));
        }
      }
    }
    Eigen::SparseMatrix<double> A{m_equalityConstraints.size(), x.rows()};
    A.setFromTriplets(triplets.begin(), triplets.end());

    // F'(x, λ) = [H(x)ₖ  −A(x)ₖᵀ]
    //            [A(x)ₖ     0   ]
    Eigen::SparseMatrix<double> FprimeUpper{H.rows(), H.cols() + A.rows()};
    FprimeUpper.leftCols(H.cols()) = H.sparseView();
    FprimeUpper.rightCols(A.rows()) = -A.transpose();

    Eigen::SparseMatrix<double> FprimeLower{A.rows(), H.cols() + A.rows()};
    FprimeLower.leftCols(H.cols()) = A;

    Eigen::SparseMatrix<double, Eigen::RowMajor> Fprime{H.rows() + A.rows(),
                                                        H.cols() + A.rows()};
    Fprime.topRows(H.rows()) = FprimeUpper;
    Fprime.bottomRows(A.rows()) = FprimeLower;

    // rhs = [−∇f(x)ₖ]
    //       [ −c(x)ₖ]
    triplets.clear();
    // Add triplets for top block
    Eigen::VectorXd g = -gradient(m_f, m_leaves);
    for (int row = 0; row < g.rows(); ++row) {
      if (g(row) != 0.0) {
        triplets.emplace_back(row, 0, g(row));
      }
    }
    // Add triplets for bottom block
    for (int row = 0; row < c.rows(); ++row) {
      double value = -val(c(row));
      if (value != 0.0) {
        triplets.emplace_back(x.rows() + row, 0, value);
      }
    }
    // Construct sparse vector
    Eigen::SparseMatrix<double> rhs{x.rows() + A.rows(), 1};
    rhs.setFromTriplets(triplets.begin(), triplets.end());

    // Solve the Newton-KKT system
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::AMDOrdering<int>>
        solver;
    solver.compute(Fprime);
    Eigen::VectorXd step = solver.solve(rhs);

    currentX = lastX + step.block(0, 0, x.rows(), 1);
    lambda = step.block(x.rows(), 0, lambda.rows(), 1);

    if ((currentX - lastX).norm() < kConvergenceTolerance) {
      return currentX;
    }
    lastX = currentX;
  }
}
