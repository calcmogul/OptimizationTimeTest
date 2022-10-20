// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#pragma once

#include <vector>

#include <wpi/SymbolExports.h>

#include "Eigen/SparseCore"
#include "frc/autodiff/Expression.h"
#include "frc/autodiff/Profiler.h"
#include "frc/autodiff/Variable.h"

namespace frc::autodiff {

/**
 * This class calculates the Jacobian of a vector of variables with respect to a
 * vector of variables.
 *
 * The Jacobian is only recomputed if the variable expression is quadratic or
 * higher order.
 */
class WPILIB_DLLEXPORT Jacobian {
 public:
  /**
   * Constructs a Jacobian object.
   *
   * @param variables Variables of which to compute the Jacobian.
   * @param wrt Variables with respect to which to compute the Jacobian.
   */
  Jacobian(VectorXvar variables, VectorXvar wrt) noexcept;

  /**
   * Calculates the Jacobian.
   */
  const Eigen::SparseMatrix<double>& Calculate();

  /**
   * Updates the values of the variables.
   */
  void Update();

  /**
   * Returns the profiler.
   */
  Profiler& GetProfiler();

 private:
  VectorXvar m_variables;
  VectorXvar m_wrt;

  std::vector<std::vector<Expression*>> m_graph;

  Eigen::SparseMatrix<double> m_J{m_variables.rows(), m_wrt.rows()};

  // Cached triplets for gradients of linear rows
  std::vector<Eigen::Triplet<double>> m_cachedTriplets;

  // List of row indices for nonlinear rows whose graients will be computed in
  // Calculate()
  std::vector<int> m_nonlinearRows;

  Profiler m_profiler;

  /**
   * Computes the gradient for the given row and stores its triplets in
   * "triplets".
   *
   * @param row The row of which to compute the gradient.
   * @param triplets The destination storage for the gradient's triplets.
   */
  void ComputeRow(int row, std::vector<Eigen::Triplet<double>>& triplets);
};

}  // namespace frc::autodiff
