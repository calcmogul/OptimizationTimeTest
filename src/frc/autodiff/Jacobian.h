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

  Eigen::SparseMatrix<double> m_J{m_variables.rows(), m_wrt.rows()};

  std::vector<Eigen::Triplet<double>> m_cachedTriplets;

  std::vector<int> m_nonlinearRows;

  Profiler m_profiler;

  void ComputeRow(int row, std::vector<Eigen::Triplet<double>>& triplets);
};

}  // namespace frc::autodiff
