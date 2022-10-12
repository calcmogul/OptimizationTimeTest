// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#pragma once

#include <vector>

#include <wpi/SymbolExports.h>

#include "Eigen/SparseCore"
#include "frc/autodiff/Expression.h"
#include "frc/autodiff/Gradient.h"
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
  std::vector<Gradient> m_gradients;

  // The highest order expression type in m_variables
  ExpressionType m_highestOrderType = ExpressionType::kNone;

  std::vector<Eigen::Triplet<double>> m_triplets;

  Eigen::SparseMatrix<double> m_J;

  Profiler m_profiler;

  void CalculateImpl();
};

}  // namespace frc::autodiff
