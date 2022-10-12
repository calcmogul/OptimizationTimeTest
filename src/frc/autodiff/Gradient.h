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
class WPILIB_DLLEXPORT Gradient {
 public:
  /**
   * Constructs a Gradient object.
   *
   * @param variable Variable of which to compute the gradient.
   * @param wrt Variable with respect to which to compute the gradient.
   */
  Gradient(Variable variable, Variable wrt) noexcept;

  /**
   * Constructs a Gradient object.
   *
   * @param variable Variable of which to compute the gradient.
   * @param wrt Variables with respect to which to compute the gradient.
   */
  Gradient(Variable variable, Eigen::Ref<VectorXvar> wrt) noexcept;

  /**
   * Calculates the gradient.
   */
  const Eigen::SparseVector<double>& Calculate();

  /**
   * Updates the value of the variable.
   */
  void Update();

  /**
   * Returns the profiler.
   */
  Profiler& GetProfiler();

 private:
  Variable m_variable;
  VectorXvar m_wrt;

  Eigen::SparseVector<double> m_g;

  Profiler m_profiler;

  void CalculateImpl();
};

}  // namespace frc::autodiff
