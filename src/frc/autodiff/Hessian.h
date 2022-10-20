// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#pragma once

#include <wpi/SymbolExports.h>

#include "Eigen/Core"
#include "Eigen/SparseCore"
#include "frc/autodiff/Jacobian.h"
#include "frc/autodiff/Profiler.h"
#include "frc/autodiff/Variable.h"

namespace frc::autodiff {

/**
 * This class calculates the Hessian of a variable with respect to a vector of
 * variables.
 *
 * The gradient tree is cached so subsequent Hessian calculations are faster,
 * and the Hessian is only recomputed if the variable expression is nonlinear.
 */
class WPILIB_DLLEXPORT Hessian {
 public:
  /**
   * Constructs a Hessian object.
   *
   * @param variable Variable of which to compute the gradient.
   * @param wrt Variables with respect to which to compute the gradient.
   */
  Hessian(Variable variable, Eigen::Ref<VectorXvar> wrt) noexcept;

  /**
   * Calculates the Hessian.
   */
  const Eigen::SparseMatrix<double>& Calculate();

  /**
   * Updates the values of the gradient tree.
   */
  void Update();

  /**
   * Returns the profiler.
   */
  Profiler& GetProfiler();

 private:
  Jacobian m_jacobian;

  /**
   * Returns the given variable's gradient tree.
   *
   * @param variable Variable of which to compute the gradient.
   * @param wrt Variables with respect to which to compute the gradient.
   */
  static VectorXvar GenerateGradientTree(Variable& variable,
                                         Eigen::Ref<VectorXvar> wrt);
};

}  // namespace frc::autodiff
