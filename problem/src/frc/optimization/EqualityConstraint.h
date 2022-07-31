// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#pragma once

#include <functional>

#include <wpi/SymbolExports.h>

#include "frc/EigenCore.h"
#include "frc/optimization/AutodiffWrapper.h"
#include "frc/optimization/ConstraintUtil.h"
#include "frc/optimization/Variable.h"

namespace frc {

/**
 * An equality constraint has the form c(x) = 0.
 */
struct WPILIB_DLLEXPORT EqualityConstraint {
  AutodiffWrapper variable;

  constexpr EqualityConstraint() = default;

  explicit EqualityConstraint(AutodiffWrapper variable);
};

/**
 * Returns a matrix of equality constraints.
 *
 * @param func A function that returns the equality constraint for a given row
 *             and column.
 */
template <int Rows, int Cols>
Eigen::Matrix<EqualityConstraint, Rows, Cols> MakeEqualityConstraintMatrix(
    std::function<EqualityConstraint(int, int)> func) {
  Eigen::Matrix<EqualityConstraint, Rows, Cols> constraints;

  for (int row = 0; row < Rows; ++row) {
    for (int col = 0; col < Cols; ++col) {
      constraints(row, col) = func(row, col);
    }
  }

  return constraints;
}

// Matrix-matrix equality operators

template <int Rows, int Cols>
Eigen::Matrix<EqualityConstraint, Rows, Cols> operator==(
    const Variable<Rows, Cols>& lhs, const Variable<Rows, Cols>& rhs) {
  return MakeEqualityConstraintMatrix<Rows, Cols>([=](int row, int col) {
    // Make right-hand side zero
    return EqualityConstraint{lhs.GetStorage()(row, col) -
                              rhs.GetStorage()(row, col)};
  });
}

template <int Rows, int Cols>
Eigen::Matrix<EqualityConstraint, Rows, Cols> operator==(
    const Variable<Rows, Cols>& lhs, const frc::Matrixd<Rows, Cols>& rhs) {
  Variable<Rows, Cols> rhsVar{rhs};
  return MakeEqualityConstraintMatrix<Rows, Cols>([=](int row, int col) {
    // Make right-hand side zero
    return EqualityConstraint{lhs.GetStorage()(row, col) -
                              rhsVar.GetStorage()(row, col)};
  });
}

template <int Rows, int Cols>
Eigen::Matrix<EqualityConstraint, Rows, Cols> operator==(
    const frc::Matrixd<Rows, Cols>& lhs, const Variable<Rows, Cols>& rhs) {
  Variable<Rows, Cols> lhsVar{lhs};
  return MakeEqualityConstraintMatrix<Rows, Cols>([=](int row, int col) {
    // Make right-hand side zero
    return EqualityConstraint{lhsVar.GetStorage()(row, col) -
                              rhs.GetStorage()(row, col)};
  });
}

// Matrix-scalar equality operator
template <int Rows, int Cols, typename Rhs,
          std::enable_if_t<IsScalar<Rhs>, int> = 0>
Eigen::Matrix<EqualityConstraint, Rows, Cols> operator==(
    const Variable<Rows, Cols>& lhs, const Rhs& rhs) {
  AutodiffWrapper rhsVar{rhs};
  return MakeEqualityConstraintMatrix<Rows, Cols>([=](int row, int col) {
    // Make right-hand side zero
    return EqualityConstraint{lhs.GetStorage()(row, col) - rhsVar};
  });
}

// Scalar-matrix equality operator
template <int Rows, int Cols, typename Lhs,
          std::enable_if_t<IsScalar<Lhs>, int> = 0>
Eigen::Matrix<EqualityConstraint, Rows, Cols> operator==(
    const Lhs& lhs, const Variable<Rows, Cols>& rhs) {
  AutodiffWrapper lhsVar{lhs};
  return MakeEqualityConstraintMatrix<Rows, Cols>([=](int row, int col) {
    // Make right-hand side zero
    return EqualityConstraint{lhsVar - rhs.GetStorage()(row, col)};
  });
}

}  // namespace frc
