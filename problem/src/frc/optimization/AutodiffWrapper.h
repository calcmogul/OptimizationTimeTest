// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#pragma once

#include <variant>

#include <autodiff/reverse/var.hpp>
#include <wpi/SymbolExports.h>

#include "frc/EigenCore.h"

namespace frc {

class Problem;

class WPILIB_DLLEXPORT AutodiffWrapper {
 public:
  constexpr AutodiffWrapper() = default;

  AutodiffWrapper(const AutodiffWrapper& rhs) = default;
  AutodiffWrapper& operator=(const AutodiffWrapper& rhs) = default;

  AutodiffWrapper(AutodiffWrapper&&) = default;
  AutodiffWrapper& operator=(AutodiffWrapper&& rhs) = default;

  explicit AutodiffWrapper(autodiff::var& value);
  AutodiffWrapper& operator=(autodiff::var& value);

  explicit AutodiffWrapper(autodiff::var&& value);
  AutodiffWrapper& operator=(autodiff::var&& value);

  AutodiffWrapper(Problem* problem, int index);

  // Scalar constructors
  explicit AutodiffWrapper(double value);
  AutodiffWrapper& operator=(double rhs);

  explicit AutodiffWrapper(int value);
  AutodiffWrapper& operator=(int rhs);

  explicit AutodiffWrapper(const frc::Matrixd<1, 1>& value);
  AutodiffWrapper& operator=(const frc::Matrixd<1, 1>& rhs);

  friend AutodiffWrapper operator*(const AutodiffWrapper& lhs,
                                   const AutodiffWrapper& rhs) {
    return AutodiffWrapper{lhs.GetAutodiff() * rhs.GetAutodiff()};
  }

  AutodiffWrapper& operator*=(const AutodiffWrapper& rhs);

  friend AutodiffWrapper operator/(const AutodiffWrapper& lhs,
                                   const AutodiffWrapper& rhs) {
    return AutodiffWrapper{lhs.GetAutodiff() / rhs.GetAutodiff()};
  }

  AutodiffWrapper& operator/=(const AutodiffWrapper& rhs);

  friend AutodiffWrapper operator+(const AutodiffWrapper& lhs,
                                   const AutodiffWrapper& rhs) {
    return AutodiffWrapper{lhs.GetAutodiff() + rhs.GetAutodiff()};
  }

  AutodiffWrapper& operator+=(const AutodiffWrapper& rhs);

  friend AutodiffWrapper operator-(const AutodiffWrapper& lhs,
                                   const AutodiffWrapper& rhs) {
    return AutodiffWrapper{lhs.GetAutodiff() - rhs.GetAutodiff()};
  }

  AutodiffWrapper& operator-=(const AutodiffWrapper& rhs);

  friend AutodiffWrapper operator-(const AutodiffWrapper& lhs) {
    return AutodiffWrapper{-lhs.GetAutodiff()};
  }

  /**
   * Returns internal value of variable.
   */
  double Value() const;

  /**
   * Return reference to internal autodiff variable.
   */
  autodiff::var& GetAutodiff();

  /**
   * Return reference to internal autodiff variable.
   */
  const autodiff::var& GetAutodiff() const;

 private:
  /**
   * Points to an autodiff variable leaf stored in a Problem instance.
   *
   * We can't store a direct pointer to the autodiff variable because appending
   * variables to Problem's variable storage can cause reallocation and
   * invalidate old pointers.
   */
  struct ProblemRef {
    Problem* problem = nullptr;
    int index = 0;
  };

  // Variant is either an autodiff variable or an index into one from Problem.
  // We can't store a pointer to it because appending variables to Problem's
  // variable storage can cause reallocation and invalidate old pointers.
  //
  // std::monostate  empty; default constructor
  // autodiff::var   owning
  // autodiff::var*  non-owning
  // ProblemRef      non-owning
  mutable std::variant<std::monostate, autodiff::var, autodiff::var*,
                       ProblemRef>
      m_value;
};

}  // namespace frc
