// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "frc/StateSpaceUtil.h"

namespace frc {

template <>
bool IsStabilizable<1, 1>(const Matrixd<1, 1>& A, const Matrixd<1, 1>& B) {
  return detail::IsStabilizableImpl<1, 1>(A, B);
}

template <>
bool IsStabilizable<2, 1>(const Matrixd<2, 2>& A, const Matrixd<2, 1>& B) {
  return detail::IsStabilizableImpl<2, 1>(A, B);
}

}  // namespace frc
