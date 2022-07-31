// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "frc/optimization/Variable.h"

namespace frc {

Variable<1, 1> abs(const Variable<1, 1>& x) {  // NOLINT
  return Variable<1, 1>{
      AutodiffWrapper{abs(x.GetStorage()(0, 0).GetAutodiff())}};
}

Variable<1, 1> acos(const Variable<1, 1>& x) {  // NOLINT
  return Variable<1, 1>{
      AutodiffWrapper{acos(x.GetStorage()(0, 0).GetAutodiff())}};
}

Variable<1, 1> asin(const Variable<1, 1>& x) {  // NOLINT
  return Variable<1, 1>{
      AutodiffWrapper{asin(x.GetStorage()(0, 0).GetAutodiff())}};
}

Variable<1, 1> atan(const Variable<1, 1>& x) {  // NOLINT
  return Variable<1, 1>{
      AutodiffWrapper{atan(x.GetStorage()(0, 0).GetAutodiff())}};
}

Variable<1, 1> atan2(const Variable<1, 1>& y,  // NOLINT
                     const Variable<1, 1>& x) {
  return Variable<1, 1>{AutodiffWrapper{atan2(
      y.GetStorage()(0, 0).GetAutodiff(), x.GetStorage()(0, 0).GetAutodiff())}};
}

Variable<1, 1> cos(const Variable<1, 1>& x) {  // NOLINT
  return Variable<1, 1>{
      AutodiffWrapper{cos(x.GetStorage()(0, 0).GetAutodiff())}};
}

Variable<1, 1> cosh(const Variable<1, 1>& x) {  // NOLINT
  return Variable<1, 1>{
      AutodiffWrapper{cosh(x.GetStorage()(0, 0).GetAutodiff())}};
}

Variable<1, 1> erf(const Variable<1, 1>& x) {  // NOLINT
  return Variable<1, 1>{
      AutodiffWrapper{erf(x.GetStorage()(0, 0).GetAutodiff())}};
}

Variable<1, 1> exp(const Variable<1, 1>& x) {  // NOLINT
  return Variable<1, 1>{
      AutodiffWrapper{exp(x.GetStorage()(0, 0).GetAutodiff())}};
}

Variable<1, 1> hypot(const Variable<1, 1>& x,  // NOLINT
                     const Variable<1, 1>& y) {
  return Variable<1, 1>{AutodiffWrapper{hypot(
      x.GetStorage()(0, 0).GetAutodiff(), y.GetStorage()(0, 0).GetAutodiff())}};
}

Variable<1, 1> hypot(const Variable<1, 1>& x,  // NOLINT
                     const Variable<1, 1>& y, const Variable<1, 1>& z) {
  return Variable<1, 1>{AutodiffWrapper{hypot(
      x.GetStorage()(0, 0).GetAutodiff(), y.GetStorage()(0, 0).GetAutodiff(),
      z.GetStorage()(0, 0).GetAutodiff())}};
}

Variable<1, 1> log(const Variable<1, 1>& x) {  // NOLINT
  return Variable<1, 1>{
      AutodiffWrapper{log(x.GetStorage()(0, 0).GetAutodiff())}};
}

Variable<1, 1> log10(const Variable<1, 1>& x) {  // NOLINT
  return Variable<1, 1>{
      AutodiffWrapper{log10(x.GetStorage()(0, 0).GetAutodiff())}};
}

Variable<1, 1> pow(const Variable<1, 1>& base, int power) {  // NOLINT
  return Variable<1, 1>{
      AutodiffWrapper{pow(base.GetStorage()(0, 0).GetAutodiff(), power)}};
}

Variable<1, 1> sin(const Variable<1, 1>& x) {  // NOLINT
  return Variable<1, 1>{
      AutodiffWrapper{sin(x.GetStorage()(0, 0).GetAutodiff())}};
}

Variable<1, 1> sinh(const Variable<1, 1>& x) {  // NOLINT
  return Variable<1, 1>{
      AutodiffWrapper{sinh(x.GetStorage()(0, 0).GetAutodiff())}};
}

Variable<1, 1> sqrt(const Variable<1, 1>& x) {  // NOLINT
  return Variable<1, 1>{
      AutodiffWrapper{sqrt(x.GetStorage()(0, 0).GetAutodiff())}};
}

Variable<1, 1> tan(const Variable<1, 1>& x) {  // NOLINT
  return Variable<1, 1>{
      AutodiffWrapper{tan(x.GetStorage()(0, 0).GetAutodiff())}};
}

Variable<1, 1> tanh(const Variable<1, 1>& x) {  // NOLINT
  return Variable<1, 1>{
      AutodiffWrapper{tanh(x.GetStorage()(0, 0).GetAutodiff())}};
}

}  // namespace frc
