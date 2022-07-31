// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "frc/optimization/AutodiffWrapper.h"

#include "frc/optimization/Problem.h"

using namespace frc;

AutodiffWrapper::AutodiffWrapper(autodiff::var& value) : m_value{&value} {}

AutodiffWrapper& AutodiffWrapper::operator=(autodiff::var& value) {
  m_value = &value;
  return *this;
}

AutodiffWrapper::AutodiffWrapper(autodiff::var&& value)
    : m_value{std::move(value)} {}

AutodiffWrapper& AutodiffWrapper::operator=(autodiff::var&& value) {
  m_value = std::move(value);
  return *this;
}

AutodiffWrapper::AutodiffWrapper(Problem* problem, int index)
    : m_value{ProblemRef{problem, index}} {}

AutodiffWrapper::AutodiffWrapper(double value) : m_value{value} {}

AutodiffWrapper& AutodiffWrapper::operator=(double rhs) {
  GetAutodiff().update(rhs);
  return *this;
}

AutodiffWrapper::AutodiffWrapper(int value)
    : AutodiffWrapper{static_cast<double>(value)} {}

AutodiffWrapper& AutodiffWrapper::operator=(int rhs) {
  return operator=(static_cast<double>(rhs));
}

AutodiffWrapper::AutodiffWrapper(const frc::Matrixd<1, 1>& value)
    : AutodiffWrapper{value(0, 0)} {}

AutodiffWrapper& AutodiffWrapper::operator=(const frc::Matrixd<1, 1>& rhs) {
  return operator=(rhs(0, 0));
}

AutodiffWrapper& AutodiffWrapper::operator*=(const AutodiffWrapper& rhs) {
  GetAutodiff() *= rhs.GetAutodiff();
  return *this;
}

AutodiffWrapper& AutodiffWrapper::operator/=(const AutodiffWrapper& rhs) {
  GetAutodiff() /= rhs.GetAutodiff();
  return *this;
}

AutodiffWrapper& AutodiffWrapper::operator+=(const AutodiffWrapper& rhs) {
  GetAutodiff() += rhs.GetAutodiff();
  return *this;
}

AutodiffWrapper& AutodiffWrapper::operator-=(const AutodiffWrapper& rhs) {
  GetAutodiff() -= rhs.GetAutodiff();
  return *this;
}

double AutodiffWrapper::Value() const {
  return val(GetAutodiff());
}

autodiff::var& AutodiffWrapper::GetAutodiff() {
  if (std::holds_alternative<std::monostate>(m_value)) {
    m_value = autodiff::var{};
    return std::get<1>(m_value);
  } else if (std::holds_alternative<autodiff::var>(m_value)) {
    return std::get<1>(m_value);
  } else if (std::holds_alternative<autodiff::var*>(m_value)) {
    return *std::get<2>(m_value);
  } else {
    auto& varRef = std::get<3>(m_value);
    auto& leaves = varRef.problem->m_leaves;
    return leaves(varRef.index);
  }
}

const autodiff::var& AutodiffWrapper::GetAutodiff() const {
  if (std::holds_alternative<std::monostate>(m_value)) {
    m_value = autodiff::var{};
    return std::get<1>(m_value);
  } else if (std::holds_alternative<autodiff::var>(m_value)) {
    return std::get<1>(m_value);
  } else if (std::holds_alternative<autodiff::var*>(m_value)) {
    return *std::get<2>(m_value);
  } else {
    auto& varRef = std::get<3>(m_value);
    auto& leaves = varRef.problem->m_leaves;
    return leaves(varRef.index);
  }
}
