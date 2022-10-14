// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "frc/autodiff/Gradient.h"

#include <tuple>
#include <unordered_map>

#include <wpi/IntrusiveSharedPtr.h>

using namespace frc::autodiff;

Gradient::Gradient(Variable variable, Variable wrt) noexcept
    : Gradient{std::move(variable), MapVectorXvar{&wrt, 1}} {}

Gradient::Gradient(Variable variable, Eigen::Ref<VectorXvar> wrt) noexcept
    : m_variable{std::move(variable)}, m_wrt{wrt}, m_g{m_wrt.rows()} {
  if (m_variable.Type() < ExpressionType::kLinear) {
    // If the expression is less than linear, the Jacobian is zero
    m_profiler.Start();
    m_g.setZero();
    m_profiler.Stop();
  } else if (m_variable.Type() == ExpressionType::kLinear) {
    // If the expression is linear, compute it once since it's constant
    CalculateImpl();
  }
}

const Eigen::SparseVector<double>& Gradient::Calculate() {
  if (m_variable.Type() > ExpressionType::kLinear) {
    CalculateImpl();
  }

  return m_g;
}

void Gradient::Update() {
  m_variable.Update();
}

Profiler& Gradient::GetProfiler() {
  return m_profiler;
}

void Gradient::CalculateImpl() {
  // Read wpimath/README.md#Reverse_accumulation_automatic_differentiation for
  // background on reverse accumulation automatic differentiation.

  m_profiler.Start();

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = row;
  }

  std::unordered_map<int, double> adjoints;

  // Stack element contains variable and its adjoint
  std::vector<std::tuple<Variable, double>> stack;
  stack.reserve(1024);

  stack.emplace_back(m_variable, 1.0);
  while (!stack.empty()) {
    Variable var = std::move(std::get<0>(stack.back()));
    double adjoint = std::move(std::get<1>(stack.back()));
    stack.pop_back();

    auto& lhs = var.expr->args[0];
    auto& rhs = var.expr->args[1];

    int row = var.expr->row;

    if (lhs != nullptr) {
      if (rhs == nullptr) {
        stack.emplace_back(
            lhs, var.expr->gradientValueFuncs[0](lhs->value, 0.0, adjoint));
      } else {
        stack.emplace_back(lhs, var.expr->gradientValueFuncs[0](
                                    lhs->value, rhs->value, adjoint));
        stack.emplace_back(rhs, var.expr->gradientValueFuncs[1](
                                    lhs->value, rhs->value, adjoint));
      }
    }

    if (row != -1) {
      adjoints[row] += adjoint;
    }
  }

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = -1;
  }

  m_g.setZero();
  for (const auto& [row, adjoint] : adjoints) {
    if (adjoint != 0.0) {
      m_g.insertBack(row) = adjoint;
    }
  }

  m_profiler.Stop();
}
