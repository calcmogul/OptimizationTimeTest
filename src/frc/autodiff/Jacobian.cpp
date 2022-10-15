// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "frc/autodiff/Jacobian.h"

#include <tuple>

#include <wpi/DenseMap.h>
#include <wpi/IntrusiveSharedPtr.h>

#include "frc/autodiff/Gradient.h"

using namespace frc::autodiff;

Jacobian::Jacobian(VectorXvar variables, VectorXvar wrt) noexcept
    : m_variables{std::move(variables)},
      m_wrt{std::move(wrt)},
      m_J{m_variables.rows(), m_wrt.rows()} {
  // Get the highest order expression type
  for (const auto& variable : m_variables) {
    if (m_highestOrderType < variable.Type()) {
      m_highestOrderType = variable.Type();
    }
  }

  // Reserve triplet space for 99% sparsity
  m_triplets.reserve(m_variables.rows() * m_wrt.rows() * 0.01);

  if (m_highestOrderType < ExpressionType::kLinear) {
    // If the expression is less than linear, the Jacobian is zero
    m_profiler.Start();
    m_J.setZero();
    m_profiler.Stop();
  } else if (m_highestOrderType == ExpressionType::kLinear) {
    // If the expression is linear, compute it once since it's constant
    CalculateImpl();
  }
}

const Eigen::SparseMatrix<double>& Jacobian::Calculate() {
  if (m_highestOrderType > ExpressionType::kLinear) {
    CalculateImpl();
  }

  return m_J;
}

void Jacobian::Update() {
  for (int row = 0; row < m_variables.rows(); ++row) {
    m_variables(row).Update();
  }
}

Profiler& Jacobian::GetProfiler() {
  return m_profiler;
}

void Jacobian::CalculateImpl() {
  m_profiler.Start();

  Update();

  m_triplets.clear();

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = row;
  }

  wpi::DenseMap<int, double> adjoints;

  // Stack element contains variable and its adjoint
  std::vector<std::tuple<Variable, double>> stack;
  stack.reserve(1024);

  for (int row = 0; row < m_variables.rows(); ++row) {
    adjoints.clear();

    stack.emplace_back(m_variables(row), 1.0);
    while (!stack.empty()) {
      Variable var = std::move(std::get<0>(stack.back()));
      double adjoint = std::move(std::get<1>(stack.back()));
      stack.pop_back();

      auto& lhs = var.expr->args[0];
      auto& rhs = var.expr->args[1];

      // The row is turned into a column to transpose the Jacobian
      int col = var.expr->row;

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

      if (col != -1) {
        adjoints[col] += adjoint;
      }
    }

    for (const auto& [col, adjoint] : adjoints) {
      m_triplets.emplace_back(row, col, adjoint);
    }
  }

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = -1;
  }

  m_J.setFromTriplets(m_triplets.begin(), m_triplets.end());

  m_profiler.Stop();
}
