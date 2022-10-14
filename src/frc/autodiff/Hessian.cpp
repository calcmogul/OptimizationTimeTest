// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "frc/autodiff/Hessian.h"

#include <tuple>
#include <unordered_map>

#include <wpi/IntrusiveSharedPtr.h>

#include "frc/autodiff/Gradient.h"

using namespace frc::autodiff;

Hessian::Hessian(Variable variable, Eigen::Ref<VectorXvar> wrt) noexcept
    : m_variables{GenerateGradientTree(variable, wrt)},
      m_wrt{std::move(wrt)},
      m_H{m_variables.rows(), m_variables.rows()} {
  // Get the highest order expression type
  for (const auto& variable : m_variables) {
    if (m_highestOrderType < variable.Type()) {
      m_highestOrderType = variable.Type();
    }
  }

  // Reserve triplet space for 99% sparsity
  m_triplets.reserve(m_variables.rows() * m_wrt.rows() * 0.01);

  if (m_highestOrderType < ExpressionType::kLinear) {
    // If the expression is less than linear, the Hessian is zero
    m_profiler.Start();
    m_H.setZero();
    m_profiler.Stop();
  } else if (m_highestOrderType == ExpressionType::kLinear) {
    // If the expression is linear, compute it once since it's constant
    CalculateImpl();
  }
}

const Eigen::SparseMatrix<double>& Hessian::Calculate() {
  if (m_highestOrderType > ExpressionType::kLinear) {
    CalculateImpl();
  }

  return m_H;
}

void Hessian::Update() {
  for (int row = 0; row < m_variables.rows(); ++row) {
    m_variables(row).Update();
  }
}

Profiler& Hessian::GetProfiler() {
  return m_profiler;
}

void Hessian::CalculateImpl() {
  m_profiler.Start();

  Update();

  m_triplets.clear();

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = row;
  }

  std::unordered_map<int, double> adjoints;

  // Stack element contains variable and its adjoint
  std::vector<std::tuple<Variable, double>> stack;
  stack.reserve(1024);

  for (int row = 0; row < m_variables.rows(); ++row) {
    if (row > 0) {
      m_wrt(row - 1).expr->row = -1;
    }

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
      if (row != col) {
        m_triplets.emplace_back(col, row, adjoint);
      }
    }
  }

  m_wrt(m_variables.size() - 1).expr->row = -1;

  m_H.setFromTriplets(m_triplets.begin(), m_triplets.end());

  m_profiler.Stop();
}

VectorXvar Hessian::GenerateGradientTree(Variable& variable,
                                         Eigen::Ref<VectorXvar> wrt) {
  // Read wpimath/README.md#Reverse_accumulation_automatic_differentiation for
  // background on reverse accumulation automatic differentiation.

  for (int row = 0; row < wrt.rows(); ++row) {
    wrt(row).expr->row = row;
  }

  std::unordered_map<int, wpi::IntrusiveSharedPtr<Expression>> adjoints;

  // Stack element contains variable and its adjoint
  std::vector<std::tuple<Variable, wpi::IntrusiveSharedPtr<Expression>>> stack;
  stack.reserve(1024);

  stack.emplace_back(variable, MakeConstant(1.0));
  while (!stack.empty()) {
    Variable var = std::move(std::get<0>(stack.back()));
    wpi::IntrusiveSharedPtr<Expression> adjoint =
        std::move(std::get<1>(stack.back()));
    stack.pop_back();

    auto& lhs = var.expr->args[0];
    auto& rhs = var.expr->args[1];

    int row = var.expr->row;

    if (lhs != nullptr) {
      stack.emplace_back(lhs, var.expr->gradientFuncs[0](lhs, rhs, adjoint));

      if (rhs != nullptr) {
        stack.emplace_back(rhs, var.expr->gradientFuncs[1](lhs, rhs, adjoint));
      }
    }

    if (row != -1) {
      if (adjoints[row] == nullptr) {
        adjoints[row] = adjoint;
      } else {
        adjoints[row] = adjoints[row] + adjoint;
      }
    }
  }

  for (int row = 0; row < wrt.rows(); ++row) {
    wrt(row).expr->row = -1;
  }

  VectorXvar grad{wrt.rows()};
  for (const auto& [row, adjoint] : adjoints) {
    auto expr = wrt(row).expr;
    if (expr != nullptr) {
      grad(row) = Variable{adjoint};
    }
  }

  // Free adjoint storage that's no longer needed
  for (auto& pair : adjoints) {
    pair.second = nullptr;
  }

  return grad;
}
