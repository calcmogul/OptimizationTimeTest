// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "frc/autodiff/Hessian.h"

#include <tuple>

#include <wpi/DenseMap.h>
#include <wpi/IntrusiveSharedPtr.h>

using namespace frc::autodiff;

Hessian::Hessian(Variable variable, Eigen::Ref<VectorXvar> wrt) noexcept
    : m_variables{GenerateGradientTree(variable, wrt)},
      m_wrt{std::move(wrt)},
      m_H{m_variables.rows(), m_variables.rows()} {
  m_profiler.Start();

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = row;
  }

  // Reserve triplet space for 99% sparsity
  m_cachedTriplets.reserve(m_variables.rows() * m_wrt.rows() * 0.01);

  for (int row = 0; row < m_variables.rows(); ++row) {
    if (m_variables(row).expr->type == ExpressionType::kLinear) {
      ComputeRow(row, m_cachedTriplets);
    } else if (m_variables(row).expr->type > ExpressionType::kLinear) {
      m_nonlinearRows.emplace_back(row);
    }
  }

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = -1;
  }

  if (m_nonlinearRows.empty()) {
    m_H.setFromTriplets(m_cachedTriplets.begin(), m_cachedTriplets.end());
  }

  m_profiler.Stop();
}

void Hessian::Update() {
  for (int row : m_nonlinearRows) {
    m_variables(row).Update();
  }
}

Profiler& Hessian::GetProfiler() {
  return m_profiler;
}

const Eigen::SparseMatrix<double>& Hessian::Calculate() {
  if (m_nonlinearRows.empty()) {
    return m_H;
  }

  m_profiler.Start();

  Update();

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = row;
  }

  auto triplets = m_cachedTriplets;
  for (int row : m_nonlinearRows) {
    ComputeRow(row, triplets);
  }

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = -1;
  }

  m_H.setFromTriplets(triplets.begin(), triplets.end());

  m_profiler.Stop();

  return m_H;
}

void Hessian::ComputeRow(int row,
                         std::vector<Eigen::Triplet<double>>& triplets) {
  wpi::DenseMap<int, double> adjoints;

  // Stack element contains variable and its adjoint
  std::vector<std::tuple<Variable, double>> stack;
  stack.reserve(1024);

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
    triplets.emplace_back(row, col, adjoint);
  }
}

VectorXvar Hessian::GenerateGradientTree(Variable& variable,
                                         Eigen::Ref<VectorXvar> wrt) {
  // Read wpimath/README.md#Reverse_accumulation_automatic_differentiation for
  // background on reverse accumulation automatic differentiation.

  for (int row = 0; row < wrt.rows(); ++row) {
    wrt(row).expr->row = row;
  }

  wpi::DenseMap<int, wpi::IntrusiveSharedPtr<Expression>> adjoints;

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
