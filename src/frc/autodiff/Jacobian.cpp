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
    : m_variables{std::move(variables)}, m_wrt{std::move(wrt)} {
  m_profiler.Start();

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = row;
  }

  // Reserve triplet space for 99% sparsity
  m_cachedTriplets.reserve(m_variables.rows() * m_wrt.rows() * 0.01);

  for (int row = 0; row < m_variables.rows(); ++row) {
    if (m_variables(row).expr->type == ExpressionType::kLinear) {
      // If the row is linear, compute its gradient once here and cache its
      // triplets. Constant rows are ignored because their gradients have no
      // nonzero triplets.
      ComputeRow(row, m_cachedTriplets);
    } else if (m_variables(row).expr->type > ExpressionType::kLinear) {
      // If the row is quadratic or nonlinear, add it to the list of nonlinear
      // rows to be recomputed in Calculate().
      m_nonlinearRows.emplace_back(row);
    }
  }

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = -1;
  }

  if (m_nonlinearRows.empty()) {
    m_J.setFromTriplets(m_cachedTriplets.begin(), m_cachedTriplets.end());
  }

  m_profiler.Stop();
}

void Jacobian::Update() {
  for (int row : m_nonlinearRows) {
    m_variables(row).Update();
  }
}

Profiler& Jacobian::GetProfiler() {
  return m_profiler;
}

const Eigen::SparseMatrix<double>& Jacobian::Calculate() {
  if (m_nonlinearRows.empty()) {
    return m_J;
  }

  m_profiler.Start();

  Update();

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = row;
  }

  // Copy the cached triplets so triplets added for the nonlinear rows are
  // thrown away at the end of the function
  auto triplets = m_cachedTriplets;

  for (int row : m_nonlinearRows) {
    ComputeRow(row, triplets);
  }

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = -1;
  }

  m_J.setFromTriplets(triplets.begin(), triplets.end());

  m_profiler.Stop();

  return m_J;
}

void Jacobian::ComputeRow(int row,
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
