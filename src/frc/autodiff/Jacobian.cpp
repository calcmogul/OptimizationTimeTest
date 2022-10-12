// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "frc/autodiff/Jacobian.h"

#include <tuple>

#include <wpi/IntrusiveSharedPtr.h>

#include "frc/autodiff/Gradient.h"

using namespace frc::autodiff;

Jacobian::Jacobian(Eigen::Ref<VectorXvar> variables, Eigen::Ref<VectorXvar> wrt)
    : m_J{variables.rows(), wrt.rows()} {
  m_gradients.reserve(variables.rows());
  for (int row = 0; row < variables.rows(); ++row) {
    m_gradients.emplace_back(variables(row), wrt);
  }

  // Get the highest order expression type
  for (const auto& variable : variables) {
    if (m_highestOrderType < variable.Type()) {
      m_highestOrderType = variable.Type();
    }
  }

  // Reserve triplet space for 99% sparsity
  m_triplets.reserve(variables.rows() * wrt.rows() * 0.01);

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

Eigen::SparseMatrix<double> Jacobian::Calculate() {
  if (m_highestOrderType > ExpressionType::kLinear) {
    CalculateImpl();
  }

  return m_J;
}

void Jacobian::Update() {
  for (auto& gradient : m_gradients) {
    gradient.Update();
  }
}

Profiler& Jacobian::GetProfiler() {
  return m_profiler;
}

void Jacobian::CalculateImpl() {
  m_profiler.Start();

  Update();

  m_triplets.clear();
  for (size_t row = 0; row < m_gradients.size(); ++row) {
    Eigen::SparseVector<double> g = m_gradients[row].Calculate();
    for (decltype(g)::InnerIterator it{g}; it; ++it) {
      m_triplets.emplace_back(row, it.index(), it.value());
    }
  }

  m_J.setFromTriplets(m_triplets.begin(), m_triplets.end());

  m_profiler.Stop();
}
