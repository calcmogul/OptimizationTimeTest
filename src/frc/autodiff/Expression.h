// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#pragma once

#include <stdint.h>

#include <array>
#include <memory>

#include <wpi/IntrusiveSharedPtr.h>
#include <wpi/SymbolExports.h>

#include "frc/autodiff/Pool.h"

namespace frc::autodiff {

enum class ExpressionType { kNone, kConstant, kLinear, kQuadratic, kNonlinear };

struct WPILIB_DLLEXPORT Expression {
  using BinaryFuncDouble = double (*)(double, double);
  using TrinaryFuncDouble = double (*)(double, double, double);
  using TrinaryFuncExpr = wpi::IntrusiveSharedPtr<Expression> (*)(
      const wpi::IntrusiveSharedPtr<Expression>&,
      const wpi::IntrusiveSharedPtr<Expression>&,
      const wpi::IntrusiveSharedPtr<Expression>&);

  double value = 0.0;
  double adjoint = 0.0;

  // Tracks the number of instances of this expression yet to be encountered in
  // an expression tree.
  int duplications = 0;

  // This expression's row in wrt for autodiff gradient, Jacobian, or Hessian.
  // This is -1 if the expression isn't in wrt.
  int row = -1;

  // The expression's creation order. The value assigned here is from a
  // monotonically increasing counter that increments every time an Expression
  // is constructed. This is used for sorting a flattened representation of the
  // expression tree in autodiff Jacobian or Hessian.
  size_t id;

  wpi::IntrusiveSharedPtr<Expression> adjointExpr;

  // Expression argument type
  ExpressionType type = frc::autodiff::ExpressionType::kLinear;

  // Either nullary operator with no arguments, unary operator with one
  // argument, or binary operator with two arguments. This operator is
  // used to update the node's value.
  BinaryFuncDouble valueFunc = [](double, double) { return 0.0; };

  /// Functions returning double adjoints of the children expressions.
  ///
  /// Parameters:
  /// <ul>
  ///   <li>lhs: Left argument to binary operator.</li>
  ///   <li>rhs: Right argument to binary operator.</li>
  ///   <li>parentAdjoint: Adjoint of parent expression.</li>
  /// </ul>
  std::array<TrinaryFuncDouble, 2> gradientValueFuncs{
      [](double, double, double) { return 0.0; },
      [](double, double, double) { return 0.0; }};

  /// Functions returning Variable adjoints of the children expressions.
  ///
  /// Parameters:
  /// <ul>
  ///   <li>lhs: Left argument to binary operator.</li>
  ///   <li>rhs: Right argument to binary operator.</li>
  ///   <li>parentAdjoint: Adjoint of parent expression.</li>
  /// </ul>
  std::array<TrinaryFuncExpr, 2> gradientFuncs{
      [](const wpi::IntrusiveSharedPtr<Expression>&,
         const wpi::IntrusiveSharedPtr<Expression>&,
         const wpi::IntrusiveSharedPtr<Expression>&) {
        return wpi::IntrusiveSharedPtr<Expression>{};
      },
      [](const wpi::IntrusiveSharedPtr<Expression>&,
         const wpi::IntrusiveSharedPtr<Expression>&,
         const wpi::IntrusiveSharedPtr<Expression>&) {
        return wpi::IntrusiveSharedPtr<Expression>{};
      }};

  // Expression arguments
  std::array<wpi::IntrusiveSharedPtr<Expression>, 2> args{nullptr, nullptr};

  // Reference count for intrusive shared pointer
  uint32_t refCount = 0;

  Expression(const Expression&) = default;
  Expression& operator=(const Expression&) = default;

  Expression(Expression&&) = default;
  Expression& operator=(Expression&&) = default;

  /**
   * Constructs a nullary expression (an operator with no arguments).
   *
   * @param value The expression value.
   * @param type The expression type. It should be either linear (the default)
   *             or constant.
   */
  explicit Expression(double value,
                      ExpressionType type = ExpressionType::kLinear);

  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param type The expression's type.
   * @param valueFunc Unary operator that produces this expression's value.
   * @param lhsGradientValueFunc Gradient with respect to the operand.
   * @param lhsGradientFunc Gradient with respect to the operand.
   * @param lhs Unary operator's operand.
   */
  Expression(ExpressionType type, BinaryFuncDouble valueFunc,
             TrinaryFuncDouble lhsGradientValueFunc,
             TrinaryFuncExpr lhsGradientFunc,
             wpi::IntrusiveSharedPtr<Expression> lhs);

  /**
   * Constructs a binary expression (an operator with two arguments).
   *
   * @param type The expression's type.
   * @param valueFunc Unary operator that produces this expression's value.
   * @param lhsGradientValueFunc Gradient with respect to the left operand.
   * @param rhsGradientValueFunc Gradient with respect to the right operand.
   * @param lhsGradientFunc Gradient with respect to the left operand.
   * @param rhsGradientFunc Gradient with respect to the right operand.
   * @param lhs Binary operator's left operand.
   * @param rhs Binary operator's right operand.
   */
  Expression(ExpressionType type, BinaryFuncDouble valueFunc,
             TrinaryFuncDouble lhsGradientValueFunc,
             TrinaryFuncDouble rhsGradientValueFunc,
             TrinaryFuncExpr lhsGradientFunc, TrinaryFuncExpr rhsGradientFunc,
             wpi::IntrusiveSharedPtr<Expression> lhs,
             wpi::IntrusiveSharedPtr<Expression> rhs);

  friend WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator*(
      double lhs, const wpi::IntrusiveSharedPtr<Expression>& rhs);

  friend WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator*(
      const wpi::IntrusiveSharedPtr<Expression>& lhs, double rhs);

  friend WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator*(
      const wpi::IntrusiveSharedPtr<Expression>& lhs,
      const wpi::IntrusiveSharedPtr<Expression>& rhs);

  friend WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator/(
      double lhs, const wpi::IntrusiveSharedPtr<Expression>& rhs);

  friend WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator/(
      const wpi::IntrusiveSharedPtr<Expression>& lhs, double rhs);

  friend WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator/(
      const wpi::IntrusiveSharedPtr<Expression>& lhs,
      const wpi::IntrusiveSharedPtr<Expression>& rhs);

  friend WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator+(
      double lhs, const wpi::IntrusiveSharedPtr<Expression>& rhs);

  friend WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator+(
      const wpi::IntrusiveSharedPtr<Expression>& lhs, double rhs);

  friend WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator+(
      const wpi::IntrusiveSharedPtr<Expression>& lhs,
      const wpi::IntrusiveSharedPtr<Expression>& rhs);

  friend WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator-(
      double lhs, const wpi::IntrusiveSharedPtr<Expression>& rhs);

  friend WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator-(
      const wpi::IntrusiveSharedPtr<Expression>& lhs, double rhs);

  friend WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator-(
      const wpi::IntrusiveSharedPtr<Expression>& lhs,
      const wpi::IntrusiveSharedPtr<Expression>& rhs);

  friend WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator-(
      const wpi::IntrusiveSharedPtr<Expression>& lhs);

  friend WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator+(
      const wpi::IntrusiveSharedPtr<Expression>& lhs);

  /**
   * Update the value of this node based on the values of its dependent
   * nodes.
   */
  void Update();
};

WPILIB_DLLEXPORT PoolAllocator<Expression> Allocator();

/**
 * Refcount increment for intrusive shared pointer.
 *
 * @param expr The shared pointer's managed object.
 */
inline void IntrusiveSharedPtrIncRefCount(Expression* expr) {
  ++expr->refCount;
}

/**
 * Refcount decrement for intrusive shared pointer.
 *
 * @param expr The shared pointer's managed object.
 */
inline void IntrusiveSharedPtrDecRefCount(Expression* expr) {
  if (--expr->refCount == 0) {
    auto alloc = Allocator();
    std::allocator_traits<decltype(alloc)>::destroy(alloc, expr);
    std::allocator_traits<decltype(alloc)>::deallocate(alloc, expr,
                                                       sizeof(Expression));
  }
}

/**
 * Creates a constant expression.
 *
 * @param x The constant.
 */
WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> MakeConstant(double x);

/**
 * std::abs() for Expressions.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> abs(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x);

/**
 * std::acos() for Expressions.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> acos(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x);

/**
 * std::asin() for Expressions.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> asin(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x);

/**
 * std::atan() for Expressions.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> atan(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x);

/**
 * std::atan2() for Expressions.
 *
 * @param y The y argument.
 * @param x The x argument.
 */
WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> atan2(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& y,
    const wpi::IntrusiveSharedPtr<Expression>& x);

/**
 * std::cos() for Expressions.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> cos(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x);

/**
 * std::cosh() for Expressions.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> cosh(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x);

/**
 * std::erf() for Expressions.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> erf(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x);

/**
 * std::exp() for Expressions.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> exp(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x);

/**
 * std::hypot() for Expressions.
 *
 * @param x The x argument.
 * @param y The y argument.
 */
WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> hypot(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x,
    const wpi::IntrusiveSharedPtr<Expression>& y);

/**
 * std::log() for Expressions.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> log(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x);

/**
 * std::log10() for Expressions.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> log10(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x);

/**
 * std::pow() for Expressions.
 *
 * @param base The base.
 * @param power The power.
 */
WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> pow(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& base,
    const wpi::IntrusiveSharedPtr<Expression>& power);

/**
 * std::sin() for Expressions.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> sin(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x);

/**
 * std::sinh() for Expressions.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> sinh(
    const wpi::IntrusiveSharedPtr<Expression>& x);

/**
 * std::sqrt() for Expressions.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> sqrt(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x);

/**
 * std::tan() for Expressions.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> tan(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x);

/**
 * std::tanh() for Expressions.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> tanh(
    const wpi::IntrusiveSharedPtr<Expression>& x);

}  // namespace frc::autodiff
