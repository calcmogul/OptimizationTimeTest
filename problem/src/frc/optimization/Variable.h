// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#pragma once

#include <initializer_list>
#include <type_traits>
#include <utility>

#include <wpi/SymbolExports.h>

#include "frc/EigenCore.h"
#include "frc/optimization/AutodiffWrapper.h"

namespace frc {

template <int _Rows, int _Cols>
class Variable {
 public:
  Variable() = default;

  template <int _Rows2 = _Rows, int _Cols2 = _Cols,
            std::enable_if_t<_Rows2 == 1 && _Cols2 == 1, int> = 0>
  Variable(double value) : m_storage{{AutodiffWrapper{value}}} {}  // NOLINT

  Variable(std::initializer_list<double> values) : m_storage{values} {}

  Variable(std::initializer_list<std::initializer_list<double>> values)
      : m_storage{values} {}

  template <int _Rows2 = _Rows, int _Cols2 = _Cols,
            std::enable_if_t<_Rows2 == 1 && _Cols2 == 1, int> = 0>
  explicit Variable(const AutodiffWrapper& rhs) {
    m_storage(0, 0) = rhs;
  }

  template <int _Rows2 = _Rows, int _Cols2 = _Cols,
            std::enable_if_t<_Rows2 == 1 && _Cols2 == 1, int> = 0>
  explicit Variable(AutodiffWrapper&& rhs) : m_storage{std::move(rhs)} {}

  Variable(const frc::Matrixd<_Rows, _Cols>& values)  // NOLINT
      : m_storage{values.template cast<AutodiffWrapper>()} {}

  Variable& operator=(const frc::Matrixd<_Rows, _Cols>& values) {
    for (size_t row = 0; row < _Rows; ++row) {
      for (size_t col = 0; col < _Cols; ++col) {
        m_storage(row, col) = values(row, col);
      }
    }

    return *this;
  }

  explicit Variable(frc::Matrixd<_Rows, _Cols>&& values)
      : m_storage{values.template cast<AutodiffWrapper>()} {}

  Variable& operator=(frc::Matrixd<_Rows, _Cols>&& values) {
    for (size_t row = 0; row < _Rows; ++row) {
      for (size_t col = 0; col < _Cols; ++col) {
        m_storage(row, col) = values(row, col);
      }
    }

    return *this;
  }

  explicit Variable(const Eigen::Matrix<AutodiffWrapper, _Rows, _Cols>& values)
      : m_storage{values} {}

  Variable& operator=(
      const Eigen::Matrix<AutodiffWrapper, _Rows, _Cols>& values) {
    m_storage = values;
    return *this;
  }

  explicit Variable(Eigen::Matrix<AutodiffWrapper, _Rows, _Cols>&& values)
      : m_storage{std::move(values)} {}

  Variable& operator=(Eigen::Matrix<AutodiffWrapper, _Rows, _Cols>&& values) {
    m_storage = std::move(values);
    return *this;
  }

  Variable<1, 1> operator()(int row, int col) {
    return Variable<1, 1>{m_storage(row, col)};
  }

  template <int _Cols2 = _Cols, std::enable_if_t<_Cols2 == 1, int> = 0>
  Variable<1, 1> operator()(int row) {
    return Variable<1, 1>{m_storage(row, 0)};
  }

  /**
   * Returns a block slice of the variable matrix.
   *
   * @tparam Block_Rows The number of rows in the block selection.
   * @tparam Block_Cols The number of columns in the block selection.
   * @param rowOffset The row offset of the block selection.
   * @param colOffset The column offset of the block selection.
   */
  template <int Block_Rows, int Block_Cols>
  Variable<Block_Rows, Block_Cols> Block(int rowOffset, int colOffset) {
    Variable<Block_Rows, Block_Cols> ret;

    for (int row = 0; row < Block_Rows; ++row) {
      for (int col = 0; col < Block_Cols; ++col) {
        ret.GetStorage()(row, col) =
            m_storage(row + rowOffset, col + colOffset);
      }
    }

    return ret;
  }

  /**
   * Returns a row slice of the variable matrix.
   *
   * @param row The row to slice.
   */
  auto Row(int row) { return Block<1, _Cols>(row, 0); }

  /**
   * Returns a column slice of the variable matrix.
   *
   * @param col The column to slice.
   */
  auto Col(int col) { return Block<_Rows, 1>(0, col); }

  template <int _Rows2 = _Rows, int _Cols2 = _Cols,
            std::enable_if_t<_Rows2 == 1 && _Cols2 == 1, int> = 0>
  Variable<_Rows, _Cols>& operator=(double rhs) {
    m_storage(0, 0) = rhs;
    return *this;
  }

  /**
   * Matrix multiplication operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  template <int _RowsRhs, int _ColsRhs>
  friend Variable<_Rows, _ColsRhs> operator*(
      const Variable<_Rows, _Cols>& lhs,
      const Variable<_RowsRhs, _ColsRhs>& rhs) {
    static_assert(_Cols == _RowsRhs, "Matrix dimension mismatch for operator*");
    return Variable<_Rows, _ColsRhs>{
        Eigen::Matrix<AutodiffWrapper, _Rows, _ColsRhs>{lhs.m_storage *
                                                        rhs.GetStorage()}};
  }

  /**
   * Matrix multiplication operator (only enabled when lhs is a scalar).
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  template <int _Rows2 = _Rows, int _Cols2 = _Cols,
            std::enable_if_t<_Rows2 == 1 && _Cols2 == 1, int> = 0>
  friend Variable<_Rows, _Cols> operator*(const Variable<_Rows, _Cols>& lhs,
                                          double rhs) {
    return Variable<_Rows, _Cols>{Eigen::Matrix<AutodiffWrapper, _Rows, _Cols>{
        lhs.m_storage *
        Eigen::Matrix<AutodiffWrapper, 1, 1>{AutodiffWrapper{rhs}}}};
  }

  /**
   * Compound matrix multiplication-assignment operator.
   *
   * @param rhs Variable to multiply.
   */
  template <int _RowsRhs, int _ColsRhs>
  Variable<_Rows, _ColsRhs>& operator*=(
      const Variable<_RowsRhs, _ColsRhs>& rhs) {
    static_assert(_Cols == _RowsRhs, "Matrix dimension mismatch for operator*");
    m_storage *= rhs.GetStorage();
    return *this;
  }

  /**
   * Compound matrix multiplication-assignment operator (only enabled when lhs
   * is a scalar).
   *
   * @param rhs Variable to multiply.
   */
  template <int _Rows2 = _Rows, int _Cols2 = _Cols,
            std::enable_if_t<_Rows2 == 1 && _Cols2 == 1, int> = 0>
  Variable<_Rows, _Cols>& operator*=(double rhs) {
    m_storage *= Eigen::Matrix<AutodiffWrapper, 1, 1>{AutodiffWrapper{rhs}};
    return *this;
  }

  /**
   * Binary addition operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend Variable<_Rows, _Cols> operator+(const Variable<_Rows, _Cols>& lhs,
                                          const Variable<_Rows, _Cols>& rhs) {
    return Variable<_Rows, _Cols>{Eigen::Matrix<AutodiffWrapper, _Rows, _Cols>{
        lhs.m_storage + rhs.m_storage}};
  }

  /**
   * Compound addition-assignment operator.
   *
   * @param rhs Variable to add.
   */
  Variable<_Rows, _Cols>& operator+=(const Variable<_Rows, _Cols>& rhs) {
    m_storage += rhs.m_storage;
    return *this;
  }

  /**
   * Binary subtraction operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend Variable<_Rows, _Cols> operator-(const Variable<_Rows, _Cols>& lhs,
                                          const Variable<_Rows, _Cols>& rhs) {
    return Variable<_Rows, _Cols>{Eigen::Matrix<AutodiffWrapper, _Rows, _Cols>{
        lhs.m_storage - rhs.m_storage}};
  }

  /**
   * Compound subtraction-assignment operator.
   *
   * @param rhs Variable to subtract.
   */
  Variable<_Rows, _Cols>& operator-=(const Variable<_Rows, _Cols>& rhs) {
    m_storage -= rhs.m_storage;
    return *this;
  }

  /**
   * Unary minus operator.
   *
   * @param lhs Operand for unary minus.
   */
  friend Variable<_Rows, _Cols> operator-(const Variable<_Rows, _Cols>& lhs) {
    return Variable<_Rows, _Cols>{
        Eigen::Matrix<AutodiffWrapper, _Rows, _Cols>{-lhs.m_storage}};
  }

  /**
   * Returns the transpose of the variable matrix.
   */
  Variable<_Cols, _Rows> Transpose() const {
    return Variable<_Cols, _Rows>{
        Eigen::Matrix<AutodiffWrapper, _Cols, _Rows>{m_storage.transpose()}};
  }

  /**
   * Returns number of rows in the matrix.
   */
  int Rows() const { return _Rows; }

  /**
   * Returns number of columns in the matrix.
   */
  int Cols() const { return _Cols; }

  /**
   * Returns an element of the variable matrix.
   *
   * @param row The row of the element to return.
   * @param col The column of the element to return.
   */
  double Value(int row, int col) const { return m_storage(row, col).Value(); }

  /**
   * Returns a row of the variable column vector (only enabled when variable is
   * a column vector).
   *
   * @param row The row of the element to return.
   */
  template <int _Rows2 = _Rows, int _Cols2 = _Cols,
            std::enable_if_t<_Rows2 != 1 && _Cols2 == 1, int> = 0>
  double Value(int row) const {
    return m_storage(row, 0).Value();
  }

  /**
   * Returns a column of the variable row vector (only enabled when variable is
   * a row vector).
   *
   * @param col The column of the element to return.
   */
  template <int _Rows2 = _Rows, int _Cols2 = _Cols,
            std::enable_if_t<_Rows2 == 1 && _Cols2 != 1, int> = 0>
  double Value(int col) const {
    return m_storage(0, col).Value();
  }

  /**
   * Returns the underlying matrix representation of this variable.
   */
  template <int _Rows2 = _Rows, int _Cols2 = _Cols,
            std::enable_if_t<_Rows2 != 1 || _Cols2 != 1, int> = 0>
  frc::Matrixd<_Rows, _Cols> Value() const {
    frc::Matrixd<_Rows, _Cols> ret;
    for (size_t row = 0; row < _Rows; ++row) {
      for (size_t col = 0; col < _Cols; ++col) {
        ret(row, col) = m_storage(row, col).Value();
      }
    }
    return ret;
  }

  /**
   * Returns the underlying scalar representation of this variable.
   */
  template <int _Rows2 = _Rows, int _Cols2 = _Cols,
            std::enable_if_t<_Rows2 == 1 && _Cols2 == 1, int> = 0>
  double Value() const {
    return m_storage(0, 0).Value();
  }

  /**
   * Returns the internal storage of autodiff variable wrappers.
   */
  Eigen::Matrix<AutodiffWrapper, _Rows, _Cols>& GetStorage() {
    return m_storage;
  }

  /**
   * Returns the internal storage of autodiff variable wrappers.
   */
  const Eigen::Matrix<AutodiffWrapper, _Rows, _Cols>& GetStorage() const {
    return m_storage;
  }

 private:
  Eigen::Matrix<AutodiffWrapper, _Rows, _Cols> m_storage;
};

/**
 * std::abs() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> abs(const Variable<1, 1>& x);  // NOLINT

/**
 * std::acos() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> acos(const Variable<1, 1>& x);  // NOLINT

/**
 * std::asin() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> asin(const Variable<1, 1>& x);  // NOLINT

/**
 * std::atan() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> atan(const Variable<1, 1>& x);  // NOLINT

/**
 * std::atan2() for Variables.
 *
 * @param y The y argument.
 * @param x The x argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> atan2(const Variable<1, 1>& y,  // NOLINT
                                      const Variable<1, 1>& x);

/**
 * std::cos() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> cos(const Variable<1, 1>& x);  // NOLINT

/**
 * std::cosh() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> cosh(const Variable<1, 1>& x);  // NOLINT

/**
 * std::erf() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> erf(const Variable<1, 1>& x);  // NOLINT

/**
 * std::exp() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> exp(const Variable<1, 1>& x);  // NOLINT

/**
 * std::hypot() for Variables.
 *
 * @param x The x argument.
 * @param y The y argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> hypot(const Variable<1, 1>& x,  // NOLINT
                                      const Variable<1, 1>& y);

/**
 * std::hypot() for Variables.
 *
 * @param x The x argument.
 * @param y The y argument.
 * @param z The z argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> hypot(const Variable<1, 1>& x,  // NOLINT
                                      const Variable<1, 1>& y,
                                      const Variable<1, 1>& z);

/**
 * std::log() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> log(const Variable<1, 1>& x);  // NOLINT

/**
 * std::log10() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> log10(const Variable<1, 1>& x);  // NOLINT

/**
 * std::pow() for Variables.
 *
 * @param base The base.
 * @param power The power.
 */
WPILIB_DLLEXPORT Variable<1, 1> pow(const Variable<1, 1>& base,  // NOLINT
                                    int power);

/**
 * std::sin() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> sin(const Variable<1, 1>& x);  // NOLINT

/**
 * std::sinh() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> sinh(const Variable<1, 1>& x);  // NOLINT

/**
 * std::sqrt() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> sqrt(const Variable<1, 1>& x);  // NOLINT

/**
 * std::tan() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> tan(const Variable<1, 1>& x);  // NOLINT

/**
 * std::tanh() for Variables.
 *
 * @param x The argument.
 */
WPILIB_DLLEXPORT Variable<1, 1> tanh(const Variable<1, 1>& x);  // NOLINT

}  // namespace frc
