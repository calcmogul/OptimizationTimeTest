// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include "frc/autodiff/Expression.h"

#include <cmath>
#include <utility>

#include <wpi/numbers>

namespace frc::autodiff {

Expression::Expression(double value, ExpressionType type)
    : value{value}, type{type} {}

Expression::Expression(ExpressionType type, BinaryFuncDouble valueFunc,
                       TrinaryFuncDouble lhsGradientValueFunc,
                       TrinaryFuncExpr lhsGradientFunc,
                       wpi::IntrusiveSharedPtr<Expression> lhs)
    : value{valueFunc(lhs->value, 0.0)},
      type{type},
      valueFunc{valueFunc},
      gradientValueFuncs{lhsGradientValueFunc, TrinaryFuncDouble{}},
      gradientFuncs{lhsGradientFunc, TrinaryFuncExpr{}},
      args{lhs, nullptr} {}

Expression::Expression(ExpressionType type, BinaryFuncDouble valueFunc,
                       TrinaryFuncDouble lhsGradientValueFunc,
                       TrinaryFuncDouble rhsGradientValueFunc,
                       TrinaryFuncExpr lhsGradientFunc,
                       TrinaryFuncExpr rhsGradientFunc,
                       wpi::IntrusiveSharedPtr<Expression> lhs,
                       wpi::IntrusiveSharedPtr<Expression> rhs)
    : value{valueFunc(lhs != nullptr ? lhs->value : 0.0,
                      rhs != nullptr ? rhs->value : 0.0)},
      type{type},
      valueFunc{valueFunc},
      gradientValueFuncs{lhsGradientValueFunc, rhsGradientValueFunc},
      gradientFuncs{lhsGradientFunc, rhsGradientFunc},
      args{lhs, rhs} {}

WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator*(
    double lhs, const wpi::IntrusiveSharedPtr<Expression>& rhs) {
  if (lhs == 0.0) {
    return nullptr;
  } else if (lhs == 1.0) {
    return rhs;
  }

  return MakeConstant(lhs) * rhs;
}

WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator*(
    const wpi::IntrusiveSharedPtr<Expression>& lhs, double rhs) {
  if (rhs == 0.0) {
    return nullptr;
  } else if (rhs == 1.0) {
    return lhs;
  }

  return lhs * MakeConstant(rhs);
}

WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator*(
    const wpi::IntrusiveSharedPtr<Expression>& lhs,
    const wpi::IntrusiveSharedPtr<Expression>& rhs) {
  if (lhs == nullptr || rhs == nullptr) {
    return nullptr;
  }

  if (lhs->type == ExpressionType::kConstant) {
    if (lhs->value == 1.0) {
      return rhs;
    } else if (lhs->value == 0.0) {
      return nullptr;
    }
  }

  if (rhs->type == ExpressionType::kConstant) {
    if (rhs->value == 1.0) {
      return lhs;
    } else if (rhs->value == 0.0) {
      return nullptr;
    }
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (lhs->type == ExpressionType::kConstant) {
    type = rhs->type;
  } else if (rhs->type == ExpressionType::kConstant) {
    type = lhs->type;
  } else if (lhs->type == ExpressionType::kLinear &&
             rhs->type == ExpressionType::kLinear) {
    type = ExpressionType::kQuadratic;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      type, [](double lhs, double rhs) { return lhs * rhs; },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint * rhs;
      },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint * lhs;
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& lhs,
         const wpi::IntrusiveSharedPtr<Expression>& rhs,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * rhs;
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& lhs,
         const wpi::IntrusiveSharedPtr<Expression>& rhs,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * lhs;
      },
      lhs, rhs);
}

WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator/(
    double lhs, const wpi::IntrusiveSharedPtr<Expression>& rhs) {
  if (lhs == 0.0) {
    return nullptr;
  }

  return MakeConstant(lhs) / rhs;
}

WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator/(
    const wpi::IntrusiveSharedPtr<Expression>& lhs, double rhs) {
  return lhs / MakeConstant(rhs);
}

WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator/(
    const wpi::IntrusiveSharedPtr<Expression>& lhs,
    const wpi::IntrusiveSharedPtr<Expression>& rhs) {
  if (lhs == nullptr) {
    return nullptr;
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (rhs->type == ExpressionType::kConstant) {
    type = lhs->type;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      type, [](double lhs, double rhs) { return lhs / rhs; },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint / rhs;
      },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint * -lhs / (rhs * rhs);
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& lhs,
         const wpi::IntrusiveSharedPtr<Expression>& rhs,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint / rhs;
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& lhs,
         const wpi::IntrusiveSharedPtr<Expression>& rhs,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * -lhs / (rhs * rhs);
      },
      lhs, rhs);
}

WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator+(
    double lhs, const wpi::IntrusiveSharedPtr<Expression>& rhs) {
  if (lhs == 0.0) {
    return rhs;
  }

  return MakeConstant(lhs) + rhs;
}

WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator+(
    const wpi::IntrusiveSharedPtr<Expression>& lhs, double rhs) {
  if (rhs == 0.0) {
    return lhs;
  }

  return lhs + MakeConstant(rhs);
}

WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator+(
    const wpi::IntrusiveSharedPtr<Expression>& lhs,
    const wpi::IntrusiveSharedPtr<Expression>& rhs) {
  if (lhs == nullptr) {
    return rhs;
  } else if (rhs == nullptr) {
    return lhs;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      ExpressionType{
          std::max(static_cast<int>(lhs->type), static_cast<int>(rhs->type))},
      [](double lhs, double rhs) { return lhs + rhs; },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint;
      },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint;
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& lhs,
         const wpi::IntrusiveSharedPtr<Expression>& rhs,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint;
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& lhs,
         const wpi::IntrusiveSharedPtr<Expression>& rhs,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint;
      },
      lhs, rhs);
}
WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator-(
    double lhs, const wpi::IntrusiveSharedPtr<Expression>& rhs) {
  if (lhs == 0.0) {
    return -rhs;
  }

  return MakeConstant(lhs) - rhs;
}

WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator-(
    const wpi::IntrusiveSharedPtr<Expression>& lhs, double rhs) {
  if (rhs == 0.0) {
    return lhs;
  }

  return lhs - MakeConstant(rhs);
}

WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator-(
    const wpi::IntrusiveSharedPtr<Expression>& lhs,
    const wpi::IntrusiveSharedPtr<Expression>& rhs) {
  if (lhs == nullptr) {
    if (rhs != nullptr) {
      return -rhs;
    } else {
      return nullptr;
    }
  } else if (rhs == nullptr) {
    return lhs;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      ExpressionType{
          std::max(static_cast<int>(lhs->type), static_cast<int>(rhs->type))},
      [](double lhs, double rhs) { return lhs - rhs; },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint;
      },
      [](double lhs, double rhs, double parentAdjoint) {
        return -parentAdjoint;
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& lhs,
         const wpi::IntrusiveSharedPtr<Expression>& rhs,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint;
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& lhs,
         const wpi::IntrusiveSharedPtr<Expression>& rhs,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return -parentAdjoint;
      },
      lhs, rhs);
}

WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator-(
    const wpi::IntrusiveSharedPtr<Expression>& lhs) {
  if (lhs == nullptr) {
    return nullptr;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      lhs->type, [](double lhs, double) { return -lhs; },
      [](double lhs, double, double parentAdjoint) { return -parentAdjoint; },
      [](const wpi::IntrusiveSharedPtr<Expression>& lhs,
         const wpi::IntrusiveSharedPtr<Expression>& rhs,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return -parentAdjoint;
      },
      lhs);
}

WPILIB_DLLEXPORT wpi::IntrusiveSharedPtr<Expression> operator+(
    const wpi::IntrusiveSharedPtr<Expression>& lhs) {
  if (lhs == nullptr) {
    return nullptr;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      lhs->type, [](double lhs, double) { return lhs; },
      [](double lhs, double, double parentAdjoint) { return parentAdjoint; },
      [](const wpi::IntrusiveSharedPtr<Expression>& lhs,
         const wpi::IntrusiveSharedPtr<Expression>& rhs,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint;
      },
      lhs);
}

void Expression::Update() {
  if (args[0] != nullptr) {
    auto& lhs = args[0];
    lhs->Update();

    if (args[1] == nullptr) {
      value = valueFunc(lhs->value, 0.0);
    } else {
      auto& rhs = args[1];
      rhs->Update();

      value = valueFunc(lhs->value, rhs->value);
    }
  }
}

wpi::IntrusiveSharedPtr<Expression> MakeConstant(double x) {
  return wpi::MakeIntrusiveShared<Expression>(x, ExpressionType::kConstant);
}

wpi::IntrusiveSharedPtr<Expression> abs(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x) {
  if (x == nullptr) {
    return nullptr;
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      type, [](double x, double) { return std::abs(x); },
      [](double x, double, double parentAdjoint) {
        if (x < 0.0) {
          return -parentAdjoint;
        } else if (x > 0.0) {
          return parentAdjoint;
        } else {
          return 0.0;
        }
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& x,
         const wpi::IntrusiveSharedPtr<Expression>&,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        if (x->value < 0.0) {
          return -parentAdjoint;
        } else if (x->value > 0.0) {
          return parentAdjoint;
        } else {
          return wpi::IntrusiveSharedPtr<Expression>{};
        }
      },
      x);
}

wpi::IntrusiveSharedPtr<Expression> acos(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x) {
  if (x == nullptr) {
    return MakeConstant(wpi::numbers::pi / 2.0);
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      type, [](double x, double) { return std::acos(x); },
      [](double x, double, double parentAdjoint) {
        return -parentAdjoint / std::sqrt(1.0 - x * x);
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& x,
         const wpi::IntrusiveSharedPtr<Expression>&,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return -parentAdjoint / autodiff::sqrt(1.0 - x * x);
      },
      x);
}

wpi::IntrusiveSharedPtr<Expression> asin(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x) {
  if (x == nullptr) {
    return nullptr;
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      type, [](double x, double) { return std::asin(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / std::sqrt(1.0 - x * x);
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& x,
         const wpi::IntrusiveSharedPtr<Expression>&,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint / autodiff::sqrt(1.0 - x * x);
      },
      x);
}

wpi::IntrusiveSharedPtr<Expression> atan(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x) {
  if (x == nullptr) {
    return nullptr;
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      type, [](double x, double) { return std::atan(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / (1.0 + x * x);
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& x,
         const wpi::IntrusiveSharedPtr<Expression>&,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint / (1.0 + x * x);
      },
      x);
}

wpi::IntrusiveSharedPtr<Expression> atan2(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& y,
    const wpi::IntrusiveSharedPtr<Expression>& x) {
  if (y == nullptr) {
    return nullptr;
  } else if (x == nullptr) {
    return MakeConstant(wpi::numbers::pi / 2.0);
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (y->type == ExpressionType::kConstant &&
      x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      type, [](double y, double x) { return std::atan2(y, x); },
      [](double y, double x, double parentAdjoint) {
        return parentAdjoint * x / (y * y + x * x);
      },
      [](double y, double x, double parentAdjoint) {
        return parentAdjoint * -y / (y * y + x * x);
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& y,
         const wpi::IntrusiveSharedPtr<Expression>& x,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * x / (y * y + x * x);
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& y,
         const wpi::IntrusiveSharedPtr<Expression>& x,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * -y / (y * y + x * x);
      },
      y, x);
}

wpi::IntrusiveSharedPtr<Expression> cos(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x) {
  if (x == nullptr) {
    return MakeConstant(1.0);
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      type, [](double x, double) { return std::cos(x); },
      [](double x, double, double parentAdjoint) {
        return -parentAdjoint * std::sin(x);
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& x,
         const wpi::IntrusiveSharedPtr<Expression>&,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * -autodiff::sin(x);
      },
      x);
}

wpi::IntrusiveSharedPtr<Expression> cosh(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x) {
  if (x == nullptr) {
    return MakeConstant(1.0);
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      type, [](double x, double) { return std::cosh(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint * std::sinh(x);
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& x,
         const wpi::IntrusiveSharedPtr<Expression>&,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * autodiff::sinh(x);
      },
      x);
}

wpi::IntrusiveSharedPtr<Expression> erf(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x) {
  static constexpr double sqrt_pi =
      1.7724538509055160272981674833411451872554456638435L;

  if (x == nullptr) {
    return nullptr;
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      type, [](double x, double) { return std::erf(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint * 2.0 / sqrt_pi * std::exp(-x * x);
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& x,
         const wpi::IntrusiveSharedPtr<Expression>&,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * 2.0 / sqrt_pi * autodiff::exp(-x * x);
      },
      x);
}

wpi::IntrusiveSharedPtr<Expression> exp(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x) {
  if (x == nullptr) {
    return MakeConstant(1.0);
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      type, [](double x, double) { return std::exp(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint * std::exp(x);
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& x,
         const wpi::IntrusiveSharedPtr<Expression>&,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * autodiff::exp(x);
      },
      x);
}

wpi::IntrusiveSharedPtr<Expression> hypot(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x,
    const wpi::IntrusiveSharedPtr<Expression>& y) {
  if (x == nullptr && y == nullptr) {
    return nullptr;
  }

  if (x == nullptr && y != nullptr) {
    // Evaluate the expression's type
    ExpressionType type;
    if (y->type == ExpressionType::kConstant) {
      type = ExpressionType::kConstant;
    } else {
      type = ExpressionType::kNonlinear;
    }

    return wpi::MakeIntrusiveShared<Expression>(
        type, [](double x, double y) { return std::hypot(x, y); },
        [](double x, double y, double parentAdjoint) {
          return parentAdjoint * x / std::hypot(x, y);
        },
        [](double x, double y, double parentAdjoint) {
          return parentAdjoint * y / std::hypot(x, y);
        },
        [](const wpi::IntrusiveSharedPtr<Expression>& x,
           const wpi::IntrusiveSharedPtr<Expression>& y,
           const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
          return parentAdjoint * x / autodiff::hypot(x, y);
        },
        [](const wpi::IntrusiveSharedPtr<Expression>& x,
           const wpi::IntrusiveSharedPtr<Expression>& y,
           const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
          return parentAdjoint * y / autodiff::hypot(x, y);
        },
        MakeConstant(0.0), y);
  } else if (x != nullptr && y == nullptr) {
    // Evaluate the expression's type
    ExpressionType type;
    if (x->type == ExpressionType::kConstant) {
      type = ExpressionType::kConstant;
    } else {
      type = ExpressionType::kNonlinear;
    }

    return wpi::MakeIntrusiveShared<Expression>(
        type, [](double x, double y) { return std::hypot(x, y); },
        [](double x, double y, double parentAdjoint) {
          return parentAdjoint * x / std::hypot(x, y);
        },
        [](double x, double y, double parentAdjoint) {
          return parentAdjoint * y / std::hypot(x, y);
        },
        [](const wpi::IntrusiveSharedPtr<Expression>& x,
           const wpi::IntrusiveSharedPtr<Expression>& y,
           const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
          return parentAdjoint * x / autodiff::hypot(x, y);
        },
        [](const wpi::IntrusiveSharedPtr<Expression>& x,
           const wpi::IntrusiveSharedPtr<Expression>& y,
           const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
          return parentAdjoint * y / autodiff::hypot(x, y);
        },
        x, MakeConstant(0.0));
  } else {
    // Evaluate the expression's type
    ExpressionType type;
    if (x->type == ExpressionType::kConstant &&
        y->type == ExpressionType::kConstant) {
      type = ExpressionType::kConstant;
    } else {
      type = ExpressionType::kNonlinear;
    }

    return wpi::MakeIntrusiveShared<Expression>(
        type, [](double x, double y) { return std::hypot(x, y); },
        [](double x, double y, double parentAdjoint) {
          return parentAdjoint * x / std::hypot(x, y);
        },
        [](double x, double y, double parentAdjoint) {
          return parentAdjoint * y / std::hypot(x, y);
        },
        [](const wpi::IntrusiveSharedPtr<Expression>& x,
           const wpi::IntrusiveSharedPtr<Expression>& y,
           const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
          return parentAdjoint * x / autodiff::hypot(x, y);
        },
        [](const wpi::IntrusiveSharedPtr<Expression>& x,
           const wpi::IntrusiveSharedPtr<Expression>& y,
           const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
          return parentAdjoint * y / autodiff::hypot(x, y);
        },
        x, y);
  }
}

wpi::IntrusiveSharedPtr<Expression> log(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x) {
  if (x == nullptr) {
    return nullptr;
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      type, [](double x, double) { return std::log(x); },
      [](double x, double, double parentAdjoint) { return parentAdjoint / x; },
      [](const wpi::IntrusiveSharedPtr<Expression>& x,
         const wpi::IntrusiveSharedPtr<Expression>&,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint / x;
      },
      x);
}

wpi::IntrusiveSharedPtr<Expression> log10(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x) {
  static constexpr double ln10 = 2.3025850929940456840179914546843L;

  if (x == nullptr) {
    return nullptr;
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      type, [](double x, double) { return std::log10(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / (ln10 * x);
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& x,
         const wpi::IntrusiveSharedPtr<Expression>&,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint / (ln10 * x);
      },
      x);
}

wpi::IntrusiveSharedPtr<Expression> pow(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& base,
    const wpi::IntrusiveSharedPtr<Expression>& power) {
  if (base == nullptr) {
    return nullptr;
  }
  if (power == nullptr) {
    return MakeConstant(1.0);
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (base->type == ExpressionType::kConstant &&
      power->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else if (power->type == ExpressionType::kConstant && power->value == 0.0) {
    type = ExpressionType::kConstant;
  } else if (base->type == ExpressionType::kLinear &&
             power->type == ExpressionType::kConstant && power->value == 1.0) {
    type = ExpressionType::kLinear;
  } else if (base->type == ExpressionType::kLinear &&
             power->type == ExpressionType::kConstant && power->value == 2.0) {
    type = ExpressionType::kQuadratic;
  } else if (base->type == ExpressionType::kQuadratic &&
             power->type == ExpressionType::kConstant && power->value == 1.0) {
    type = ExpressionType::kQuadratic;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      type, [](double base, double power) { return std::pow(base, power); },
      [](double base, double power, double parentAdjoint) {
        return parentAdjoint * std::pow(base, power - 1) * power;
      },
      [](double base, double power, double parentAdjoint) {
        // Since x * std::log(x) -> 0 as x -> 0
        if (base == 0.0) {
          return 0.0;
        } else {
          return parentAdjoint * std::pow(base, power - 1) * base *
                 std::log(base);
        }
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& base,
         const wpi::IntrusiveSharedPtr<Expression>& power,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * autodiff::pow(base, power - 1) * power;
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& base,
         const wpi::IntrusiveSharedPtr<Expression>& power,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        // Since x * std::log(x) -> 0 as x -> 0
        if (base->value == 0.0) {
          return wpi::IntrusiveSharedPtr<Expression>{};
        } else {
          return parentAdjoint * autodiff::pow(base, power - 1) * base *
                 autodiff::log(base);
        }
      },
      base, power);
}

wpi::IntrusiveSharedPtr<Expression> sin(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x) {
  if (x == nullptr) {
    return nullptr;
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      type, [](double x, double) { return std::sin(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint * std::cos(x);
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& x,
         const wpi::IntrusiveSharedPtr<Expression>&,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * autodiff::cos(x);
      },
      x);
}

wpi::IntrusiveSharedPtr<Expression> sinh(
    const wpi::IntrusiveSharedPtr<Expression>& x) {
  if (x == nullptr) {
    return nullptr;
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      type, [](double x, double) { return std::sinh(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint * std::cosh(x);
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& x,
         const wpi::IntrusiveSharedPtr<Expression>&,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * autodiff::cosh(x);
      },
      x);
}

wpi::IntrusiveSharedPtr<Expression> sqrt(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x) {
  if (x == nullptr) {
    return nullptr;
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      type, [](double x, double) { return std::sqrt(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / (2.0 * std::sqrt(x));
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& x,
         const wpi::IntrusiveSharedPtr<Expression>&,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint / (2.0 * autodiff::sqrt(x));
      },
      x);
}

wpi::IntrusiveSharedPtr<Expression> tan(  // NOLINT
    const wpi::IntrusiveSharedPtr<Expression>& x) {
  if (x == nullptr) {
    return nullptr;
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      type, [](double x, double) { return std::tan(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / (std::cos(x) * std::cos(x));
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& x,
         const wpi::IntrusiveSharedPtr<Expression>&,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint / (autodiff::cos(x) * autodiff::cos(x));
      },
      x);
}

wpi::IntrusiveSharedPtr<Expression> tanh(
    const wpi::IntrusiveSharedPtr<Expression>& x) {
  if (x == nullptr) {
    return nullptr;
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return wpi::MakeIntrusiveShared<Expression>(
      type, [](double x, double) { return std::tanh(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / (std::cosh(x) * std::cosh(x));
      },
      [](const wpi::IntrusiveSharedPtr<Expression>& x,
         const wpi::IntrusiveSharedPtr<Expression>&,
         const wpi::IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint / (autodiff::cosh(x) * autodiff::cosh(x));
      },
      x);
}

}  // namespace frc::autodiff
