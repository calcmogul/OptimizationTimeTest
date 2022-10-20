// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#pragma once

#include <cstddef>

namespace frc::autodiff {

class Indexer {
 private:
  static inline size_t index = 0u;

 public:
  static size_t GetIndex();
};

}  // namespace frc::autodiff
