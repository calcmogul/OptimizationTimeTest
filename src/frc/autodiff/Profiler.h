// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#pragma once

#include <chrono>

namespace frc::autodiff {

/**
 * Records the number of profiler measurements (start/stop pairs) and the
 * average duration between each start and stop call.
 */
class Profiler {
 public:
  /**
   * Tell the profiler to start measuring.
   */
  void Start() { m_startTime = std::chrono::system_clock::now(); }

  /**
   * Tell the profiler to stop measuring, increment the number of averages, and
   * incorporate the latest measurement into the average.
   */
  void Stop() {
    auto now = std::chrono::system_clock::now();
    ++m_measurements;
    m_averageDuration =
        (m_measurements - 1.0) / m_measurements * m_averageDuration +
        1.0 / m_measurements * (now - m_startTime);
  }

  /**
   * The number of measurements taken.
   */
  int Measurements() const { return m_measurements; }

  /**
   * The average duration in milliseconds as a double.
   */
  double AverageDuration() const {
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    return duration_cast<microseconds>(m_averageDuration).count() / 1000.0;
  }

 private:
  int m_measurements = 0;
  std::chrono::duration<double> m_averageDuration{0.0};
  std::chrono::system_clock::time_point m_startTime;
};

}  // namespace frc::autodiff
