#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "common.hpp"

typedef std::vector<std::vector<double>> Matrix2D;
typedef std::vector<std::vector<std::vector<double>>> Matrix3D;

// 目標位置の価値を初期化する関数
void initialize_goal_values(Matrix3D& values, int size, int theta_size) {
  int goal_x = size - 11;
  int goal_y = size - 11;
  if (goal_x >= size || goal_y >= size)
    throw std::out_of_range("Goal position out of bounds");
  for (int t = 0; t < theta_size; ++t) {
    values[goal_x][goal_y][t] = 0.0;
  }
  std::cout << "Goal Init: (" << goal_x << ", " << goal_y << ")" << std::endl;
}

// 報酬と価値の配列を初期化する関数
void initialize_arrays(Matrix2D& rewards, Matrix3D& values, int size,
                       int theta_size) {
  rewards.resize(size, std::vector<double>(size, -1.0));
  values.resize(size, std::vector<std::vector<double>>(
                          size, std::vector<double>(theta_size, -100.0)));
}

// 目標位置を設定する関数
void set_goal(Matrix2D& rewards, int size) {
  int goal_x = size - 11;
  int goal_y = size - 11;
  if (goal_x >= size || goal_y >= size)
    throw std::out_of_range("Goal position out of bounds");
  rewards[goal_x][goal_y] = 0.0;
}

// 境界の報酬を設定する関数
void set_boundaries(Matrix2D& rewards, int size) {
  for (int i = 0; i < size; ++i) {
    rewards[i][0] = -5.0;
    rewards[i][size - 1] = -100.0;
    rewards[0][i] = -5.0;
    rewards[size - 1][i] = -100.0;
  }
}

// 水たまりの報酬を設定する関数
void set_puddle(Matrix2D& rewards, int size) {
  int obstacle_end = 2 * size / 4;
  int puddle_start = obstacle_end;
  int puddle_end = obstacle_end + (obstacle_end - (size / 4)) / 2;
  for (int i = puddle_start; i < puddle_end; ++i) {
    for (int j = puddle_start; j < puddle_end; ++j) {
      if (i < size && j < size) {
        rewards[i][j] = -10.0;
      }
    }
  }
}

// 障害物の報酬を設定する関数
void set_obstacles(Matrix2D& rewards, int size) {
  int obstacle_start = size / 4;
  int obstacle_end = 2 * size / 4;
  for (int i = obstacle_start; i < obstacle_end; ++i) {
    for (int j = obstacle_start; j < obstacle_end; ++j) {
      rewards[i][j] = -20.0;
    }
  }
}

// 行動を生成する関数
std::vector<std::tuple<int, int, int>> generate_actions() {
  return {
      std::make_tuple(0, 1, 0),    // 右
      std::make_tuple(1, 0, 0),    // 下
      std::make_tuple(0, -1, 0),   // 左
      std::make_tuple(-1, 0, 0),   // 上
      std::make_tuple(1, 1, 0),    // 右下
      std::make_tuple(-1, 1, 0),   // 左下
      std::make_tuple(1, -1, 0),   // 右上
      std::make_tuple(-1, -1, 0),  // 左上
      std::make_tuple(0, 0, 1),    // 時計回り
      std::make_tuple(0, 0, -1)    // 反時計回り
  };
}

// 与えられた状態に対して価値を計算する関数
double calculate_value(int i, int j, int theta, double gamma,
                       const Matrix2D& rewards, const Matrix3D& values,
                       const std::vector<std::tuple<int, int, int>>& actions,
                       int size, int theta_size) {
  double value = -1e9;
  for (const auto& action : actions) {
    int di, dj, dtheta;
    std::tie(di, dj, dtheta) = action;
    int ni = i + di;
    int nj = j + dj;
    int ntheta = (theta + dtheta + theta_size) % theta_size;
    if (is_within_bounds(ni, nj, size)) {
      double cost_multiplier =
          (std::abs(di) == 1 && std::abs(dj) == 1) ? std::sqrt(2) : 1.0;
      double new_value =
          rewards[ni][nj] * cost_multiplier + gamma * values[ni][nj][ntheta];
      if (new_value > value) {
        value = new_value;
      }
    }
  }
  return value;
}

int main() {
  int size = 200;           // マップサイズ設定
  int theta_size = 8;       // 各位置で進める方向の数
  double threshold = 1e-9;  // 収束判定閾値

  Matrix2D rewards;
  Matrix3D values;

  // 配列の初期化
  initialize_arrays(rewards, values, size, theta_size);

  // 各種設定
  set_goal(rewards, size);        // 目標位置の設定
  set_boundaries(rewards, size);  // 境界の設定
  set_puddle(rewards, size);      // 水たまりの設定
  set_obstacles(rewards, size);   // 障害物の設定

  // 目標位置の価値を初期化
  initialize_goal_values(values, size, theta_size);

  // アクションの生成
  std::vector<std::tuple<int, int, int>> actions = generate_actions();

  // 値の反復計算のパラメータ設定
  double gamma = 1.0;
  int max_iterations = 1000;

  // 計算時間の測定開始
  auto start = std::chrono::high_resolution_clock::now();

  // 値の反復計算を実行
  for (int iter = 0; iter < max_iterations; ++iter) {
    Matrix3D new_values = values;
    double max_delta = 0.0;
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        for (int theta = 0; theta < theta_size; ++theta) {
          if (rewards[i][j] != 0.0) {
            double max_value = calculate_value(
                i, j, theta, gamma, rewards, values, actions, size, theta_size);
            new_values[i][j][theta] = max_value;
            max_delta =
                std::max(max_delta, std::abs(values[i][j][theta] - max_value));
          }
        }
      }
    }
    values = new_values;
    if (max_delta < threshold) {
      std::cout << "Converged after " << iter + 1
                << " iterations with max delta: " << max_delta << std::endl;
      break;
    }
  }

  // 計算時間の測定終了
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

  // 各グリッドにおいて最大の価値を計算して保存
  std::ofstream outFile("max_values.txt");
  if (outFile.is_open()) {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        double max_value = values[i][j][0];
        for (int theta = 1; theta < theta_size; ++theta) {
          if (values[i][j][theta] > max_value) {
            max_value = values[i][j][theta];
          }
        }
        outFile << max_value << " ";
      }
      outFile << std::endl;
    }
    outFile.close();
  } else {
    std::cerr << "File cannot open" << std::endl;
  }

  std::cout << "Value Iteration Complete !!!" << std::endl;
  return 0;
}