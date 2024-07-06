#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "common.hpp"
#include "obstacle.hpp"

// 与えられた状態に対して価値を計算する関数
double calculate_value(int i, int j, int theta, double gamma,
                       const Matrix2D& rewards, const Matrix3D& values,
                       const std::vector<Action>& actions, int size,
                       int theta_size) {
  double value = -1e9;
  for (const auto& action : actions) {
    int di = action.di;
    int dj = action.dj;
    int dtheta = action.dtheta;
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

// 各グリッドにおいて最大の価値を計算して保存する関数
void save_results(const std::string& filename, const Matrix3D& values, int size,
                  int theta_size) {
  std::ofstream outFile(filename);
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
}

// 並列価値反復
void parallel_value_iteration(int start, int end, int size, int theta_size,
                              double gamma, double threshold,
                              const Matrix2D& rewards, Matrix3D& values,
                              const std::vector<Action>& actions,
                              double& max_delta) {
  for (int i = start; i < end; ++i) {
    for (int j = 0; j < size; ++j) {
      for (int theta = 0; theta < theta_size; ++theta) {
        if (rewards[i][j] != 0.0) {
          double max_value = calculate_value(i, j, theta, gamma, rewards,
                                             values, actions, size, theta_size);
          double delta = std::abs(values[i][j][theta] - max_value);
          values[i][j][theta] = max_value;
          if (delta > max_delta) {
            max_delta = delta;
          }
        }
      }
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " size num_threads" << std::endl;
    return 1;
  }

  int size = std::atoi(argv[1]);
  int num_threads = std::atoi(argv[2]);

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
  std::vector<Action> actions = generate_actions();

  // 値の反復計算のパラメータ設定
  double gamma = 1.0;
  int max_iterations = 100000;

  std::cout << "Using " << num_threads << " threads" << std::endl;

  // 計算時間の測定開始
  auto start = std::chrono::high_resolution_clock::now();

  // 値の反復計算を実行
  int chunk_size = size / num_threads;

  for (int iter = 0; iter < max_iterations; ++iter) {
    Matrix3D new_values = values;
    double max_delta = 0.0;
    std::vector<std::thread> threads;
    std::vector<double> deltas(num_threads, 0.0);

    // 並列計算
    for (int t = 0; t < num_threads; ++t) {
      int start = t * chunk_size;
      int end = (t == num_threads - 1) ? size : start + chunk_size;
      threads.emplace_back(parallel_value_iteration, start, end, size,
                           theta_size, gamma, threshold, std::cref(rewards),
                           std::ref(new_values), std::cref(actions),
                           std::ref(deltas[t]));
    }

    // 同期処理
    for (auto& th : threads) {
      th.join();
    }

    // 収束判定
    for (const auto& delta : deltas) {
      max_delta = std::max(max_delta, delta);
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
  save_results("max_values.txt", values, size, theta_size);

  std::cout << "Value Iteration Complete !!!" << std::endl;
  return 0;
}
