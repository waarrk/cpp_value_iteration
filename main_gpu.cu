#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

typedef std::vector<std::vector<double>> Matrix2D;
typedef std::vector<std::vector<std::vector<double>>> Matrix3D;

// CUDA用のデバイスメモリポインタ
double* d_rewards;
double* d_values;
double* d_new_values;

struct Action {
  int di;
  int dj;
  int dtheta;
};

Action* d_actions;

// 座標が範囲内にあるかどうかを確認する関数
bool is_within_bounds(int x, int y, int size) {
  return (x >= 0 && x < size && y >= 0 && y < size);
}

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
std::vector<Action> generate_actions() {
  return {
      {0, 1, 0},    // 右
      {1, 0, 0},    // 下
      {0, -1, 0},   // 左
      {-1, 0, 0},   // 上
      {1, 1, 0},    // 右下
      {-1, 1, 0},   // 左下
      {1, -1, 0},   // 右上
      {-1, -1, 0},  // 左上
      {0, 0, 1},    // 時計回り
      {0, 0, -1}    // 反時計回り
  };
}

// CUDAカーネル関数
__global__ void calculate_value_kernel(double* d_rewards, double* d_values,
                                       double* d_new_values, Action* d_actions,
                                       int size, int theta_size, double gamma,
                                       int num_actions) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int theta = blockIdx.z * blockDim.z + threadIdx.z;
  if (i < size && j < size && theta < theta_size) {
    double max_value = -1e9;
    for (int k = 0; k < num_actions; ++k) {
      int di = d_actions[k].di;
      int dj = d_actions[k].dj;
      int dtheta = d_actions[k].dtheta;
      int ni = i + di;
      int nj = j + dj;
      int ntheta = (theta + dtheta + theta_size) % theta_size;
      if (ni >= 0 && ni < size && nj >= 0 && nj < size) {
        double cost_multiplier =
            (abs(di) == 1 && abs(dj) == 1) ? sqrt(2.0) : 1.0;
        double new_value =
            d_rewards[ni * size + nj] * cost_multiplier +
            gamma * d_values[(ni * size + nj) * theta_size + ntheta];
        if (new_value > max_value) {
          max_value = new_value;
        }
      }
    }
    d_new_values[(i * size + j) * theta_size + theta] = max_value;
  }
}

void initialize_cuda_memory(const Matrix2D& rewards, const Matrix3D& values,
                            const std::vector<Action>& actions, int size,
                            int theta_size) {
  int num_elements = size * size * theta_size;
  int reward_elements = size * size;

  cudaMalloc(&d_rewards, reward_elements * sizeof(double));
  cudaMalloc(&d_values, num_elements * sizeof(double));
  cudaMalloc(&d_new_values, num_elements * sizeof(double));
  cudaMalloc(&d_actions, actions.size() * sizeof(Action));

  std::vector<double> h_rewards(reward_elements);
  std::vector<double> h_values(num_elements);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      h_rewards[i * size + j] = rewards[i][j];
      for (int theta = 0; theta < theta_size; ++theta) {
        h_values[(i * size + j) * theta_size + theta] = values[i][j][theta];
      }
    }
  }

  cudaMemcpy(d_rewards, h_rewards.data(), reward_elements * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, h_values.data(), num_elements * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_actions, actions.data(), actions.size() * sizeof(Action),
             cudaMemcpyHostToDevice);
}

void execute_value_iteration(int size, int theta_size, double gamma,
                             int max_iterations, double threshold,
                             const std::vector<Action>& actions) {
  dim3 blockDim(8, 8, 8);
  dim3 gridDim((size + blockDim.x - 1) / blockDim.x,
               (size + blockDim.y - 1) / blockDim.y,
               (theta_size + blockDim.z - 1) / blockDim.z);

  std::vector<double> h_values(size * size * theta_size);
  std::vector<double> h_new_values(size * size * theta_size);

  for (int iter = 0; iter < max_iterations; ++iter) {
    calculate_value_kernel<<<gridDim, blockDim>>>(
        d_rewards, d_values, d_new_values, d_actions, size, theta_size, gamma,
        actions.size());
    cudaDeviceSynchronize();

    cudaMemcpy(h_values.data(), d_values,
               size * size * theta_size * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_new_values.data(), d_new_values,
               size * size * theta_size * sizeof(double),
               cudaMemcpyDeviceToHost);

    double max_delta = 0.0;
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        for (int theta = 0; theta < theta_size; ++theta) {
          int idx = (i * size + j) * theta_size + theta;
          max_delta =
              std::max(max_delta, std::abs(h_values[idx] - h_new_values[idx]));
        }
      }
    }

    if (max_delta < threshold) {
      std::cout << "Converged after " << iter + 1
                << " iterations with max delta: " << max_delta << std::endl;
      break;
    }

    cudaMemcpy(d_values, h_new_values.data(),
               size * size * theta_size * sizeof(double),
               cudaMemcpyHostToDevice);
  }
}

// GPUの情報を表示する関数
void print_gpu_info() {
  int device_count;
  cudaGetDeviceCount(&device_count);

  if (device_count == 0) {
    std::cout << "No CUDA-compatible GPU detected." << std::endl;
    return;
  }

  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, i);

    std::cout << "GPU " << i << ": " << device_prop.name << std::endl;
    std::cout << "CUDA Cores: " << device_prop.multiProcessorCount * 128
              << std::endl;
    std::cout << "Clock Rate: " << device_prop.clockRate / 1000 << " MHz"
              << std::endl;
    std::cout << "Global Memory: " << device_prop.totalGlobalMem / (1 << 20)
              << " MB" << std::endl;
  }
}

void save_results(const std::string& filename, int size, int theta_size) {
  Matrix3D values(size, std::vector<std::vector<double>>(
                            size, std::vector<double>(theta_size)));

  std::vector<double> h_values(size * size * theta_size);
  cudaMemcpy(h_values.data(), d_values,
             size * size * theta_size * sizeof(double), cudaMemcpyDeviceToHost);

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      for (int theta = 0; theta < theta_size; ++theta) {
        values[i][j][theta] = h_values[(i * size + j) * theta_size + theta];
      }
    }
  }

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

void cleanup_cuda_memory() {
  cudaFree(d_rewards);
  cudaFree(d_values);
  cudaFree(d_new_values);
  cudaFree(d_actions);
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
  std::vector<Action> actions = generate_actions();

  // CUDAメモリの初期化
  initialize_cuda_memory(rewards, values, actions, size, theta_size);

  // GPUの情報を表示
  print_gpu_info();

  // 計算時間の測定開始
  auto start = std::chrono::high_resolution_clock::now();

  // 値の反復計算を実行
  execute_value_iteration(size, theta_size, 1.0, 1000, threshold, actions);

  // 計算時間の測定終了
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

  // 各グリッドにおいて最大の価値を計算して保存
  save_results("max_values.txt", size, theta_size);

  // CUDAメモリのクリーンアップ
  cleanup_cuda_memory();

  std::cout << "Value Iteration Complete !!!" << std::endl;
  return 0;
}