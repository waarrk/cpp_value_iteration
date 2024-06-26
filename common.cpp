﻿#include "common.hpp"

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