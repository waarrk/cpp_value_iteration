#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
#include <vector>

typedef std::vector<std::vector<double>> Matrix2D;
typedef std::vector<std::vector<std::vector<double>>> Matrix3D;

struct Action {
  int di;
  int dj;
  int dtheta;
};

// 座標が範囲内にあるかどうかを確認する関数
bool is_within_bounds(int x, int y, int size);

// 目標位置の価値を初期化する関数
void initialize_goal_values(Matrix3D& values, int size, int theta_size);

// 報酬と価値の配列を初期化する関数
void initialize_arrays(Matrix2D& rewards, Matrix3D& values, int size,
                       int theta_size);

// 目標位置を設定する関数
void set_goal(Matrix2D& rewards, int size);

// 境界の報酬を設定する関数
void set_boundaries(Matrix2D& rewards, int size);

// 行動を生成する関数
std::vector<Action> generate_actions();

#endif
