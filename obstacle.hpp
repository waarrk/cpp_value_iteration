#ifndef OBSTACLE_HPP
#define OBSTACLE_HPP

#include "common.hpp"

// 水たまりの報酬を設定する関数
void set_puddle(Matrix2D& rewards, int size);

// 障害物の報酬を設定する関数
void set_obstacles(Matrix2D& rewards, int size);

#endif