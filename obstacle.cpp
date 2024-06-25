#include "obstacle.hpp"

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