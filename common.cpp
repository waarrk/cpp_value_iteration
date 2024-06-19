#include "common.hpp"

bool is_within_bounds(int x, int y, int size) {
  return (x >= 0 && x < size && y >= 0 && y < size);
}