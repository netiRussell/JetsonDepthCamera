#pragma once
#include <geos_c.h>

struct Point {
  double x,y,z;
};

inline bool operator<(const Point &a, const Point &b) {
  if (a.x != b.x) return a.x < b.x;
  if (a.y != b.y) return a.y < b.y;
  return a.z < b.z;
}

inline bool operator==(const Point &a, const Point &b) {
  return (a.x == b.x && a.y == b.y && a.z == b.z);
}

struct Edge {
  Point node;
  double weight;
};

void add_edges_without_intersection(const Point &point, const std::vector<std::pair<Point,Point>> &cube_edges, int num_rows, std::vector< std::array<double, 3> > cube_vertices, GEOSContextHandle_t geos_ctx, std::map<Point, std::vector<Edge>>& graph);
std::vector<std::pair<Point,Point>> add_outer_edges_cube(int num_rows, std::vector< std::array<double, 3> > cube_vertices, std::map<Point, std::vector<Edge>>& graph);
bool segments_intersect_no_touches_geos(const Point &A, const Point &B, const Point &C, const Point &D, GEOSContextHandle_t geos_ctx);
double distance3D(const Point &a, const Point &b);
double distance2D(const Point &a, const Point &b);
Point arrToPoint(const std::array<double, 3> arr);
void add_node(const Point &p, std::map<Point, std::vector<Edge>>& graph);
void add_edge(const Point &u, const Point &v, double w, std::map<Point, std::vector<Edge>>& graph);
std::string fmtPoint(const Point &p);
std::string fmtArray(const Point &p);
std::string fmtEdgeAsArrays(const Point &p1, const Point &p2);
std::vector<Point> astar_path(const Point &start, const Point &goal, std::map<Point, std::vector<Edge>>& graph);
void add_edges_without_intersection(const Point &point, const std::vector<std::pair<Point,Point>> &cube_edges, int num_rows, std::vector< std::array<double, 3> > cube_vertices, GEOSContextHandle_t geos_ctx, std::map<Point, std::vector<Edge>>& graph);
