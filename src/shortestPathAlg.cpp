#include <iostream>
#include <cmath>
#include <vector>
#include <limits>
#include <numeric>
#include <map>
#include <set>
#include <queue>
#include <tuple>
#include <algorithm>
#include <string>
#include <sstream>
#include <chrono>

#include "shortestPathAlg.h"


Point arrToPoint(const std::array<double, 3> arr) {
  Point p; 
  p.x=arr[0]; 
  p.y=arr[1]; 
  p.z=arr[2]; 
  return p;
}

void add_node(const Point &p, std::map<Point, std::vector<Edge>> graph) {
  if (graph.find(p)==graph.end()) graph[p]=std::vector<Edge>();
}

void add_edge(const Point &u, const Point &v, double w, std::map<Point, std::vector<Edge>> graph) {
  graph[u].push_back({v,w});
  graph[v].push_back({u,w});
}


std::string fmtPoint(const Point &p) {
  std::ostringstream oss;
  oss << "(" << p.x << ", " << p.y << ", " << p.z << ")";
  return oss.str();
}


std::string fmtArray(const Point &p) {
  std::ostringstream oss;
  oss << "array([" << p.x << ", " << p.y << ", " << p.z << "])";
  return oss.str();
}

std::string fmtEdgeAsArrays(const Point &p1, const Point &p2) {
  std::ostringstream oss;
  oss << "(" << fmtArray(p1) << ", " << fmtArray(p2) << ")";
  return oss.str();
}

// Distance in 2D for weights
double distance2D(const Point &a, const Point &b) {
  double dx=a.x-b.x; double dy=a.y-b.y;
  return std::sqrt(dx*dx+dy*dy);
}

// Distance in 3D for heuristic
double distance3D(const Point &a, const Point &b) {
  double dx=a.x-b.x; double dy=a.y-b.y; double dz=a.z-b.z;
  return std::sqrt(dx*dx+dy*dy+dz*dz);
}

// Use GEOS to check intersection
bool segments_intersect_no_touches_geos(const Point &A, const Point &B, const Point &C, const Point &D, GEOSContextHandle_t geos_ctx) {
  // Create line segment for AB
  GEOSCoordSequence* seq1 = GEOSCoordSeq_create_r(geos_ctx, 2, 2);
  GEOSCoordSeq_setXY_r(geos_ctx, seq1, 0, A.x, A.y);
  GEOSCoordSeq_setXY_r(geos_ctx, seq1, 1, B.x, B.y);
  GEOSGeometry* line1 = GEOSGeom_createLineString_r(geos_ctx, seq1);

  // Create line segment for CD
  GEOSCoordSequence* seq2 = GEOSCoordSeq_create_r(geos_ctx, 2, 2);
  GEOSCoordSeq_setXY_r(geos_ctx, seq2, 0, C.x, C.y);
  GEOSCoordSeq_setXY_r(geos_ctx, seq2, 1, D.x, D.y);
  GEOSGeometry* line2 = GEOSGeom_createLineString_r(geos_ctx, seq2);

  char intersects = GEOSIntersects_r(geos_ctx, line1, line2);
  char touches = GEOSTouches_r(geos_ctx, line1, line2);

  bool result = false;
  if (intersects && !touches) {
      result = true;
  }

  GEOSGeom_destroy_r(geos_ctx, line1);
  GEOSGeom_destroy_r(geos_ctx, line2);

  return result;
}

// Connect consecutive convex hull points.
std::vector<std::pair<Point,Point>> add_outer_edges_cube(int num_rows, std::vector< std::array<double, 3> > cube_vertices, std::map<Point, std::vector<Edge>> graph) {
  Point vs[num_rows];
  for (int i=0;i<num_rows;i++) vs[i]=arrToPoint(cube_vertices[i]);
  std::vector<std::pair<Point,Point>> edges;
  for (int i=0; i<num_rows; i++) {
      int j = (i + 1) % num_rows;
      edges.push_back({vs[i], vs[j]});
      double w = distance2D(vs[i], vs[j]);
      add_edge(vs[i], vs[j], w, graph);
  }
  return edges;
}

void add_edges_without_intersection(const Point &point, const std::vector<std::pair<Point,Point>> &cube_edges, int num_rows, std::vector< std::array<double, 3> > cube_vertices, GEOSContextHandle_t geos_ctx, std::map<Point, std::vector<Edge>> graph) {
  Point vs[num_rows];
  for (int i=0;i<num_rows;i++) vs[i]=arrToPoint(cube_vertices[i]);

  for (int i=0;i<num_rows;i++) {
      Point vertex = vs[i];
      bool intersects = false;
      for (auto &ce: cube_edges) {
          if (segments_intersect_no_touches_geos(point, vertex, ce.first, ce.second, geos_ctx)) {
              //std::cout << "Edge from " << fmtPoint(point) << " to " << fmtPoint(vertex)
              //          << " intersects with cube edge " << fmtEdgeAsArrays(ce.first, ce.second) << "\n";
              intersects = true;
              break;
          }
      }
      if (!intersects) {
          double w = distance2D(point, vertex);
          add_edge(point, vertex, w, graph);
          //std::cout << "Added edge from " << fmtPoint(point) << " to " << fmtPoint(vertex) << "\n";
      }
  }
}

std::vector<Point> astar_path(const Point &start, const Point &goal, std::map<Point, std::vector<Edge>> graph) {
  std::map<Point,double> gScore;
  std::map<Point,double> fScore;
  std::map<Point,Point> cameFrom;

  for (auto &kv: graph) {
      gScore[kv.first] = std::numeric_limits<double>::infinity();
      fScore[kv.first] = std::numeric_limits<double>::infinity();
  }

  gScore[start] = 0.0;
  fScore[start] = distance3D(start, goal);

 
  static long long counter = 0;

  struct PQItem {
      Point node;
      double f;
      long long order;
      bool operator>(const PQItem &o) const {
          if (f == o.f) return order > o.order;
          return f > o.f;
      }
  };

  std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> openSet;
  openSet.push({start, fScore[start], counter++});
  std::map<Point, bool> inOpen;
  inOpen[start] = true;

  while (!openSet.empty()) {
      Point current = openSet.top().node;
      openSet.pop();
      inOpen[current] = false;

      if (current == goal) {
          std::vector<Point> path;
          Point cur = current;
          while (!(cur == start)) {
              path.push_back(cur);
              cur = cameFrom[cur];
          }
          path.push_back(start);
          std::reverse(path.begin(), path.end());
          return path;
      }

      for (auto &edge: graph[current]) {
          Point neighbor = edge.node;
          double tentative = gScore[current] + edge.weight;
          if (tentative < gScore[neighbor]) {
              cameFrom[neighbor] = current;
              gScore[neighbor] = tentative;
              fScore[neighbor] = tentative + distance3D(neighbor, goal);
              if (!inOpen[neighbor]) {
                  openSet.push({neighbor, fScore[neighbor], counter++});
                  inOpen[neighbor] = true;
              }
          }
      }
  }

  return {};
}
