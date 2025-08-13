#pragma once

// math.hpp - Minimal pedagogical SE(3) interpolation helpers
// -----------------------------------------------------------------------------
// This header provides lightweight vector, quaternion, rotation matrix, and
// SE(3) exponential / logarithm utilities used by the demo application.
// Design goals:
//  * Readability over abstraction (no templates, fixed double precision).
//  * Numerical safety (graceful fallbacks instead of assertions / UB).
//  * Header-only (all functions inline) for easy reuse in a single-file demo.
//  * No external dependencies; only <cmath>, <algorithm>, <utility>.
//
// Limitations / Non-goals:
//  * Not a full-featured linear algebra package; minimal operations only.
//  * No SIMD / advanced optimization; clarity preferred.
//  * Not constexpr beyond simple aggregates (C++17 target here).
//
// Numerical safety considerations implemented below:
//  * Robust handling of near-zero norms for vectors & quaternions (avoids division by ~0).
//  * Threshold-based Taylor series for SO(3) exp/log & Jacobians with documented cutoffs.
//  * Clamping of cosine argument for acos to [-1,1] to avoid NaNs from FP drift.
// -----------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <utility>

namespace math {

// Small constants
inline constexpr double kTiny = 1e-12; // generic tiny threshold
inline constexpr double kTinyAngle = 1e-8; // series expansion threshold
inline constexpr double kQuatNormEps = 1e-12; // quaternion norm safeguard

struct Vec3 {
  double x{}, y{}, z{};
  constexpr Vec3() noexcept = default;

  constexpr Vec3(double X, double Y, double Z) noexcept : x(X), y(Y), z(Z) {}

  constexpr Vec3 operator+(const Vec3& b) const noexcept { return {x + b.x, y + b.y, z + b.z}; }

  constexpr Vec3 operator-(const Vec3& b) const noexcept { return {x - b.x, y - b.y, z - b.z}; }

  constexpr Vec3 operator*(double s) const noexcept { return {x * s, y * s, z * s}; }

  constexpr Vec3& operator+=(const Vec3& b) noexcept {
    x += b.x;
    y += b.y;
    z += b.z;
    return *this;
  }
};

inline constexpr double dot(const Vec3& a, const Vec3& b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline constexpr Vec3 cross(const Vec3& a, const Vec3& b) noexcept {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline double norm(const Vec3& a) noexcept {
  return std::sqrt(std::max(1e-18, dot(a, a)));
}

inline Vec3 normalized(const Vec3& a) noexcept {
  const double n = norm(a);
  return (n < kTiny) ? Vec3{0, 0, 0} : Vec3{a.x / n, a.y / n, a.z / n};
}

struct Quat { // (x,y,z,w)
  double x{}, y{}, z{}, w{1};
  constexpr Quat() noexcept = default;

  constexpr Quat(double X, double Y, double Z, double W) noexcept : x(X), y(Y), z(Z), w(W) {}
};

inline Quat q_safe_normalize(const Quat& q) noexcept {
  const double n2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  if (n2 < kQuatNormEps) { return {0, 0, 0, 1}; }
  const double inv = 1.0 / std::sqrt(n2);
  return {q.x * inv, q.y * inv, q.z * inv, q.w * inv};
}

inline Quat q_normalized(const Quat& q) noexcept {
  return q_safe_normalize(q);
}

inline constexpr double q_dot(const Quat& a, const Quat& b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline constexpr Quat q_mul(const Quat& a, const Quat& b) noexcept {
  return {a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

inline constexpr Quat q_conj(const Quat& q) noexcept {
  return {-q.x, -q.y, -q.z, q.w};
}

inline Quat q_from_euler(double rx, double ry, double rz) noexcept { // XYZ intrinsic
  const double cx = std::cos(rx * 0.5), sx = std::sin(rx * 0.5);
  const double cy = std::cos(ry * 0.5), sy = std::sin(ry * 0.5);
  const double cz = std::cos(rz * 0.5), sz = std::sin(rz * 0.5);
  const Quat qx{sx, 0, 0, cx}, qy{0, sy, 0, cy}, qz{0, 0, sz, cz};
  return q_normalized(q_mul(q_mul(qx, qy), qz));
}

inline Quat q_slerp(Quat a, Quat b, double t) noexcept {
  a = q_normalized(a);
  b = q_normalized(b);
  double d = q_dot(a, b);
  d = std::max(-1.0, std::min(1.0, d));
  if (d < 0) {
    b = {-b.x, -b.y, -b.z, -b.w};
    d = -d;
  }
  constexpr double EPS = 1e-6;
  if (1.0 - d < EPS) { // near-linear
    const Quat q{
      a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), a.z + t * (b.z - a.z), a.w + t * (b.w - a.w)};
    return q_normalized(q);
  }
  const double theta = std::acos(d);
  const double s = std::sin(theta);
  if (std::abs(s) < kTiny) { return a; }
  const double w1 = std::sin((1 - t) * theta) / s;
  const double w2 = std::sin(t * theta) / s;
  return q_normalized(
    {a.x * w1 + b.x * w2, a.y * w1 + b.y * w2, a.z * w1 + b.z * w2, a.w * w1 + b.w * w2});
}

struct Mat3 {
  double m[3][3]{
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 1}
  };
};

inline constexpr Mat3 I3() noexcept {
  return Mat3{};
}

inline constexpr Mat3 matmul(const Mat3& A, const Mat3& B) noexcept {
  Mat3 C{};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      C.m[i][j] = A.m[i][0] * B.m[0][j] + A.m[i][1] * B.m[1][j] + A.m[i][2] * B.m[2][j];
    }
  }
  return C;
}

inline constexpr Mat3 transpose(const Mat3& A) noexcept {
  Mat3 T{};
  T.m[0][0] = A.m[0][0];
  T.m[0][1] = A.m[1][0];
  T.m[0][2] = A.m[2][0];
  T.m[1][0] = A.m[0][1];
  T.m[1][1] = A.m[1][1];
  T.m[1][2] = A.m[2][1];
  T.m[2][0] = A.m[0][2];
  T.m[2][1] = A.m[1][2];
  T.m[2][2] = A.m[2][2];
  return T;
}

inline constexpr Vec3 Rmul(const Mat3& R, const Vec3& v) noexcept {
  return {R.m[0][0] * v.x + R.m[0][1] * v.y + R.m[0][2] * v.z,
    R.m[1][0] * v.x + R.m[1][1] * v.y + R.m[1][2] * v.z,
    R.m[2][0] * v.x + R.m[2][1] * v.y + R.m[2][2] * v.z};
}

inline Mat3 R_from_quat(const Quat& q) noexcept {
  const Quat n = q_normalized(q);
  const double x = n.x, y = n.y, z = n.z, w = n.w;
  Mat3 R{};
  R.m[0][0] = 1 - 2 * (y * y + z * z);
  R.m[0][1] = 2 * (x * y - z * w);
  R.m[0][2] = 2 * (x * z + y * w);
  R.m[1][0] = 2 * (x * y + z * w);
  R.m[1][1] = 1 - 2 * (x * x + z * z);
  R.m[1][2] = 2 * (y * z - x * w);
  R.m[2][0] = 2 * (x * z - y * w);
  R.m[2][1] = 2 * (y * z + x * w);
  R.m[2][2] = 1 - 2 * (x * x + y * y);
  return R;
}

inline Quat quat_from_R(const Mat3& R) noexcept {
  const double tr = R.m[0][0] + R.m[1][1] + R.m[2][2];
  Quat q;
  if (tr > 0) {
    const double S = std::sqrt(tr + 1.0) * 2;
    q.w = 0.25 * S;
    q.x = (R.m[2][1] - R.m[1][2]) / S;
    q.y = (R.m[0][2] - R.m[2][0]) / S;
    q.z = (R.m[1][0] - R.m[0][1]) / S;
  } else if (R.m[0][0] > R.m[1][1] && R.m[0][0] > R.m[2][2]) {
    const double S = std::sqrt(1.0 + R.m[0][0] - R.m[1][1] - R.m[2][2]) * 2;
    q.w = (R.m[2][1] - R.m[1][2]) / S;
    q.x = 0.25 * S;
    q.y = (R.m[0][1] + R.m[1][0]) / S;
    q.z = (R.m[0][2] + R.m[2][0]) / S;
  } else if (R.m[1][1] > R.m[2][2]) {
    const double S = std::sqrt(1.0 - R.m[0][0] + R.m[1][1] - R.m[2][2]) * 2;
    q.w = (R.m[0][2] - R.m[2][0]) / S;
    q.x = (R.m[0][1] + R.m[1][0]) / S;
    q.y = 0.25 * S;
    q.z = (R.m[1][2] + R.m[2][1]) / S;
  } else {
    const double S = std::sqrt(1.0 - R.m[0][0] - R.m[1][1] + R.m[2][2]) * 2;
    q.w = (R.m[1][0] - R.m[0][1]) / S;
    q.x = (R.m[0][2] + R.m[2][0]) / S;
    q.y = (R.m[1][2] + R.m[2][1]) / S;
    q.z = 0.25 * S;
  }
  return q_normalized(q);
}

inline constexpr Mat3 hat(const Vec3& w) noexcept {
  Mat3 W{};
  W.m[0][0] = 0;
  W.m[0][1] = -w.z;
  W.m[0][2] = w.y;
  W.m[1][0] = w.z;
  W.m[1][1] = 0;
  W.m[1][2] = -w.x;
  W.m[2][0] = -w.y;
  W.m[2][1] = w.x;
  W.m[2][2] = 0;
  return W;
}

inline constexpr Mat3 add(const Mat3& A, const Mat3& B) noexcept {
  Mat3 C{};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      C.m[i][j] = A.m[i][j] + B.m[i][j];
    }
  }
  return C;
}

inline constexpr Mat3 scale(const Mat3& A, double s) noexcept {
  Mat3 C{};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      C.m[i][j] = A.m[i][j] * s;
    }
  }
  return C;
}

inline Mat3 so3_exp(const Vec3& w) noexcept {
  const double th = norm(w);
  const Mat3 I = I3();
  if (th < kTinyAngle) {
    const Mat3 W = hat(w);
    const Mat3 W2 = matmul(W, W);
    return add(I, add(scale(W, 1 - th * th / 6.0), scale(W2, 0.5 - th * th / 24.0)));
  } else {
    const Vec3 a = w * (1.0 / th);
    const Mat3 W = hat(a), W2 = matmul(W, W);
    const double s = std::sin(th), c = std::cos(th);
    return add(I, add(scale(W, s), scale(W2, 1 - c)));
  }
}

inline Vec3 vee(const Mat3& M) noexcept {
  return {
    0.5 * (M.m[2][1] - M.m[1][2]), 0.5 * (M.m[0][2] - M.m[2][0]), 0.5 * (M.m[1][0] - M.m[0][1])};
}

inline Vec3 so3_log(const Mat3& R) noexcept {
  const double tr = R.m[0][0] + R.m[1][1] + R.m[2][2];
  const double c = std::max(-1.0, std::min(1.0, (tr - 1.0) / 2.0));
  const double th = std::acos(c);
  if (th < kTinyAngle) {
    Mat3 S{};
    S.m[0][1] = R.m[0][1] - R.m[1][0];
    S.m[1][0] = -S.m[0][1];
    S.m[0][2] = R.m[0][2] - R.m[2][0];
    S.m[2][0] = -S.m[0][2];
    S.m[1][2] = R.m[1][2] - R.m[2][1];
    S.m[2][1] = -S.m[1][2];
    return vee(S);
  } else {
    Mat3 S{};
    S.m[0][1] = R.m[0][1] - R.m[1][0];
    S.m[1][0] = -S.m[0][1];
    S.m[0][2] = R.m[0][2] - R.m[2][0];
    S.m[2][0] = -S.m[0][2];
    S.m[1][2] = R.m[1][2] - R.m[2][1];
    S.m[2][1] = -S.m[1][2];
    const double f = th / (2.0 * std::sin(th));
    const Vec3 w = vee(S);
    return {w.x * f, w.y * f, w.z * f};
  }
}

inline Mat3 so3_left_jacobian(const Vec3& w) noexcept {
  const double th = norm(w);
  const Mat3 I = I3();
  if (th < kTinyAngle) {
    const Mat3 W = hat(w), W2 = matmul(W, W);
    return add(I, add(scale(W, 0.5), scale(W2, 1.0 / 6.0)));
  } else {
    const Vec3 a = w * (1.0 / th);
    const Mat3 W = hat(a), W2 = matmul(W, W);
    const double A = (1 - std::cos(th)) / (th * th);
    const double B = (th - std::sin(th)) / (th * th * th);
    return add(I, add(scale(W, A * th), scale(W2, B * th * th)));
  }
}

inline Mat3 so3_left_jacobian_inv(const Vec3& w) noexcept {
  const double th = norm(w);
  const Mat3 I = I3();
  if (th < kTinyAngle) {
    const Mat3 W = hat(w), W2 = matmul(W, W);
    return add(I, add(scale(W, -0.5), scale(W2, 1.0 / 12.0)));
  } else {
    const Vec3 a = w * (1.0 / th);
    const Mat3 W = hat(a), W2 = matmul(W, W);
    const double half = 0.5;
    const double cot_half = (1 + std::cos(th)) / std::sin(th);
    const double B = (1.0 / (th * th)) * (1 - (th * 0.5) * cot_half);
    return add(I, add(scale(W, -half * th), scale(W2, B * th * th)));
  }
}

struct SE3 {
  Mat3 R;
  Vec3 p;
  constexpr SE3() noexcept = default;

  constexpr SE3(const Mat3& R_, const Vec3& p_) noexcept : R(R_), p(p_) {}
};

inline SE3 se3_exp(const Vec3& v, const Vec3& w) noexcept {
  const Mat3 R = so3_exp(w);
  const Mat3 J = so3_left_jacobian(w);
  Vec3 p{J.m[0][0] * v.x + J.m[0][1] * v.y + J.m[0][2] * v.z,
    J.m[1][0] * v.x + J.m[1][1] * v.y + J.m[1][2] * v.z,
    J.m[2][0] * v.x + J.m[2][1] * v.y + J.m[2][2] * v.z};
  return {R, p};
}

inline std::pair<Vec3, Vec3> se3_log(const SE3& T) noexcept {
  const Vec3 w = so3_log(T.R);
  const Mat3 Jinv = so3_left_jacobian_inv(w);
  Vec3 v{Jinv.m[0][0] * T.p.x + Jinv.m[0][1] * T.p.y + Jinv.m[0][2] * T.p.z,
    Jinv.m[1][0] * T.p.x + Jinv.m[1][1] * T.p.y + Jinv.m[1][2] * T.p.z,
    Jinv.m[2][0] * T.p.x + Jinv.m[2][1] * T.p.y + Jinv.m[2][2] * T.p.z};
  return std::pair<Vec3, Vec3>(v, w);
}

inline constexpr SE3 se3_mul(const SE3& A, const SE3& B) noexcept {
  return {matmul(A.R, B.R), A.p + Rmul(A.R, B.p)};
}

inline constexpr SE3 se3_inv(const SE3& T) noexcept {
  const Mat3 Rt = transpose(T.R);
  const Vec3 p = Rmul(Rt, Vec3{-T.p.x, -T.p.y, -T.p.z});
  return {Rt, p};
}

inline SE3 interpolate_se3(const SE3& T0, const SE3& T1, double t) noexcept {
  const SE3 dT = se3_mul(se3_inv(T0), T1);
  const auto [v, w] = se3_log(dT);
  const SE3 dTt = se3_exp(Vec3{v.x * t, v.y * t, v.z * t}, Vec3{w.x * t, w.y * t, w.z * t});
  return se3_mul(T0, dTt);
}

inline constexpr Vec3 lerp(const Vec3& a, const Vec3& b, double t) noexcept {
  return a * (1.0 - t) + b * t;
}

} // namespace math
