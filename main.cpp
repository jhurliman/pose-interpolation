#include <foxglove/channel.hpp>
#include <foxglove/foxglove.hpp>
#include <foxglove/schemas.hpp>
#include <foxglove/server.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// ---------- Small math helpers (Vec/Quat/Mat3 + SE(3) log/exp) ----------
struct Vec3 {
  double x{}, y{}, z{};
  Vec3() = default;

  Vec3(double X, double Y, double Z) : x(X), y(Y), z(Z) {}

  Vec3 operator+(const Vec3& b) const { return {x + b.x, y + b.y, z + b.z}; }

  Vec3 operator-(const Vec3& b) const { return {x - b.x, y - b.y, z - b.z}; }

  Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }

  Vec3& operator+=(const Vec3& b) {
    x += b.x;
    y += b.y;
    z += b.z;
    return *this;
  }
};

inline double dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline double norm(const Vec3& a) {
  return std::sqrt(std::max(1e-18, dot(a, a)));
}

inline Vec3 normalized(const Vec3& a) {
  double n = norm(a);
  return {a.x / n, a.y / n, a.z / n};
}

struct Quat { // (x,y,z,w) per Foxglove schema
  double x{}, y{}, z{}, w{1};
};

inline Quat q_mul(const Quat& a, const Quat& b) {
  return {a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

inline double q_dot(const Quat& a, const Quat& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline Quat q_normalized(Quat q) {
  double n = std::sqrt(q_dot(q, q));
  q.x /= n;
  q.y /= n;
  q.z /= n;
  q.w /= n;
  return q;
}

inline Quat q_conj(const Quat& q) {
  return {-q.x, -q.y, -q.z, q.w};
}

inline Quat q_from_euler(double rx, double ry, double rz) { // XYZ intrinsic
  double cx = std::cos(rx * 0.5), sx = std::sin(rx * 0.5);
  double cy = std::cos(ry * 0.5), sy = std::sin(ry * 0.5);
  double cz = std::cos(rz * 0.5), sz = std::sin(rz * 0.5);
  Quat qx{sx, 0, 0, cx}, qy{0, sy, 0, cy}, qz{0, 0, sz, cz};
  return q_normalized(q_mul(q_mul(qx, qy), qz));
}

inline Quat q_slerp(Quat a, Quat b, double t) {
  a = q_normalized(a);
  b = q_normalized(b);
  double d = q_dot(a, b);
  if (d < 0) {
    b = {-b.x, -b.y, -b.z, -b.w};
    d = -d;
  }
  const double EPS = 1e-6;
  if (1.0 - d < EPS) {
    // LERP then normalize
    Quat q{
      a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), a.z + t * (b.z - a.z), a.w + t * (b.w - a.w)};
    return q_normalized(q);
  }
  double theta = std::acos(d);
  double s = std::sin(theta);
  double w1 = std::sin((1 - t) * theta) / s;
  double w2 = std::sin(t * theta) / s;
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

inline Mat3 I3() {
  return Mat3{};
}

inline Mat3 matmul(const Mat3& A, const Mat3& B) {
  Mat3 C{};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      C.m[i][j] = A.m[i][0] * B.m[0][j] + A.m[i][1] * B.m[1][j] + A.m[i][2] * B.m[2][j];
    }
  }
  return C;
}

inline Vec3 Rmul(const Mat3& R, const Vec3& v) {
  return {R.m[0][0] * v.x + R.m[0][1] * v.y + R.m[0][2] * v.z,
    R.m[1][0] * v.x + R.m[1][1] * v.y + R.m[1][2] * v.z,
    R.m[2][0] * v.x + R.m[2][1] * v.y + R.m[2][2] * v.z};
}

inline Mat3 R_from_quat(const Quat& q) {
  Quat n = q_normalized(q);
  double x = n.x, y = n.y, z = n.z, w = n.w;
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

inline Quat quat_from_R(const Mat3& R) {
  double tr = R.m[0][0] + R.m[1][1] + R.m[2][2];
  Quat q;
  if (tr > 0) {
    double S = std::sqrt(tr + 1.0) * 2;
    q.w = 0.25 * S;
    q.x = (R.m[2][1] - R.m[1][2]) / S;
    q.y = (R.m[0][2] - R.m[2][0]) / S;
    q.z = (R.m[1][0] - R.m[0][1]) / S;
  } else if (R.m[0][0] > R.m[1][1] && R.m[0][0] > R.m[2][2]) {
    double S = std::sqrt(1.0 + R.m[0][0] - R.m[1][1] - R.m[2][2]) * 2;
    q.w = (R.m[2][1] - R.m[1][2]) / S;
    q.x = 0.25 * S;
    q.y = (R.m[0][1] + R.m[1][0]) / S;
    q.z = (R.m[0][2] + R.m[2][0]) / S;
  } else if (R.m[1][1] > R.m[2][2]) {
    double S = std::sqrt(1.0 - R.m[0][0] + R.m[1][1] - R.m[2][2]) * 2;
    q.w = (R.m[0][2] - R.m[2][0]) / S;
    q.x = (R.m[0][1] + R.m[1][0]) / S;
    q.y = 0.25 * S;
    q.z = (R.m[1][2] + R.m[2][1]) / S;
  } else {
    double S = std::sqrt(1.0 - R.m[0][0] - R.m[1][1] + R.m[2][2]) * 2;
    q.w = (R.m[1][0] - R.m[0][1]) / S;
    q.x = (R.m[0][2] + R.m[2][0]) / S;
    q.y = (R.m[1][2] + R.m[2][1]) / S;
    q.z = 0.25 * S;
  }
  return q_normalized(q);
}

inline Mat3 hat(const Vec3& w) {
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

inline Mat3 add(const Mat3& A, const Mat3& B) {
  Mat3 C{};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      C.m[i][j] = A.m[i][j] + B.m[i][j];
    }
  }
  return C;
}

inline Mat3 scale(const Mat3& A, double s) {
  Mat3 C{};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      C.m[i][j] = A.m[i][j] * s;
    }
  }
  return C;
}

inline Mat3 so3_exp(const Vec3& w) {
  double th = norm(w);
  Mat3 I = I3();
  if (th < 1e-8) {
    Mat3 W = hat(w);
    Mat3 W2 = matmul(W, W);
    return add(I, add(scale(W, 1 - th * th / 6.0), scale(W2, 0.5 - th * th / 24.0)));
  } else {
    Vec3 a = w * (1.0 / th);
    Mat3 W = hat(a), W2 = matmul(W, W);
    double s = std::sin(th), c = std::cos(th);
    return add(I, add(scale(W, s), scale(W2, 1 - c)));
  }
}

inline Vec3 vee(const Mat3& M) {
  return {
    0.5 * (M.m[2][1] - M.m[1][2]), 0.5 * (M.m[0][2] - M.m[2][0]), 0.5 * (M.m[1][0] - M.m[0][1])};
}

inline Vec3 so3_log(const Mat3& R) {
  double tr = R.m[0][0] + R.m[1][1] + R.m[2][2];
  double c = std::max(-1.0, std::min(1.0, (tr - 1.0) / 2.0));
  double th = std::acos(c);
  if (th < 1e-8) {
    Mat3 Rt{};
    Rt.m[0][1] = R.m[0][1];
    Rt.m[1][0] = R.m[1][0];
    Rt.m[0][2] = R.m[0][2];
    Rt.m[2][0] = R.m[2][0];
    Rt.m[1][2] = R.m[1][2];
    Rt.m[2][1] = R.m[2][1];
    Mat3 S{}; // R - R^T packed in S off-diagonals
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
    double f = th / (2.0 * std::sin(th));
    Vec3 w = vee(S);
    return {w.x * f, w.y * f, w.z * f};
  }
}

inline Mat3 so3_left_jacobian(const Vec3& w) {
  double th = norm(w);
  Mat3 I = I3();
  if (th < 1e-8) {
    Mat3 W = hat(w), W2 = matmul(W, W);
    return add(I, add(scale(W, 0.5), scale(W2, 1.0 / 6.0)));
  } else {
    Vec3 a = w * (1.0 / th);
    Mat3 W = hat(a), W2 = matmul(W, W);
    double A = (1 - std::cos(th)) / (th * th);
    double B = (th - std::sin(th)) / (th * th * th);
    return add(I, add(scale(W, A * th), scale(W2, B * th * th)));
  }
}

inline Mat3 so3_left_jacobian_inv(const Vec3& w) {
  double th = norm(w);
  Mat3 I = I3();
  if (th < 1e-8) {
    Mat3 W = hat(w), W2 = matmul(W, W);
    return add(I, add(scale(W, -0.5), scale(W2, 1.0 / 12.0)));
  } else {
    Vec3 a = w * (1.0 / th);
    Mat3 W = hat(a), W2 = matmul(W, W);
    double half = 0.5;
    double cot_half = (1 + std::cos(th)) / std::sin(th); // cot(th/2)
    double B = (1.0 / (th * th)) * (1 - (th * 0.5) * cot_half);
    return add(I, add(scale(W, -half * th), scale(W2, B * th * th)));
  }
}

struct SE3 {
  Mat3 R;
  Vec3 p;
};

inline SE3 se3_exp(const Vec3& v, const Vec3& w) {
  Mat3 R = so3_exp(w);
  Mat3 J = so3_left_jacobian(w);
  Vec3 p = {J.m[0][0] * v.x + J.m[0][1] * v.y + J.m[0][2] * v.z,
    J.m[1][0] * v.x + J.m[1][1] * v.y + J.m[1][2] * v.z,
    J.m[2][0] * v.x + J.m[2][1] * v.y + J.m[2][2] * v.z};
  return {R, p};
}

inline std::pair<Vec3, Vec3> se3_log(const SE3& T) {
  Vec3 w = so3_log(T.R);
  Mat3 Jinv = so3_left_jacobian_inv(w);
  Vec3 v{Jinv.m[0][0] * T.p.x + Jinv.m[0][1] * T.p.y + Jinv.m[0][2] * T.p.z,
    Jinv.m[1][0] * T.p.x + Jinv.m[1][1] * T.p.y + Jinv.m[1][2] * T.p.z,
    Jinv.m[2][0] * T.p.x + Jinv.m[2][1] * T.p.y + Jinv.m[2][2] * T.p.z};
  return {v, w};
}

inline SE3 se3_mul(const SE3& A, const SE3& B) {
  return {matmul(A.R, B.R), A.p + Rmul(A.R, B.p)};
}

inline SE3 se3_inv(const SE3& T) {
  Mat3 Rt{}; // transpose
  Rt.m[0][0] = T.R.m[0][0];
  Rt.m[0][1] = T.R.m[1][0];
  Rt.m[0][2] = T.R.m[2][0];
  Rt.m[1][0] = T.R.m[0][1];
  Rt.m[1][1] = T.R.m[1][1];
  Rt.m[1][2] = T.R.m[2][1];
  Rt.m[2][0] = T.R.m[0][2];
  Rt.m[2][1] = T.R.m[1][2];
  Rt.m[2][2] = T.R.m[2][2];
  Vec3 p = Rmul(Rt, Vec3{-T.p.x, -T.p.y, -T.p.z});
  return {Rt, p};
}

// Interpolants
inline SE3 interpolate_se3(const SE3& T0, const SE3& T1, double t) {
  SE3 dT = se3_mul(se3_inv(T0), T1);
  auto [v, w] = se3_log(dT);
  SE3 dTt = se3_exp(Vec3{v.x * t, v.y * t, v.z * t}, Vec3{w.x * t, w.y * t, w.z * t});
  return se3_mul(T0, dTt);
}

inline Vec3 lerp(const Vec3& a, const Vec3& b, double t) {
  return a * (1.0 - t) + b * t;
}

// ---------- Foxglove helpers ----------
using namespace foxglove::schemas;

inline Color rgba(double r, double g, double b, double a = 1.0) {
  return Color{r, g, b, a};
}

inline Quaternion q_to_schema(const Quat& q) {
  return Quaternion{q.x, q.y, q.z, q.w};
}

inline Vector3 v_to_schema(const Vec3& v) {
  return Vector3{v.x, v.y, v.z};
}

inline Pose pose_from_se3(const SE3& T) {
  return Pose{v_to_schema(T.p), q_to_schema(quat_from_R(T.R))};
}

// Triad arrows (X red, Y green, Z blue). Arrow points along +X in local frame.
static ArrowPrimitive make_arrow(
  const Pose& base, const Quat& extra_rot, Color c, double L = 0.3, double d = 0.02) {
  Pose p = base;

  // Read base orientation as Quat (identity if missing)
  Quat f = base.orientation ?
    Quat{base.orientation->x, base.orientation->y, base.orientation->z, base.orientation->w} :
    Quat{0, 0, 0, 1};

  // Compose: world frame * axis-rotation
  Quat q = q_mul(f, extra_rot);

  // Write back to schema
  p.orientation = q_to_schema(q);

  ArrowPrimitive a;
  a.pose = p;
  a.shaft_length = L;
  a.shaft_diameter = d;
  a.head_length = 2 * d;
  a.head_diameter = 3 * d;
  a.color = c;
  return a;
}

static foxglove::schemas::TextPrimitive make_label_above(const SE3& T, const std::string& text) {
  using namespace foxglove::schemas;
  TextPrimitive tx;

  // Put label above the frame in world +Y. Triad scale ~0.35 → offset ~0.45 looks good.
  const Vec3 po = T.p + Vec3{0.45, 0.45, 0.0}; // tweak if you want higher/lower

  Pose p;
  p.position = v_to_schema(po);
  p.orientation = Quaternion{0, 0, 0, 1}; // ignored when billboard=true, but safe to set
  tx.pose = p;

  tx.text = text;
  tx.font_size = 14.0; // pixels when scale_invariant=true
  tx.scale_invariant = true; // keep same on-screen size
  tx.billboard = true; // always face the camera
  tx.color = rgba(0.1, 0.1, 0.1, 1.0);

  return tx;
}

static Quat q_axis_y() { // rot +90deg about Z: +X -> +Y
  double s = std::sin(M_PI * 0.25), c = std::cos(M_PI * 0.25);
  return Quat{0, 0, s, c};
}

static Quat q_axis_z() { // rot +90deg about Y: +X -> +Z
  double s = std::sin(M_PI * 0.25), c = std::cos(M_PI * 0.25);
  return Quat{0, s, 0, c};
}

static void add_triad(SceneEntity& e, const Pose& base, double scale = 0.35) {
  e.arrows.push_back(make_arrow(base, Quat{0, 0, 0, 1}, rgba(1, 0, 0, 1), scale, 0.02)); // X red
  e.arrows.push_back(make_arrow(base, q_axis_y(), rgba(0, 1, 0, 1), scale, 0.02)); // Y green
  e.arrows.push_back(make_arrow(base, q_axis_z(), rgba(0, 0, 1, 1), scale, 0.02)); // Z blue
}

static LinePrimitive line_from_points(
  const std::vector<Vec3>& pts, Color c, double thickness = 2.0, bool scaleInvariant = true) {
  LinePrimitive L;
  L.type = foxglove::schemas::LinePrimitive::LineType::LINE_STRIP;
  L.pose = Pose{
    Vector3{0, 0, 0},
    Quaternion{0, 0, 0, 1}
  };
  L.thickness = thickness;
  L.scale_invariant = scaleInvariant;
  L.color = c;
  L.points.reserve(pts.size());
  for (const auto& p : pts) {
    L.points.push_back(Point3{p.x, p.y, p.z});
  }
  return L;
}

// ---------- Main ----------
int main(int, const char**) {
  foxglove::setLogLevel(foxglove::LogLevel::Debug);

  // Start a WebSocket server (connect from the Foxglove app to ws://127.0.0.1:8765)
  foxglove::WebSocketServerOptions ws_options;
  ws_options.host = "127.0.0.1";
  ws_options.port = 8765;
  auto server_result = foxglove::WebSocketServer::create(std::move(ws_options));
  if (!server_result.has_value()) {
    std::cerr << "Failed to create server: " << foxglove::strerror(server_result.error()) << "\n";
    return 1;
  }
  auto server = std::move(server_result.value());
  std::cerr << "Foxglove server on ws://127.0.0.1:" << server.port() << "\n";

  // SceneUpdate channel
  auto scene_chan_result = foxglove::schemas::SceneUpdateChannel::create("/scene");
  if (!scene_chan_result.has_value()) {
    std::cerr << "Failed to create /scene channel: "
              << foxglove::strerror(scene_chan_result.error()) << "\n";
    return 1;
  }
  auto scene = std::move(scene_chan_result.value());

  // ---- Start & End poses (match the React demo) ----
  Vec3 p0{-1.2, 0.4, -0.6};
  Quat q0 = q_from_euler(0.2, 0.6, -0.1);
  Vec3 p1{1.1, 0.5, 0.8};
  Quat q1 = q_from_euler(-1.2, -0.5, 1.0);

  SE3 T0{R_from_quat(q0), p0};
  SE3 T1{R_from_quat(q1), p1};

  // Animation state (ping-pong t in [0,1])
  double t = 0.0;
  double dir = 1.0;
  double speed = 0.5; // ~0.5 cycles/sec
  auto last = std::chrono::steady_clock::now();

  // Ctrl-C to quit
  static std::function<void()> on_sigint;
  std::atomic_bool done = false;
  std::signal(SIGINT, [](int) {
    if (on_sigint) { on_sigint(); }
  });
  on_sigint = [&] { done = true; };

  while (!done) {
    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - last).count();
    last = now;
    t += dir * dt * speed;
    if (t > 1) {
      t = 1;
      dir = -1;
    }
    if (t < 0) {
      t = 0;
      dir = 1;
    }

    // Interpolate
    Vec3 p_lerp = lerp(p0, p1, t);
    Quat q_lerp = q_slerp(q0, q1, t);
    SE3 T_lerp{R_from_quat(q_lerp), p_lerp};

    SE3 T_se3 = interpolate_se3(T0, T1, t);

    // Paths (resampled every tick; fine for demo)
    const int N = 100;
    std::vector<Vec3> pts_lerp, pts_se3;
    pts_lerp.reserve(N + 1);
    pts_se3.reserve(N + 1);
    for (int i = 0; i <= N; i++) {
      double s = double(i) / N;
      pts_lerp.push_back(lerp(p0, p1, s));
      SE3 Ti = interpolate_se3(T0, T1, s);
      pts_se3.push_back(Ti.p);
    }

    // Screw axis from log(T0^{-1}T1)
    SE3 dT = se3_mul(se3_inv(T0), T1);
    auto [v, w] = se3_log(dT);
    double wlen = norm(w);
    Vec3 w_hat{0, 0, 0}, q{0, 0, 0};
    if (wlen > 1e-8) {
      w_hat = {w.x / wlen, w.y / wlen, w.z / wlen};
      q = cross(w, v) * (1.0 / (wlen * wlen)); // a point on axis
    }
    // Draw axis as a segment around q
    double L = 3.0;
    std::vector<Vec3> axis_pts{q - w_hat * L, q + w_hat * L};

    // Entities
    SceneUpdate upd;

    // Start frame triad (green label color via ARGB already on arrows)
    {
      SceneEntity e;
      e.id = "start_frame";
      e.texts.push_back(make_label_above(T0, "T_0"));
      add_triad(e, pose_from_se3(T0), 0.35);
      upd.entities.push_back(std::move(e));
    }
    // End frame triad
    {
      SceneEntity e;
      e.id = "end_frame";
      e.texts.push_back(make_label_above(T1, "T_1"));
      add_triad(e, pose_from_se3(T1), 0.35);
      upd.entities.push_back(std::move(e));
    }
    // Moving frame: LERP+SLERP (blue-ish)
    {
      SceneEntity e;
      e.id = "lerp_frame";
      e.texts.push_back(make_label_above(T_lerp, "LERP+SLERP"));
      add_triad(e, pose_from_se3(T_lerp), 0.30);
      upd.entities.push_back(std::move(e));
    }
    // Moving frame: SE(3) (orange-ish)
    {
      SceneEntity e;
      e.id = "se3_frame";
      e.texts.push_back(make_label_above(T_se3, "SE(3)"));
      add_triad(e, pose_from_se3(T_se3), 0.30);
      upd.entities.push_back(std::move(e));
    }
    // Paths
    {
      SceneEntity e;
      e.id = "lerp_path";
      e.lines.push_back(line_from_points(pts_lerp, rgba(0.23, 0.51, 0.96, 1.0), 2.0, true)); // blue
      upd.entities.push_back(std::move(e));
    }
    {
      SceneEntity e;
      e.id = "se3_path";
      e.lines.push_back(
        line_from_points(pts_se3, rgba(0.96, 0.62, 0.11, 1.0), 2.0, true)); // orange
      upd.entities.push_back(std::move(e));
    }
    // Screw axis (if rotational)
    if (wlen > 1e-8) {
      SceneEntity e;
      e.id = "screw_axis";
      e.texts.push_back(make_label_above(T_lerp, "Screw axis"));
      e.lines.push_back(line_from_points(axis_pts, rgba(0.98, 0.45, 0.17, 1.0), 1.5, true));
      SpherePrimitive s;
      s.pose = Pose{
        v_to_schema(q), Quaternion{0, 0, 0, 1}
      };
      s.size = Vector3{0.10, 0.10, 0.10}; // diameter along x,y,z (radius 0.05 ⇒ size 0.10)
      s.color = rgba(0.98, 0.57, 0.24, 1.0);
      e.spheres.push_back(s);
      upd.entities.push_back(std::move(e));
    } else {
      // If pure translation, remove any previous axis
      SceneEntityDeletion del;
      del.id = "screw_axis";
      SceneUpdate tmp;
      tmp.deletions.push_back(del);
      scene.log(tmp);
    }

    scene.log(upd); // send one update (replaces entities by matching id)

    std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 Hz
  }

  return 0;
}
