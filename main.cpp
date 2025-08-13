#include "math.hpp"

#include <foxglove/channel.hpp>
#include <foxglove/foxglove.hpp>
#include <foxglove/schemas.hpp>
#include <foxglove/server.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

// shared state updated from Foxglove callbacks (callbacks can run on multiple threads)
std::atomic<bool> g_animated{true};
std::atomic<double> g_paramT{0.0};

using math::interpolate_se3;
using math::lerp;
using math::Mat3;
using math::q_from_euler;
using math::q_slerp;
using math::Quat;
using math::quat_from_R;
using math::R_from_quat;
using math::SE3;
using math::Vec3;

// ---------- Foxglove helpers ----------
using namespace foxglove::schemas;

constexpr Color rgba(double r, double g, double b, double a = 1.0) {
  return Color{r, g, b, a};
}

constexpr Quaternion q_to_schema(const Quat& q) {
  return Quaternion{q.x, q.y, q.z, q.w};
}

constexpr Vector3 v_to_schema(const Vec3& v) {
  return Vector3{v.x, v.y, v.z};
}

inline Pose pose_from_se3(const SE3& T) {
  return Pose{v_to_schema(T.p), q_to_schema(quat_from_R(T.R))};
}

static ArrowPrimitive make_arrow(
  const Pose& base, const Quat& extra_rot, const Color& c, double L = 0.3, double d = 0.02) {
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

static foxglove::schemas::TextPrimitive make_label_above(const SE3& T, std::string_view text) {
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

static inline Quat q_axis_y() { // rot +90deg about Z: +X -> +Y
  const double s = std::sin(M_PI * 0.25), c = std::cos(M_PI * 0.25);
  return Quat{0, 0, s, c};
}

static inline Quat q_axis_z() { // rot +90deg about Y: +X -> +Z
  const double s = std::sin(M_PI * 0.25), c = std::cos(M_PI * 0.25);
  return Quat{0, s, 0, c};
}

// Triad arrows (X red, Y green, Z blue). Arrow points along +X in local frame.
static void add_triad(SceneEntity& e, const Pose& base, double scale = 0.35) {
  e.arrows.push_back(make_arrow(base, Quat{0, 0, 0, 1}, rgba(1, 0, 0, 1), scale, 0.02)); // X red
  e.arrows.push_back(make_arrow(base, q_axis_y(), rgba(0, 1, 0, 1), scale, 0.02)); // Y green
  e.arrows.push_back(make_arrow(base, q_axis_z(), rgba(0, 0, 1, 1), scale, 0.02)); // Z blue
}

static LinePrimitive line_from_points(const std::vector<Vec3>& pts,
  const Color& c,
  double thickness = 2.0,
  bool scaleInvariant = true) {
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
  ws_options.capabilities = foxglove::WebSocketServerCapabilities::Parameters;
  ws_options.callbacks.onGetParameters = [&](uint32_t /*client_id*/,
                                           std::optional<std::string_view> /*req_id*/,
                                           const std::vector<std::string_view>& names) {
    std::vector<foxglove::Parameter> out;
    auto emit = [&](std::string_view n) {
      if (n == std::string_view("animated")) {
        out.emplace_back("animated", g_animated.load());
      } else if (n == std::string_view("t")) {
        out.emplace_back("t", g_paramT.load()); // double value
      }
    };
    if (names.empty()) {
      emit("animated");
      emit("t");
    } else {
      for (auto n : names) {
        emit(n);
      }
    }
    return out;
  };
  ws_options.callbacks.onSetParameters = [&](uint32_t /*client_id*/,
                                           std::optional<std::string_view> /*req_id*/,
                                           const std::vector<foxglove::ParameterView>& params) {
    for (const auto& p : params) {
      if (p.name() == std::string_view("animated") && p.is<bool>()) {
        g_animated.store(p.get<bool>());
      } else if (p.name() == std::string_view("t") && p.is<double>()) {
        g_paramT.store(std::clamp(p.get<double>(), 0.0, 1.0));
      }
    }

    // Build result without copies (Parameter is move-only)
    std::vector<foxglove::Parameter> result;
    result.reserve(2);
    result.emplace_back("animated", g_animated.load());
    result.emplace_back("t", g_paramT.load());
    return result;
  };

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

  // ---- Start & End poses ----
  const Vec3 p0{-1.2, 0.4, -0.6};
  const Quat q0 = q_from_euler(0.2, 0.6, -0.1);
  const Vec3 p1{1.1, 0.5, 0.8};
  const Quat q1 = q_from_euler(-1.2, -0.5, 1.0);

  const SE3 T0{R_from_quat(q0), p0};
  const SE3 T1{R_from_quat(q1), p1};

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
    const double dt = std::chrono::duration<double>(now - last).count();
    last = now;
    if (g_animated.load()) {
      // Update t based on speed and direction
      t += dir * dt * speed;
      if (t > 1) {
        t = 1;
        dir = -1; // reverse direction
      }
      if (t < 0) {
        t = 0;
        dir = 1; // forward again
      }
      g_paramT.store(t); // update parameter value
    } else {
      // Use the parameter value from the client
      t = g_paramT.load();
    }

    // Interpolate
    const Vec3 p_lerp = lerp(p0, p1, t);
    const Quat q_lerp = q_slerp(q0, q1, t);
    const SE3 T_lerp{R_from_quat(q_lerp), p_lerp};

    const SE3 T_se3 = interpolate_se3(T0, T1, t);

    // Paths (resampled every tick; fine for a demo)
    constexpr int N = 100;
    std::vector<Vec3> pts_lerp, pts_se3;
    pts_lerp.reserve(N + 1);
    pts_se3.reserve(N + 1);
    for (int i = 0; i <= N; i++) {
      double s = double(i) / N;
      pts_lerp.push_back(lerp(p0, p1, s));
      SE3 Ti = interpolate_se3(T0, T1, s);
      pts_se3.push_back(Ti.p);
    }

    // 3D rendered entities
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
    // Moving frame: LERP+SLERP
    {
      SceneEntity e;
      e.id = "lerp_frame";
      e.texts.push_back(make_label_above(T_lerp, "LERP+SLERP"));
      add_triad(e, pose_from_se3(T_lerp), 0.30);
      upd.entities.push_back(std::move(e));
    }
    // Moving frame: SE(3)
    {
      SceneEntity e;
      e.id = "se3_frame";
      e.texts.push_back(make_label_above(T_se3, "SE(3)"));
      add_triad(e, pose_from_se3(T_se3), 0.30);
      upd.entities.push_back(std::move(e));
    }
    // LERP+SLERP path (blue-ish)
    {
      SceneEntity e;
      e.id = "lerp_path";
      e.lines.push_back(line_from_points(pts_lerp, rgba(0.23, 0.51, 0.96, 1.0), 2.0, true)); // blue
      upd.entities.push_back(std::move(e));
    }
    // SE(3) path (orange-ish)
    {
      SceneEntity e;
      e.id = "se3_path";
      e.lines.push_back(
        line_from_points(pts_se3, rgba(0.96, 0.62, 0.11, 1.0), 2.0, true)); // orange
      upd.entities.push_back(std::move(e));
    }

    scene.log(upd); // send one update (replaces entities by matching id)

    if (g_animated.load()) {
      // Continuously publish parameter values
      std::vector<foxglove::Parameter> params;
      params.emplace_back("animated", true);
      params.emplace_back("t", std::floor(g_paramT.load() * 100.0) / 100.0);
      server.publishParameterValues(std::move(params));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 Hz
  }

  return 0;
}
