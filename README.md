# Pose Interpolation Demo

A minimal C++20 example showing how to visualize real‑time 3D data from a C++ app in Foxglove using the Foxglove SDK. It animates a rigid body pose between two endpoints with two interpolation methods:

1. LERP (translation) + SLERP (quaternion rotation) – simple, decoupled.
2. SE(3) interpolation via the Lie group exponential / logarithm – couples rotation + translation through the twist.

You can toggle animation and scrub the interpolation parameter live from Foxglove parameters.

![Pose interpolation visualization in Foxglove](./docs/pose-interpolation-looping.gif)

## Why this repo?

Provides a compact, readable reference for:

- Integrating the Foxglove WebSocket server in a native C++ application.
- Publishing a SceneUpdate channel with custom entities (arrows, lines, text labels).
- Exposing live parameters (bool + double) for interactive control.
- Implementing numerically safe SE(3) utilities (exp/log, Jacobians) without external math libs.

## Project layout

```
main.cpp   // Foxglove server setup, parameter handling, scene building, animation loop
math.hpp   // Header‑only minimal Vec3 / Quat / Mat3 / SE(3) + interpolation helpers
build.sh   // Convenience build script (CMake + Ninja)
```

## Build

Prerequisites: CMake ≥ 3.14, a C++20 compiler (clang++ / g++ / MSVC), and internet access for first‑time Foxglove SDK fetch.

Option 1 (script):

```
./build.sh
```

Option 2 (manual example):

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

The first configure step downloads the prebuilt Foxglove SDK archive matching your platform (hash verified).

## Run

```
./build/se3_foxglove
```

You should see a log line: `Foxglove server on ws://127.0.0.1:8765`.

## View in Foxglove

1. Open the Foxglove desktop (or web) app at [https://app.foxglove.dev/](https://app.foxglove.dev/). You will need to create a free account if you have not already.
2. "Open connection...": WebSocket → `ws://127.0.0.1:8765`.
3. Select the layout dropdown and choose "Import from file...". Select `foxglove_layout.json` from this repository.

## What you’re seeing

Entities:

- Two fixed endpoint frames (T₀, T₁).
- Two moving frames at the interpolated pose:
  - LERP+SLERP (blue).
  - SE(3) (orange).
- Two path polylines tracing the translation trajectories.

Key visual differences: For generic endpoint poses, the SE(3) path can curve differently from naive LERP of translations, especially when large rotations occur—reflecting the coupled nature of body twists.

## Customization ideas

- Adjust endpoints in `main.cpp`.
- Add more interpolation variants (e.g., screw linear interpolation with pitch).
- Extend parameters (speed, sample count, color themes).

## License

MIT – see `LICENSE`.
