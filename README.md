# N-Body Simulation

Real-time gravity simulation using Barnes-Hut tree decomposition. Written in C++ with OpenGL rendering.

![screenshot](https://github.com/user-attachments/assets/3433162c-d0c5-4870-9437-c5accc6f91af)

## What it does

Simulates N-body gravitational interactions in real time. Barnes-Hut reduces the naive O(n²) complexity to O(n log n) by approximating distant particle clusters as single bodies. The algorithm currently runs across two passes for gravity computation, accelerated on GPU via CUDA.

## Current state

This is my first simulation project and an active work-in-progress. Expect rough edges — naming inconsistencies, performance headroom left on the table, and occasional bugs. Source code will be published as the codebase gets cleaned up.

## Requirements

- Windows 10 / 11
- NVIDIA GPU with CUDA support

## Usage

Download the latest release and run the executable. No install needed.

## Dependencies

- OpenGL
- GLFW
- GLAD
- Dear ImGui
- CUDA

## License

[PolyForm Noncommercial License](LICENSE). Free to read, modify, and contribute. Commercial use is not permitted — a commercial license will be offered separately in the future.
