# 3D Print Preview Webapp

A real-time 3D print preview renderer that runs in the browser.

Takes in a 3D model, runs it through OrcaSlicer, and renders the sliced GCode in WebGL2.

An intermediate format `.segbin` is used to store the extrusion segments in a compact binary format for the frontend to load and render.

![preview](assets/preview.png)

## Disclaimer

This was semi-vibe-coded with significant oversight.

That said, I'm not a graphics programmer nor a strong algorithms specialist so there's likely graphics, etc., code that could be improved more than I know how to.

## How it works

1. User uploads a 3D model (STL, OBJ, etc.) or selects a builtin model
2. The model is sent to the backend, which runs OrcaSlicer to generate GCode
3. A rust program parses the GCode, culls hidden segments, and outputs a `.segbin` file
4. The frontend loads the `.segbin` file and renders the extrusion segments

## Features

Server

- Runs OrcaSlicer to generate GCode from 3D models
- Uses a ray-based surface culling algorithm to remove hidden segments from the GCode to minimize size and frontend rendering effort
- Uses a custom binary format `.segbin` to store extrusion segments in a compact format

Frontend

- Vue 3 + Babylon.js WebGL2 renderer
- Uses instanced rendering to render thousands of extrusion segments
- 3-level level-of-detail (LOD) system to reduce rendering cost
- Tuneable material properties (color, roughness, metallic, etc.)

## Quick start

### Nix (recommended)

```bash
nix-shell
```

### Without Nix

Install `bun` and `cargo`

### Running the project

```bash
cd packages/gcode-to-segbin
cargo build --release
```

```bash
cd packages/frontend
bun i
bun dev
```

```bash
cd packages/backend
bun i
bun dev
```

## Code layout

```text
packages/
├── frontend/           # Vue 3 + Babylon.js WebGL2 renderer
│   ├── src/
│   │   ├── renderer/   # GLSL shaders, geometry, segbin loader
│   │   ├── components/ # ModelViewer, App
│   │   └── __tests__/
│   └── public/         # Static assets (env maps)
├── backend/            # Express API for slicing pipeline and frontend proxy
└── gcode-to-segbin/    # Rust CLI: GCode → .segbin
    └── src/
        ├── parser.rs   # GCode state machine
        ├── arcs.rs     # Conic fillet subdivision at corners
        └── cull/       # Surface culling (contour / ray)
```

## Todo (Future wishlist)

- WebGPU renderer while using the WebGL2 as a fallback
- Temporal reprojection for the existing TAA
- Better LOD since large models can still end up with >20M triangles
- A way to get Deepseek v4 to stop trying to give up and revert its changes all the time

## Licenses

- **Code**: MIT (see [`LICENSE`](LICENSE))
- **Calibration cube model**: CC-BY 4.0 by DoomBro on [Printables](https://www.printables.com/model/32539-xyz-10mm-calibration-cube)
- **Environment map**: CC0 from [Poly Haven](https://polyhaven.com/a/horn-koppe_spring)
