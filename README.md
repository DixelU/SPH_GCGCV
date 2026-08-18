# SPH Gas Cloud 3D

This project simulates a self-gravitating gas cloud with three-dimensional
smoothed-particle hydrodynamics (SPH). The original planar simulation and
custom immediate-mode widget framework have been replaced by a 3D simulation
core, a GLFW/OpenGL renderer, and Dear ImGui controls.

## Dependencies

The Visual Studio project uses the system-wide vcpkg integration with the
`x64-windows-static` triplet. Install these packages if they are not already
available:

- `glfw3:x64-windows-static`
- `imgui[glfw-binding,opengl3-binding]:x64-windows-static`

The project uses the OpenGL declarations shipped with the platform and loads
the modern entry points it needs through `glfwGetProcAddress`.

## Build

Open `GasCloudGravCollapseVis.sln` in Visual Studio and build `Release | x64`,
or run:

```powershell
msbuild GasCloudGravCollapseVis.sln /m /p:Configuration=Release /p:Platform=x64
```

## GUI controls

- **Run/Pause/Step** control the simulation worker.
- **Reset** regenerates a spherical cloud using the initial-condition panel.
- Left-drag orbits the camera.
- Right- or middle-drag pans.
- The mouse wheel zooms.
- `F` resets the camera.
- The field selector colors particles by density, energy, speed,
  acceleration, or any velocity component.

The renderer uses depth-tested round point sprites. The outlined cube shows
the initial reference domain; the physical simulation itself uses open space
and a dynamic octree rather than artificial periodic wrapping.

## Command-line validation

```powershell
# Numerical and conservation tests
x64\Release\GasCloudGravCollapseVis.exe --self-test

# Headless simulation: steps, seed, particles
x64\Release\GasCloudGravCollapseVis.exe --headless 100 1 1000

# Cubic-lattice preparation/iteration benchmark: particles, steps, support
x64\Release\GasCloudGravCollapseVis.exe --benchmark-preparation 10000 0 2.5

# Hidden one-frame GLFW/OpenGL/ImGui renderer check
x64\Release\GasCloudGravCollapseVis.exe --gui-smoke-test
```

## Architecture

- `simulation.h/.cpp`: particles, 3D Wendland C2 kernel, spatial hash,
  Barnes-Hut octree, SPH integration, snapshots, and worker thread.
- `renderer.h/.cpp`: orbit camera, shader management, VBOs, point sprites,
  field coloring, and domain guide.
- `GasCloudGravCollapseVis.cpp`: Dear ImGui application and command-line test
  harness.

Dense volume ray marching and isosurface extraction are intentionally outside
this first 3D renderer. The current point representation remains responsive at
particle counts for which a dense voxel volume would be impractical.
