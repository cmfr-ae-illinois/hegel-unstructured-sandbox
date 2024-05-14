# Sandbox for PETSc-plex Hegel conversion

- The first example initializes a 2D triangular mesh and creates a cell-centered array whose value depends on the cell-center position.
  - Tested on 1 and 4 procs with [PETSc](https://gitlab.com/petsc/petsc) (branch `main`, commit `2cac467e9d6cbb8f79fe7bcfeb2b1f07091fa1af`) configured with the options `--download-triangle --download-metis --download-parmetis`
  - Compilation with CMake and execution:
   ```
      mkdir build
      cd build
      cmake ..
      make
      mpirun -np 4 ./uhegel
   ```
