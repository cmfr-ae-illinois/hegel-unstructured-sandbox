static char help[] = "Unstructured Hegel sandbox\n\n";

#include <petsc.h>
#include <petscdmplex.h>

int main(int argc, char **argv)
{
  PetscFunctionBegin;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  // Get rank and size of processors
  PetscMPIInt rank, size;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  // Create 2D rectangular domain filled with triangles (Requires configuring PETSc with --download-triangle)
  DM surface, dm;
  PetscScalar xmax[2] = {1.5, 1.0};
  PetscCall(DMPlexCreateBoxSurfaceMesh(PETSC_COMM_WORLD, 2, NULL, NULL, xmax, PETSC_TRUE, &surface));
  char triangle_options[1000];
  const double area_constraint = 0.001;
  sprintf(triangle_options, "pqa%.6fezQD", area_constraint);
  PetscCall(DMPlexTriangleSetOptions(surface, triangle_options));
  PetscCall(DMPlexGenerate(surface, "triangle", PETSC_TRUE, &dm));

  // If nproc > 1, distribute mesh with Parmetis (Requires configuring PETSc with --download-metis --download-parmetis)
  if (size > 1)
  {
    // Get DM partitioner and set it to Parmetis
    PetscPartitioner part;
    PetscCall(DMPlexGetPartitioner(dm, &part));
    if (part)
    {
      PetscCall(PetscPartitionerSetType(part, PETSCPARTITIONERPARMETIS));
      PetscCall(PetscPartitionerSetFromOptions(part));
    }
    // Partition mesh and distribute
    const PetscInt ghost_layers = 1;
    DM dm_dist;
    PetscCall(DMPlexDistribute(dm, ghost_layers, NULL, &dm_dist));
    PetscCall(DMDestroy(&dm));
    dm = dm_dist;
  }

  // Get mesh information
  PetscInt vert_start, vert_end, edge_start, edge_end, cell_start, cell_end;
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vert_start, &vert_end));
  PetscCall(DMPlexGetDepthStratum(dm, 1, &edge_start, &edge_end));
  PetscCall(DMPlexGetDepthStratum(dm, 2, &cell_start, &cell_end));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Rank %3i has %6i vertices, %6i edges, %6i cells\n", rank, vert_end - vert_start, edge_end - edge_start,
                        cell_end - cell_start));

  // Create section that stores 1 DOF for each cell
  PetscInt num_fields = 1;
  PetscInt num_DOF[num_fields * 3], num_Comp[num_fields];
  PetscSection section;
  num_Comp[0] = 1;
  for (PetscInt i = 0; i < num_fields * 3; i++) num_DOF[i] = 0;
  num_DOF[2] = 1;
  PetscCall(DMSetNumFields(dm, num_fields));
  PetscCall(DMPlexCreateSection(dm, NULL, num_Comp, num_DOF, 0, NULL, NULL, NULL, NULL, &section));
  PetscCall(PetscSectionSetFieldName(section, 0, "ScalarTracer"));
  PetscCall(DMSetSection(dm, section));
  PetscCall(PetscSectionDestroy(&section));

  // Create vector and initialize with 2D Gaussian pulse
  Vec data;
  PetscScalar *data_array;
  PetscCall(DMCreateLocalVector(dm, &data));
  PetscCall(VecSet(data, 0.));
  PetscCall(VecGetArray(data, &data_array));
  const PetscScalar std = 0.2 * xmax[0];
  for (PetscInt c = cell_start; c < cell_end; ++c)
  {
    // Get volume and centroid of cell
    PetscScalar volume, centroid[2];
    PetscCall(DMPlexComputeCellGeometryFVM(dm, c, &volume, centroid, NULL));
    // Compute distance squared to the center of the domain
    PetscScalar dist_to_center_sq =
        (centroid[0] - 0.5 * xmax[0]) * (centroid[0] - 0.5 * xmax[0]) + (centroid[1] - 0.5 * xmax[1]) * (centroid[1] - 0.5 * xmax[1]);
    // Initialize data with Gaussian pulse
    data_array[c - cell_start] = PetscExpReal(-dist_to_center_sq / (0.5 * std * std));
  }
  PetscCall(VecRestoreArray(data, &data_array));

  // Write DM in VTK format (readable with Paraview)
  PetscViewer viewer = NULL;
  PetscCall(PetscViewerVTKOpen(PETSC_COMM_WORLD, "mesh.vtu", FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(data, viewer));
  PetscCall(DMView(dm, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());

  return 0;
}