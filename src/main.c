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
  PetscScalar xmax[2] = {1.25, 1.0};
  PetscCall(DMPlexCreateBoxSurfaceMesh(PETSC_COMM_WORLD, 2, NULL, NULL, xmax, PETSC_TRUE, &surface));
  char triangle_options[1000];
  const double area_constraint = 0.0001;
  sprintf(triangle_options, "pq25.0a%.6fezQD", area_constraint);
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
  PetscCall(PetscSectionSetFieldName(section, 0, ""));
  PetscCall(DMSetSection(dm, section));
  PetscCall(PetscSectionDestroy(&section));

  // Create vector and initialize with 2D Gaussian pulse
  Vec data, data_gradx;
  PetscScalar *data_array, *data_gradx_array;
  PetscCall(DMCreateLocalVector(dm, &data));
  PetscCall(DMCreateLocalVector(dm, &data_gradx));
  PetscCall(VecSet(data, 0.));
  PetscCall(VecSet(data_gradx, 0.));
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

  DMLabel ghost_label;
  PetscCall(DMGetLabel(dm, "ghost", &ghost_label));
  if (!ghost_label) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "There is not ghost label\n"));
  PetscCall(VecGetArray(data_gradx, &data_gradx_array));
  for (PetscInt c = cell_start; c < cell_end; ++c)
  {
    // If cell is ghost, skip
    PetscInt ghost_label_value = -1;
    if (ghost_label) PetscCall(DMLabelGetValue(ghost_label, c, &ghost_label_value));
    if (ghost_label_value >= 0) continue;

    // Get volume and centroid of cell
    PetscScalar cell_volume, cell_centroid[2];
    PetscCall(DMPlexComputeCellGeometryFVM(dm, c, &cell_volume, cell_centroid, NULL));

    // Get number of faces
    PetscInt nfaces = 0;
    const PetscInt *faces;
    PetscCall(DMPlexGetConeSize(dm, c, &nfaces));  // Should be 3
    PetscAssert(nfaces == 3, NULL, 1, "Cell has a number of faces != 3");
    PetscCall(DMPlexGetCone(dm, c, &faces));

    // Loop over faces and get neighbours and normal
    PetscScalar cell_volume_gauss = 0.0;  // For testing
    for (PetscInt i = 0; i < nfaces; ++i)
    {
      const PetscInt f = faces[i];
      PetscScalar face_area, face_centroid[2], face_normal[2];
      PetscCall(DMPlexComputeCellGeometryFVM(dm, f, &face_area, face_centroid, face_normal));
      PetscScalar cell_to_face_centroid[2] = {face_centroid[0] - cell_centroid[0], face_centroid[1] - cell_centroid[1]};
      PetscScalar dot_product = cell_to_face_centroid[0] * face_normal[0] + cell_to_face_centroid[1] * face_normal[1];
      cell_volume_gauss += 0.5 * PetscAbs(dot_product) * face_area;

      // Orient normal outwards
      if (dot_product < 0.0)
        for (PetscInt d = 0; d < 2; d++) face_normal[d] = -face_normal[d];

      // Get number of neighbour
      PetscInt nsupport = 0;
      const PetscInt *support;
      PetscCall(DMPlexGetSupportSize(dm, f, &nsupport));  // Should be <= 2
      PetscAssert(nsupport <= 2, NULL, 1, "Face has more than 2 neighbours");

      if (nsupport == 1)  // Boundary face
      {
        // This is where you implement a boundary condition.
      }
      else if (nsupport == 2)
      {
        // Get neighbour id
        PetscCall(DMPlexGetSupport(dm, f, &support));
        PetscInt neigh = support[0] == c ? support[1] : support[0];
        // Get neighbour properties
        PetscScalar neigh_centroid[2];
        PetscCall(DMPlexComputeCellGeometryFVM(dm, neigh, NULL, neigh_centroid, NULL));
        PetscScalar neigh_to_face_centroid[2] = {face_centroid[0] - neigh_centroid[0], face_centroid[1] - neigh_centroid[1]};
        PetscScalar dist_face_cell = PetscSqrtReal(PetscSqr(cell_to_face_centroid[0]) + PetscSqr(cell_to_face_centroid[1]));
        PetscScalar dist_face_neigh = PetscSqrtReal(PetscSqr(neigh_to_face_centroid[0]) + PetscSqr(neigh_to_face_centroid[1]));
        // Interpolate field at face center
        PetscScalar face_value =
            (dist_face_neigh * data_array[c - cell_start] + dist_face_cell * data_array[neigh - cell_start]) / (dist_face_neigh + dist_face_cell);
        // Add Gauss gradient contribution
        data_gradx_array[c - cell_start] += face_value * face_normal[0] * face_area;
      }
    }
    data_gradx_array[c - cell_start] /= cell_volume;

    // Verify that the computed cell volume with Gauss theorem is correct
    PetscAssert(PetscAbs(cell_volume_gauss - cell_volume) < PETSC_MACHINE_EPSILON, NULL, 1, "Computed cell volume different than returned by PETSc");
  }

  PetscCall(VecRestoreArray(data, &data_array));
  PetscCall(VecRestoreArray(data_gradx, &data_gradx_array));

  // Write DM in VTK format (readable with Paraview)
  PetscViewer viewer = NULL;
  PetscCall(PetscViewerVTKOpen(PETSC_COMM_WORLD, "mesh.vtu", FILE_MODE_WRITE, &viewer));
  Vec data_global, data_gradx_global;
  PetscCall(DMCreateGlobalVector(dm, &data_global));
  PetscCall(DMCreateGlobalVector(dm, &data_gradx_global));
  PetscCall(PetscObjectSetName((PetscObject) data_global, "Scalar"));
  PetscCall(PetscObjectSetName((PetscObject) data_gradx_global, "Scalar_GradX"));
  PetscCall(DMLocalToGlobal(dm, data, INSERT_VALUES, data_global));
  PetscCall(DMLocalToGlobal(dm, data_gradx, INSERT_VALUES, data_gradx_global));
  PetscCall(VecView(data_global, viewer));
  PetscCall(VecView(data_gradx_global, viewer));
  PetscCall(DMView(dm, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());

  return 0;
}