Tractography issue demonstration
================================

This repository contains a series of Jupyter notebooks that demonstrate
multiple issues that can arise when dealing with tractography data.

- ``tracking_spatial_mismatch.ipynb``: Issues that arise when the wrong spatial
  information is used when tracking and serializing the resulting data.
- ``lps_vs_ras.ipynb``: Interpretation of tractography files containing data
  in LPS and RAS conventions.
- ``trackvis_corner_vs_center.ipynb``: Issues derived from considering the
  origin at the corner of center of the voxel for ``trackvis`` (``*.trk``)
  files across different versions of the ``NiBabel`` API.

In order to run each notebook, the appropriate dependencies need to be
installed employing the corresponding requirements files.
