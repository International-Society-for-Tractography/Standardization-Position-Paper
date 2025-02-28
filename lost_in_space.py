# encoding: utf-8
# Author: Francois Rheault
# Date: 2025-09-01

"""
This script is a demonstration of the different cases of affine mismatch
between the streamline generator, the tractogram instanciation and the file
header.

It generates a set of streamlines from a probabilistic direction getter and
saves them in different files with different configurations.
"""

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames, small_sphere
from dipy.direction import ProbabilisticDirectionGetter
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.utils import get_reference_info, create_tractogram_header
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion

import nibabel as nib
from nibabel.streamlines import detect_format
from nibabel.streamlines.tractogram import Tractogram
import numpy as np

# Prepare the data
fname, bval_fname, bvec_fname = get_fnames(name="stanford_hardi")
label_fname = get_fnames(name="stanford_labels")

data, affine, img = load_nifti(fname, return_img=True)
labels = load_nifti_data(label_fname)
bvals, bvecs = read_bvals_bvecs(bval_fname, bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

seed_mask = labels == 2
white_matter = (labels == 1) | (labels == 2)
seeds = utils.seeds_from_mask(seed_mask, affine, density=1)

# Fit the CSD model
response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=6)
csd_fit = csd_model.fit(data, mask=white_matter)

# Tissue Classifier
csa_model = CsaOdfModel(gtab, sh_order_max=6)
gfa = csa_model.fit(data, mask=white_matter).gfa
stopping_criterion = ThresholdStoppingCriterion(gfa, 0.25)

fod = csd_fit.odf(small_sphere)
pmf = fod.clip(min=0)
prob_dg = ProbabilisticDirectionGetter.from_pmf(pmf, max_angle=30.0,
                                                sphere=small_sphere)

affine, dimensions, voxel_sizes, voxel_order = get_reference_info(img)

# Case #1 - Matching Affine for Tracking and Tractogram (in RASMM) + Correct Affine Header
# Generates a file with streamlines correctly positioned in both VOX and RASMM space,
# and a file header that matches its associated NIFTI.
seeds = utils.seeds_from_mask(seed_mask, affine, density=1)
streamline_generator = LocalTracking(prob_dg, stopping_criterion,
                                     seeds, affine, step_size=0.5)
streamlines = list(streamline_generator)

filename = "tracking_match_rasmm_header_good.trk"
tractogram_type = detect_format(filename)
header = create_tractogram_header(tractogram_type,
                                  affine, dimensions,
                                  voxel_sizes, voxel_order)
new_tractogram = Tractogram(streamlines, affine_to_rasmm=np.eye(4))

fileobj = tractogram_type(new_tractogram, header=header)
nib.streamlines.save(fileobj, filename)


# Case #2 - Mismatched Affine between Tracking and Tractogram + Correct Affine Header
# Generates a file with streamlines correctly written in VOX space but not in RASMM space,
# and a file header that matches its associated NIFTI.
seeds = utils.seeds_from_mask(seed_mask, np.eye(4), density=1)
streamline_generator = LocalTracking(prob_dg, stopping_criterion,
                                     seeds, np.eye(4), step_size=0.5)
streamlines = list(streamline_generator)

filename = "tracking_unmatch_header_good.trk"

tractogram_type = detect_format(filename)
header = create_tractogram_header(tractogram_type,
                                  affine, dimensions,
                                  voxel_sizes, voxel_order)
new_tractogram = Tractogram(streamlines, affine_to_rasmm=np.eye(4)) # WRONG

fileobj = tractogram_type(new_tractogram, header=header)
nib.streamlines.save(fileobj, filename)

# Case #3 - Matching Affine for Tracking and Tractogram (in VOX) + Correct Affine Header
# Generates a file with streamlines correctly positioned in both VOX and RASMM space,
# and a file header that matches its associated NIFTI.
seeds = utils.seeds_from_mask(seed_mask, np.eye(4), density=1)
streamline_generator = LocalTracking(prob_dg, stopping_criterion,
                                     seeds, np.eye(4), step_size=0.5)
streamlines = list(streamline_generator)

filename = "tracking_match_vox_header_good.trk"

tractogram_type = detect_format(filename)
header = create_tractogram_header(tractogram_type,
                                  affine, dimensions,
                                  voxel_sizes, voxel_order)
new_tractogram = Tractogram(streamlines, affine_to_rasmm=affine) # RIGHT

fileobj = tractogram_type(new_tractogram, header=header)
nib.streamlines.save(fileobj, filename)


# Case #4 - Matching Affine for Tracking but Re-used for Tractogram + Correct Affine Header
# Generates a file with streamlines correctly generated in RASMM space but moved again using RASMM affine before writing,
# and a file header that matches its associated NIFTI.
seeds = utils.seeds_from_mask(seed_mask, affine, density=1)
streamline_generator = LocalTracking(prob_dg, stopping_criterion,
                                     seeds, affine, step_size=0.5)
streamlines = list(streamline_generator)

filename = "tracking_unmatch_twice_header_good.trk"

tractogram_type = detect_format(filename)
header = create_tractogram_header(tractogram_type,
                                  affine, dimensions,
                                  voxel_sizes, voxel_order)
new_tractogram = Tractogram(streamlines, affine_to_rasmm=affine) # WRONG

fileobj = tractogram_type(new_tractogram, header=header)
nib.streamlines.save(fileobj, filename)


# Case #5 - Matching Affine for Tracking and Tractogram (in RASMM) + Incorrect Affine Header
# Generates a file with streamlines correctly positioned in both VOX and RASMM space,
# but with a file header that does not match its associated NIFTI.
seeds = utils.seeds_from_mask(seed_mask, affine, density=1)
streamline_generator = LocalTracking(prob_dg, stopping_criterion,
                                     seeds, affine, step_size=0.5)
streamlines = list(streamline_generator)

filename = "tracking_match_header_bad.trk"

tractogram_type = detect_format(filename)
header = create_tractogram_header(tractogram_type,
                                  np.eye(4), dimensions, # WRONG
                                  voxel_sizes, voxel_order)
new_tractogram = Tractogram(streamlines, affine_to_rasmm=np.eye(4))

fileobj = tractogram_type(new_tractogram, header=header)
nib.streamlines.save(fileobj, filename)

# Case #6 - Matching Affine for Tracking (in VOX) but not in Tractogram + Incorrect Affine Header
# Generates a file with streamlines correctly positioned in VOX space but not in RASMM space,
# and with a file header that does not match its associated NIFTI.
seeds = utils.seeds_from_mask(seed_mask, np.eye(4), density=1)
streamline_generator = LocalTracking(prob_dg, stopping_criterion,
                                     seeds, np.eye(4), step_size=0.5)
streamlines = list(streamline_generator)

filename = "tracking_bad_header_bad.trk"

tractogram_type = detect_format(filename)
header = create_tractogram_header(tractogram_type,
                                  np.eye(4), dimensions, # wrong
                                  voxel_sizes, voxel_order)
new_tractogram = Tractogram(streamlines, affine_to_rasmm=np.eye(4)) # wrong

fileobj = tractogram_type(new_tractogram, header=header)
nib.streamlines.save(fileobj, filename)