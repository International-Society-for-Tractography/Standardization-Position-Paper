#!/usr/bin/env python
# Copyright International Society for Tractography (IST)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import default_sphere, get_fnames
from dipy.data.fetcher import dipy_home
from dipy.direction import peaks_from_model
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.tracker import probabilistic_tracking
from dipy.tracking.utils import seeds_from_mask


hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")
label_fname = get_fnames(name="stanford_labels")

data, affine, img = load_nifti(hardi_fname, return_img=True)
labels = load_nifti_data(label_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

seed_mask = labels == 2
seeds = seeds_from_mask(seed_mask, affine, density=2)

seeds_img = nib.Nifti1Image(seeds, affine)
seeds_fname = Path(dipy_home) / "tracking_seeds.nii.gz"
nib.save(seeds_img, seeds_fname)

white_matter = (labels == 1) | (labels == 2)

white_matter_img = nib.Nifti1Image(white_matter.astype(np.uint16), affine)
white_matter_fname = Path(dipy_home) / "wm.nii.gz"
nib.save(white_matter_img, white_matter_fname)

sc = BinaryStoppingCriterion(white_matter)

response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=6)
csd_fit = csd_model.fit(data, mask=white_matter)

fod = csd_fit.odf(default_sphere)

fod_img = nib.Nifti1Image(fod, affine)
fod_fname = Path(dipy_home) / "fod.nii.gz"
nib.save(fod_img, fod_fname)

streamline_generator = probabilistic_tracking(
    seeds,
    sc,
    affine,
    sf=fod,
    random_seed=1,
    sphere=default_sphere,
    max_angle=30.0,
    step_size=0.5,
)

streamlines = Streamlines(streamline_generator)
sft = StatefulTractogram(streamlines, img, Space.RASMM)

sft_fname = Path(dipy_home) / "tractogram_probabilistic_sf.trk"
save_trk(sft, sft_fname)
