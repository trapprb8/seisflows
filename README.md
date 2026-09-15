SeisFlows 
==========

[![PyPI Version](https://img.shields.io/pypi/v/seisflows.svg)](https://pypi.python.org/pypi/seisflows)
[![Documentation Status](https://readthedocs.org/projects/seisflows/badge/?version=devel)](https://seisflows.readthedocs.io/en/devel/?badge=devel)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![SCOPED](https://img.shields.io/endpoint?url=https://runkit.io/wangyinz/scoped/branches/master/adjTomo)](https://github.com/SeisSCOPED/container/pkgs/container/adjtomo)

SeisFlows is an open-source, Python-based waveform inversion package that tackles the problems of full waveform inversion, seismic migration, and adjoint tomography. 

With a user base in both academia and industry, this package has been used for production scale inversions, some with over a billion model parameters, for research problems related to oil and gas exploration, earthquake seismology and general nonlinear optimization problems.

---

- SeisFlows is under current active development, and is bundled with other inversion software under the [adjTomo organization](https://github.com/adjtomo).

- Documentation, including install instructions, example problems and API, can be found on [Read the Docs](https://seisflows.readthedocs.io).

- If you find any issues, have questions, or would like to join the community, please feel free to open up a [GitHub Issue](https://github.com/adjtomo/seisflows/issues) or [start a discussion](https://github.com/orgs/adjtomo/discussions). 


SFB 1683 / B04 — Patch MAX (devel branch)
------------------------------------------

This fork extends SeisFlows for non-destructive testing of concrete foundation slabs, developed at Ruhr-Universität Bochum (RUB) as part of **SFB 1683** (Collaborative Research Centre "Modular Re-use of Existing Structures"), subproject **B04**. The goal is to assess the reusability of foundation slabs using Full Waveform Inversion (FWI).

### Reference

Trapp, M., Nestorović, T. *Three-step full-waveform inversion for the joint reconstruction of material and voids*. Mechanical Systems and Signal Processing, under review.

### Extensions in this fork (`seisflows/solver/specfem.py`)

- **Gradient masking** (`mask_sr`, `mask_top_layer`): Suppresses near-source, near-receiver and free-surface artefacts in the gradient; configurable taper.
- **Model parameter bounds** (`limit_vpvs`, `limit_poisson`): Keeps vp/vs within physical ranges; enforces Poisson's ratio bounds with a user-selectable strategy (`vp_from_vs` or `vs_from_vp`); fluid/void regions skipped via `nu_skip_vs_below`.
- **Kernel size correction** (`_crop_kernel_to_model_size`): Ensures kernel binary files match the model file length exactly (SPECFEM2D compatibility).
- **`seisflows plot2d_truemodel`**: Plots the true model even when it is meshed differently from the initial model.

### Extensions in this fork (`seisflows/preprocess/pyaflowa.py`)

- **`fix_windows: FULLTRACE`**: Uses the entire trace as the misfit window instead of PyFlex windowing; optionally bounded by `window_starttime`/`window_endtime`.
- **`win_mode: relative_t0`**: Geometrically computed misfit window — window start = source–receiver distance / `vp_ref` + offsets. Robust for varying source positions.
- **`obs_data_format`**: Configurable format for observed data (e.g. `ascii`).

### Extensions in this fork (`seisflows/system/slurm.py`)

- **`partition` / `submit_to`**: Setup job and workflow job can run on different SLURM partitions.
- **`_partitions` dictionary**: Cluster topology (cores per partition) configurable.

### Installation

```bash
conda env remove -n seisflows
conda env create -f environment.yml   # includes pypdf<4, numpy<2.0.0
# Copy the EXAMPLES folder out before installing
conda activate seisflows
pip install -e .
```

### Running a workflow

```bash
conda activate seisflows_slurm

# Automated multi-stage run (SLURM):
python 01_iteration_plan_slurm.py --plan iteration_plan.yaml --assume-yes
tail -n +1 -F slurm.setup.current.out slurm.current.out sflog.txt

# Local run (Stage 3 workstation):
python 01_iteration_plan.py --plan iteration_plan.yaml
```


References
----------
If you use this package in your own research, please cite the following papers:

- Bryant Chow, Yoshihiro Kaneko, Carl Tape, Ryan Modrak, John Townend, *An automated workflow for adjoint tomography     —waveform misfits and synthetic inversions for the North Island, New Zealand*, Geophysical Journal International, Volume 223, Issue 3, December 2020, Pages 1461–1480, https://doi.org/10.1093/gji/ggaa381

- Ryan Modrak, Dmitry Borisov, Matthieu Lefebvre, Jeroen Tromp; *SeisFlows—Flexible waveform inversion software*, Computers & Geosciences, Volume 115, June 2018, Pages 88-95, https://doi.org/10.1016/j.cageo.2018.02.004

The following paper can also be cited relative to this software:

- Ryan Modrak, Jeroen Tromp; *Seismic waveform inversion best practices: regional, global and exploration test cases*, Geophysical Journal International, Volume 206, Issue 3, 1 September 2016, Pages 1864–1889, https://doi.org/10.1093/gji/ggw202


