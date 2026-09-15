#!/usr/bin/env python3
"""
This Solver module is in charge of interacting with external numerical solvers
such as SPECFEM (2D/3D/3D_GLOBE). This SPECFEM base class provides general
functions that work with all versions of SPECFEM. Subclasses will provide
additional capabilities unique to each version of SPECFEM.

.. note::
    The Base class implementation is almost completely SPECFEM2D related.
    However, SPECFEM2D requires a few unique parameters that 3D/3D_GLOBE
    do not. Because of the inheritance architecture of SeisFlows, we do not
    want the 3D and 3D_GLOBE versions to inherit 2D-specific parameters, so
    we need this this more generalized SPECFEM base class.

TODO
    - add in `apply_hess` functionality that was partially written in legacy code
    - move `_initialize_adjoint_traces` to workflow.migration
    - Add density scaling based on Vp?
"""
import os
import sys
import shlex
from seisflows.tools import unix
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, wait
from glob import glob
import numpy as np
from seisflows import logger
from seisflows.tools import msg, unix
from seisflows.tools.config import get_task_id, Dict
from seisflows.tools.model import Model
from seisflows.tools.specfem import (getpar, setpar, check_source_names,
                                      read_fortran_binary, write_fortran_binary)

####PATCH MAX####
# --- begin: per-proc size alignment helper ---
import re
from pathlib import Path
import numpy as np

import os, re
from pathlib import Path
import numpy as np

#### PATCH MAX ####

#Hilfsfunktion: kann später denk ich weg:
def _find_global_surface_z(model_init_dir, zs_hint=None, logger=None):
    """Liest alle proc??????_z.bin aus model_init_dir und liefert den globalen Oberflächen-z.
       Wenn zs_hint (Quellentiefe) gegeben ist, wird die Richtung automatisch erkannt."""
    zmins, zmaxs = [], []
    for zfile in sorted(Path(model_init_dir).glob("proc??????_z.bin")):
        try:
            zz = np.fromfile(zfile, dtype=np.float32)
        except Exception:
            continue
        if zz.size:
            zmins.append(float(zz.min()))
            zmaxs.append(float(zz.max()))
    if not zmins:
        if logger: logger.warning("[mask] cannot determine global surface z (no z.bin found)")
        return None

    gmin, gmax = min(zmins), max(zmaxs)
    if zs_hint is None:
        # Default: Oberfläche = größter z-Wert (häufig in Specfem2D)
        zsurf = gmax
    else:
        # Nimm das Ende, das näher an der Quellentiefe liegt
        zsurf = gmax if abs(gmax - zs_hint) <= abs(zs_hint - gmin) else gmin

    if logger: logger.info(f"[mask] global surface z: {zsurf:.6f} (zmin={gmin:.6f}, zmax={gmax:.6f}, hint={zs_hint})")
    return zsurf


def _find_global_x_bounds(model_init_dir, logger=None):
    """Liest alle proc??????_x.bin aus model_init_dir und liefert den globalen
       linken/rechten Modellrand (x_left, x_right)."""
    xmins, xmaxs = [], []
    for xfile in sorted(Path(model_init_dir).glob("proc??????_x.bin")):
        try:
            xx = np.fromfile(xfile, dtype=np.float32)
        except Exception:
            continue
        if xx.size:
            xmins.append(float(xx.min()))
            xmaxs.append(float(xx.max()))
    if not xmins:
        if logger: logger.warning("[mask] cannot determine global x bounds (no x.bin found)")
        return None, None

    x_left, x_right = min(xmins), max(xmaxs)
    if logger: logger.info(f"[mask] global x bounds: left={x_left:.6f}, right={x_right:.6f}")
    return x_left, x_right


####PATCH MAX: lokal begrenztes Smoothing um Void-Waende####
# Motivation: xsmooth_sem glaettet global mit einer einzigen Spannweite.
# Scharfe, einspringende Void-Ecken sind FEM-Singularitaeten (keine lokale
# Netzverfeinerung, kein Smoothing -> Rauschen bis weit ausserhalb des
# physikalischen vp/vs-Bereichs, siehe Analyse). Globales Smoothing daempft
# dieses Randrauschen zwar, verwischt aber gleichzeitig echte, aehnlich
# kleine Materialanomalien (Einschluesse) im ganzen Modellgebiet.
# Diese Erweiterung mischt die global geglaettete Kernel-Datei nur INNERHALB
# eines schmalen Puffers um die erkannten Void-Waende ein; ausserhalb bleibt
# der rohe (ungeglaettete) Kernel unveraendert. Ueber den SeisFlows-Parameter
# `local_void_smooth_buffer_m` steuerbar (z.B. per `seisflows par
# local_void_smooth_buffer_m 0.02` in 1_SpecFEM_setup.py), 0 (Default) = aus.
# WICHTIG: `smooth_h`/`smooth_v` muessen weiterhin > 0 sein, sonst hat
# xsmooth_sem nichts Sinnvolles zum Einmischen (siehe smooth()-Aufruf unten).


def _void_boundary_weight(x, z, buffer_m, cell_size_factor=3.5,
                           edge_margin_m=None, logger=logger):
    """
    Gewichtsfeld in [0,1] je GLL-Punkt (x,z): 1 direkt an einer Void-Wand,
    linear abfallend auf 0 ab `buffer_m` Abstand. Bestimmt Void-Waende ueber
    Rasterbelegung (regelmaessiges Binning, Zellgroesse deutlich groesser als
    der typische GLL-Punktabstand, aber viel kleiner als jede Anomalie):
    belegte Zellen mit mindestens einem leeren 3x3-Nachbarn sind Randzellen.

    HINWEIS: eine reine Nachbarschafts-DICHTE (statt Rasterbelegung) waere
    hier ungeeignet -- GLL-Knoten liegen INNERHALB eines Spektralelements
    absichtlich ungleichmaessig (Gauss-Lobatto-Legendre-Verteilung, dichter
    an Elementkanten), das wuerde grosse Teile des Volumenmaterials faelschlich
    als "duenn besetzt" markieren (empirisch geprueft: >30% des gesamten
    Gebiets statt nur der beiden echten Voids).

    Die Aussenkante der Modelldomaene wird ueber einen physischen Randstreifen
    (`edge_margin_m`, per Default max(2cm, 3 Zellen)) ausgeschlossen, da die
    Punktwolke dort ebenfalls "leere" Nachbarzellen ausserhalb der Bounding-Box
    hat. Materialkontraste (Einschluesse) erzeugen KEINE leeren Zellen und
    werden dadurch nicht als Rand erkannt -- das ist beabsichtigt.
    """
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree

    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.zeros(0)

    # typischer GLL-Punktabstand; Duplikate an Elementkanten (Abstand 0)
    # zuerst entfernen, sonst wird die Median-Schaetzung durch sie verzerrt
    keys = np.round(np.column_stack([x, z]), 6)
    uniq = np.unique(keys, axis=0)
    if len(uniq) < 2:
        if logger:
            logger.warning("[local_smooth] zu wenige eindeutige Punkte -> "
                            "kein lokales Smoothing angewendet")
        return np.zeros(n)
    d_nn, _ = cKDTree(uniq).query(uniq, k=2)
    typical_spacing = float(np.median(d_nn[:, 1]))
    if not np.isfinite(typical_spacing) or typical_spacing <= 0:
        if logger:
            logger.warning("[local_smooth] konnte Rasterabstand nicht bestimmen -> "
                            "kein lokales Smoothing angewendet")
        return np.zeros(n)

    cell = cell_size_factor * typical_spacing
    x_lo, x_hi = x.min(), x.max()
    z_lo, z_hi = z.min(), z.max()
    nx = int(np.ceil((x_hi - x_lo) / cell)) + 2
    nz = int(np.ceil((z_hi - z_lo) / cell)) + 2
    ix = np.clip(((x - x_lo) / cell).astype(int), 0, nx - 1)
    iz = np.clip(((z - z_lo) / cell).astype(int), 0, nz - 1)
    occ = np.zeros((nz, nx), dtype=bool)
    occ[iz, ix] = True

    interior = binary_erosion(occ, structure=np.ones((3, 3), dtype=bool))
    boundary_cells = occ & ~interior

    if edge_margin_m is None:
        edge_margin_m = max(0.02, 3 * cell)
    edge_cells = int(np.ceil(edge_margin_m / cell))
    outer = np.zeros_like(occ)
    outer[:edge_cells, :] = True
    outer[-edge_cells:, :] = True
    outer[:, :edge_cells] = True
    outer[:, -edge_cells:] = True
    void_wall_cells = boundary_cells & ~outer

    n_cells = int(void_wall_cells.sum())
    if logger:
        logger.info(f"[local_smooth] {n_cells} Void-Wand-Zellen erkannt "
                    f"(Zellgroesse {cell*1000:.1f}mm, Randstreifen "
                    f"{edge_margin_m*1000:.0f}mm, Puffer {buffer_m*1000:.0f}mm)")
    if n_cells == 0:
        return np.zeros(n)

    zc, xc = np.where(void_wall_cells)
    wall_pts = np.column_stack([x_lo + (xc + 0.5) * cell, z_lo + (zc + 0.5) * cell])
    wall_tree = cKDTree(wall_pts)
    dist, _ = wall_tree.query(np.column_stack([x, z]), k=1)
    return np.clip(1.0 - dist / buffer_m, 0.0, 1.0)


def _blend_local_void_smoothing(input_path, output_path, parameters, ext,
                                 buffer_m, logger=logger):
    """
    Ersetzt die global (xsmooth_sem-)geglaetteten Kernel-Dateien in
    `output_path` durch eine Mischung aus roh (`input_path`) und geglaettet:
    volle Glaettung nur innerhalb `buffer_m` um die erkannten Void-Waende,
    ausserhalb unveraendert roh. Reine Nachbearbeitung bereits vorhandener
    Dateien -- xsmooth_sem/MPI-Aufruf bleiben unangetastet.
    """
    for xfile in sorted(Path(input_path).glob(f"proc??????_x{ext}")):
        proc = xfile.name.split("_")[0]
        zfile = Path(input_path) / f"{proc}_z{ext}"
        if not zfile.exists():
            if logger:
                logger.warning(f"[local_smooth] {zfile} fehlt -> {proc} uebersprungen")
            continue
        x = np.fromfile(xfile, dtype="float32")
        z = np.fromfile(zfile, dtype="float32")
        weight = _void_boundary_weight(x, z, buffer_m=buffer_m, logger=logger)

        for par in parameters:
            raw_file = Path(input_path) / f"{proc}_{par}_kernel{ext}"
            smooth_file = Path(output_path) / f"{proc}_{par}_kernel{ext}"
            if not (raw_file.exists() and smooth_file.exists()):
                if logger:
                    logger.warning(f"[local_smooth] {raw_file.name} oder "
                                    f"{smooth_file.name} fehlt -> uebersprungen")
                continue
            raw = np.fromfile(raw_file, dtype="float32")
            smoothed = np.fromfile(smooth_file, dtype="float32")
            if raw.shape != weight.shape or smoothed.shape != weight.shape:
                if logger:
                    logger.warning(f"[local_smooth] Groessen passen nicht zusammen "
                                    f"({proc}_{par}) -> uebersprungen")
                continue
            blended = (weight * smoothed + (1.0 - weight) * raw).astype("float32")
            blended.tofile(smooth_file)
        if logger:
            logger.info(f"[local_smooth] {proc}: lokal geglaettet "
                        f"(Puffer {buffer_m*1000:.0f}mm um Void-Wand)")
#### PATCH MAX ####


def _crop_kernel_to_model_size(
    kfile,
    model_dir=None,
    dtype="float32",
    backup=True,
    logger=logger,
):
    """
    Bringt ein einzelnes Kernel-Binärfile (…/procXXXXXX_{vp|vs}_kernel.bin)
    exakt auf die Länge der entsprechenden MODEL-Datei
    (…/OUTPUT_FILES_INIT/procXXXXXX_{vp|vs}.bin).

    Return:
      >0  : Anzahl abgeschnittener Elemente
       0  : nichts geändert
    """
    itemsize = np.dtype(dtype).itemsize
    kpath = Path(kfile)

    # proc + Feld erkennen (vp, vs, optional rho)
    m = re.match(r"^(proc\d{6})_([a-zA-Z0-9]+)_kernel\.bin$", kpath.name)
    if not m:
        if logger:
            logger.debug(f"[crop] übersprungen (Pattern passt nicht): {kpath.name}")
        return 0
    proc, field = m.group(1), m.group(2)

    # --- Modell-Datei suchen ---
    candidates = []
    if model_dir is not None:
        candidates.append(Path(model_dir))
    else:
        # häufige Orte – Reihenfolge: bevorzugt OUTPUT_FILES_INIT
        candidates += [
            Path.cwd() / "specfem2d_workdir" / "OUTPUT_FILES_INIT",
            Path.cwd() / "OUTPUT_FILES_INIT",
        ]
        # falls du self.path._mainsolver oder self.path.model_databases hast:
        try:
            from . import paths  # falls du eine Pfadklasse hast; sonst ignorieren
            ms = Path(paths.Path._mainsolver) / "specfem2d_workdir" / "OUTPUT_FILES_INIT"
            candidates.append(ms)
        except Exception:
            pass

    mfile = None
    for c in candidates:
        candidate = Path(c) / f"{proc}_{field}.bin"
        if candidate.exists():
            mfile = candidate
            break

    if mfile is None:
        if logger:
            logger.warning(f"[crop] Model-Datei nicht gefunden für {proc}_{field}. "
                           f"Gesucht in: {', '.join(str(p) for p in candidates)}")
        return 0

    try:
        expected_elems = os.path.getsize(mfile) // itemsize
    except OSError as e:
        if logger: logger.warning(f"[crop] Kann Größe nicht lesen: {mfile}: {e}")
        return 0

    try:
        k_bytes = os.path.getsize(kpath)
    except OSError as e:
        if logger: logger.warning(f"[crop] Kernel fehlt/unerreichbar: {kpath}: {e}")
        return 0

    if k_bytes % itemsize != 0:
        if logger:
            logger.warning(f"[crop] Unerwartete Kernel-Bytegröße (kein Vielfaches von {itemsize}): {kpath}")
        # hier dennoch weiter, SPECFEM kann sonst stolpern
    k_elems = k_bytes // itemsize

    if k_elems == expected_elems:
        return 0

    # Backup
    bak = kpath.with_suffix(kpath.suffix + ".precrop")
    if backup and not bak.exists():
        try:
            kpath.replace(bak)
            # weiterarbeiten ab Backup
            kpath = bak  # Quelle bleibt bak; Ziel wird der ursprüngliche Name ohne .precrop
        except OSError as e:
            if logger: logger.warning(f"[crop] Backup fehlgeschlagen: {bak}: {e}")

    if k_elems > expected_elems:
        # Ziel: ursprünglicher Dateiname ohne .precrop
        out_path = Path(str(kpath).replace(".precrop", ""))
        # wir kopieren nur die ersten expected_elems*itemsize Bytes
        nbytes = expected_elems * itemsize
        with open(kpath, "rb") as fin, open(out_path, "wb") as fout:
            # chunked copy
            left = nbytes
            bufsize = 1024 * 1024
            while left > 0:
                chunk = fin.read(min(bufsize, left))
                if not chunk:
                    break
                fout.write(chunk)
                left -= len(chunk)
        if logger:
            logger.info(f"[crop] {out_path.name}: {k_elems} -> {expected_elems} "
                        f"(-{k_elems - expected_elems} Elemente)")
        return k_elems - expected_elems

    # Kernel kleiner als Modell – kein Padding vornehmen
    if logger:
        logger.warning(f"[crop] {Path(str(kpath).replace('.precrop','')).name}: "
                       f"Kernel kleiner als Model ({k_elems} < {expected_elems}); kein Padding.")
    # Falls wir ein Backup angelegt haben und noch keine gültige Zieldatei existiert:
    out_path = Path(str(kpath).replace(".precrop", ""))
    if backup and not out_path.exists():
        # Original (bak) zurückspielen
        try:
            Path(kpath).replace(out_path)
        except OSError:
            pass
    return 0



class Specfem:
    """
    Solver SPECFEM [Solver Base]
    ----------------------------
    Defines foundational structure for Specfem-based solver module. 
    Generalized SPECFEM interface to manipulate SPECFEM2D/3D/3D_GLOBE w/ Python

    Parameters
    ----------
    :type syn_data_format: str
    :param syn_data_format: data format for reading synthetic traces into memory.
        Available: ['SU': seismic unix format, 'ASCII': human-readable ascii]
    :type materials: str or list
    :param materials: Material name used to define the model parameters that
        will be updated during an Inversion workflow. Available options 
        (case-insensitive):
        - `type: list` (2D, 3D, 3D_GLOBE): User-defined list of lower-case
            parameters (e.g., ['vp', 'vs'] to mimic 'ELASTIC')
            NOTE: User is responsible for understanding if their chosen 
            parameters are actually represented in SPECFEM, there are no guard
            rails here to protect incorrect parameter naming
        - ACOUSTIC (2D, 3D, 3D_GLOBE): vp 
        - ELASTIC (2D, 3D, 3D_GLOBE): vp, vs 
        - TRANSVERSE_ISOTROPIC (3D, 3D_GLOBE): vpv, vph, vsv, vsh, eta
        - 2D_ANISOTROPIC (2D): c11 c13 c15 c33 c35 c55 c12 c23 c25 c22
        - ANISOTROPIC (3D, 3D_GLOBE): c_ij  (21 parameter anisotropy)
        
    :type update_density: bool
    :param update_density: How to treat density during inversion. If True, 
        updates density during inversion. If False, keeps it constant.
        TODO allow density scaling during an inversion
    :type attenuation: bool
    :param attenuation: How to treat attenuation during inversion.
        if True, turns on attenuation during forward simulations only. If
        False, attenuation is always set to False. Requires underlying
        attenution (Q_mu, Q_kappa) model
    :type smooth_h: float
    :param smooth_h: Gaussian half-width for horizontal smoothing in units
        of meters. If 0., no smoothing applied. Only applicable for workflows:
        ['migration', 'inversion'], ignored for 'forward' workflow.
        SPECFEM3D_GLOBE only: if `smooth_type`=='laplacian' then this is just 
        the X and Y extent of the applied smoothing
    :type smooth_v: float
    :param smooth_v: Gaussian half-width for vertical smoothing in units
        of meters. Only applicable for workflows: ['migration', 'inversion'],
        ignored for 'forward' workflow.
        SPECFEM3D_GLOBE only: if `smooth_type`=='laplacian' then this is just 
        the Z extent of the applied smoothing
    :type smooth_type: str
    :param smooth_type: choose how smoothing is performed for gradients.
        these are tied to the internal smoothing functions available, and only
        certain code flavors have certain smoothing functions available:
        - 'gaussian' [2D/3D/3D_GLOBE]: Default, convolve with a 3D gaussian, 
            slow and computationally intensive. Only option for 2D.
        - 'laplacian' [3D_GLOBE]: RECOMMENDED FOR 3D_GLOBE. Average points 
            around vertex to smooth. Much faster than Gaussian.
        - 'pde' [3D]: RECOMMENDED for 3D. Diffusion-based PDE smoothing. 
            Much faster than Gaussian. See SPECFEM3D PR#1725
    :type smooth_use_gpu: bool
    :param smooth_use_gpu: Use GPU acceleration for xsmooth_sem (passes
        '.true' instead of '.false'). Requires CUDA-compiled binaries and an
        available GPU. For large meshes (>500k GLL points) with sigma > a few
        element widths, CPU smoothing can take hours; GPU is typically 100x
        faster. Default: False.
    :type components: str
    :param components: components to search for synthetic data with. None by
        default which uses a wildcard when searching for synthetics. If
        provided, User only wants to use a subset of components generated by
        SPECFEM. In that case, `components` should be string of letters such
        as 'ZN' (for up and north components)
    :type solver_io: str
    :param solver_io: format of model/kernel/gradient files expected by the
        numerical solver. Available: ['fortran_binary': default .bin files].
        TODO: ['adios': ADIOS formatted files]
    :type source_prefix: str
    :param source_prefix: prefix of source/event/earthquake files. If None,
        will attempt to guess based on the specific solver chosen.
    :type mpiexec: str
    :param mpiexec: MPI executable used to run parallel processes. Should also
        be defined for the system module

    Paths
    -----
    :type path_data: str
    :param path_data: path to any externally stored waveform data required for 
        data-synthetic comparison
    :type path_specfem_bin: str
    :param path_specfem_bin: path to SPECFEM bin/ directory which
        contains binary executables for running SPECFEM
    :type path_specfem_data: str
    :param path_specfem_data: path to SPECFEM DATA/ directory which must
        contain the CMTSOLUTION, STATIONS and Par_file files used for
        running SPECFEM
    ***
    """
    def __init__(self, syn_data_format="ascii", materials="acoustic",
                 update_density=False, nproc=1, ntask=1, attenuation=False,
                 smooth_h=0., smooth_v=0., smooth_type="gaussian",
                 components=None, source_prefix=None, mpiexec=None,
                 workdir=os.getcwd(), path_solver=None, path_eval_grad=None,
                 path_data=None, path_specfem_bin=None, path_specfem_data=None,
                 path_model_init=None, path_model_true=None, path_output=None,
                 # >>> Neu:
                 limit_vpvs=False,
                 vp_min=None,
                 vp_max=None,
                 vs_min=None,
                 vs_max=None,
                 # --- Poisson control (optional) ---
                 limit_poisson=False,
                 nu_min=0.05,
                 nu_max=0.45,
                 nu_strategy="vp_from_vs",     # or: "vs_from_vp"
                 nu_skip_vs_below=1.0,                                       
                 mask_sr=False,
                 mask_src_radius_m=0.0,
                 mask_rec_radius_m=0.0,
                 mask_taper_m=0.0,               
                 # --- NEU: Top-Layer-Optionen ---
                 mask_top_layer=False,
                 mask_top_thickness_m=0.0,
                 mask_top_taper_m=0.0,
                 # --- NEU: Side-Layer-Optionen (links/rechts) ---
                 mask_side_layer=False,
                 mask_side_thickness_m=0.0,
                 mask_side_taper_m=0.0,
                 smooth_use_gpu=False,
                 local_void_smooth_buffer_m=0.0,
                 **kwargs):
        """
        Set default SPECFEM interface parameters

        .. note::
            Paths listed here are shared with `workflow.forward` and so are not
            included in the class docstring.

        :type workdir: str
        :param workdir: working directory in which to look for data and store
            results. Defaults to current working directory
        :type path_solver: str
        :param path_solver: scratch path for all solver related tasks
        :type path_model_init: str
        :param path_model_init: path to the starting model used to calculate the
            initial misfit. Must match the expected `solver_io` format.
        :type path_model_true: str
        :param path_model_true: path to a target model if `case`=='synthetic' and
            a set of synthetic 'observations' are required for workflow.
        :type path_output: str
        :param path_output: shared output directory on disk for more permanent
            storage of solver related files such as traces, kernels, gradients.
        """
        
        
        # >>> Neu PATCH MAX
        self.mask_sr = bool(mask_sr)
        self.mask_src_radius_m = float(mask_src_radius_m)
        self.mask_rec_radius_m = float(mask_rec_radius_m)
        self.mask_taper_m = float(mask_taper_m)
        # --- NEU: Top-Layer speichern ---
        self.mask_top_layer = bool(mask_top_layer)
        self.mask_top_thickness_m = float(mask_top_thickness_m)
        self.mask_top_taper_m = float(mask_top_taper_m)
        # --- NEU: Side-Layer speichern (links/rechts) ---
        self.mask_side_layer = bool(mask_side_layer)
        self.mask_side_thickness_m = float(mask_side_thickness_m)
        self.mask_side_taper_m = float(mask_side_taper_m)
        self.smooth_use_gpu = bool(smooth_use_gpu)
        # --- lokal begrenztes Smoothing um Void-Waende (0 = aus, siehe smooth()) ---
        self.local_void_smooth_buffer_m = float(local_void_smooth_buffer_m)
        # --- Velocity clipping (optional) ---
        # When enabled, clamp vp and vs in every trial model before any
        # forward run. Bounds can be left as None to disable a side.
        self.limit_vpvs = bool(limit_vpvs)
        self.vp_min = None if vp_min is None else float(vp_min)
        self.vp_max = None if vp_max is None else float(vp_max)
        self.vs_min = None if vs_min is None else float(vs_min)
        self.vs_max = None if vs_max is None else float(vs_max)     
        # --- Poisson control (optional) ---
        self.limit_poisson   = bool(limit_poisson)
        self.nu_min          = float(nu_min)
        self.nu_max          = float(nu_max)
        self.nu_strategy     = str(nu_strategy)
        self.nu_skip_vs_below = float(nu_skip_vs_below)
        # Publically accessible parameters
        self.syn_data_format = syn_data_format
        self.materials = materials
        self.nproc = nproc
        self.ntask = ntask
        self.update_density = update_density
        self.attenuation = attenuation
        self.smooth_h = smooth_h
        self.smooth_v = smooth_v
        self.smooth_type = smooth_type  # can be overwritten by child class
        self.components = components
        self.source_prefix = source_prefix or "SOURCE"
        self.prune_scratch = None  # SPECFEM3D/GLOBE only

        # Define internally used directory structure
        self.path = Dict(
            scratch=path_solver or os.path.join(workdir, "scratch", "solver"),
            eval_grad=path_eval_grad or
                      os.path.join(workdir, "scratch", "eval_grad"),
            data=path_data or os.path.join(workdir, "SFDATA"),
            output=path_output or os.path.join(workdir, "output"),
            specfem_bin=path_specfem_bin,
            specfem_data=path_specfem_data,
            model_init=path_model_init,
            model_true=path_model_true,
        )
        self.path["_mainsolver"] = os.path.join(self.path.scratch, "mainsolver")
        self.path["_solver_output"] = os.path.join(self.path.output, "solver")

        # Define parameters to be updated based on `material` type.  
        # If User inputs as a list then set and forget. Some of these are Solver
        # specific and `Solver.check()` will fail the workflow if the incorrect
        # solver is used.
        self._parameters = []
        if isinstance(self.materials, list):  # Custom list e.g., ['vp', 'vs']
            self._parameters = self.materials
        elif self.materials.upper() == "ACOUSTIC":  # 2D/3D/3D_GLOBE
            self._parameters = ["vp"]
        elif self.materials.upper()  == "ELASTIC":  # 2D/3D/3D_GLOBE
            self._parameters = ["vp", "vs"]
        # Transverse Isotropic / Radially Anisotropic (2D/3D/3D_GLOBE)
        elif self.materials.upper() == "TRANSVERSE_ISOTROPIC":  # 
            self._parameters = ["vpv", "vph", "vsv", "vsh", "eta"]
        # General anisotropic in 2D (SPECFEM2D)
        elif self.materials.upper() == "2D_ANISOTROPIC":
            self._parameters = ["c11", "c13", "c15", "c33", "c35", "c55", 
                                "c12", "c23", "c25", "c22"]
        # General 21 parameter anisotropy c_ij in 3D: c11, c12... c66
        elif self.materials.upper() == "ANISOTROPIC":
            for i in range(1, 7):
                for j in range(1, 7):
                    if j >= i:
                        self._parameters.append(f"c{i}{j}")

        # Allow density to be updated. `setup` will remove doubles if 
        # `materials` wase a custom list
        if self.update_density and ("rho" not in self._parameters):
            self._parameters.append("rho")

        self._mpiexec = mpiexec
        self._source_names = None  # for property source_names
        self._ext = ""  # for database file extensions

        # Define available choices for check parameters
        self._available_model_types = ["gll"]
        self._available_materials = None  # To be overwritten by Child classes
        
        # SPECFEM2D specific attributes. Should be overwritten by 3D versions
        self._syn_available_data_formats = ["ASCII", "SU"]
        self._required_binaries = ["xspecfem2D", "xmeshfem2D", "xcombine_sem"]
        self._acceptable_source_prefixes = ["SOURCE", "FORCE", "FORCESOLUTION"]

        # Constants that will be referenced during simulations and file I/O.
        # These should be overwritten by all child classes (3D, 3D_GLOBE)
        self._fwd_simulation_executables = ["bin/xmeshfem2D", "bin/xspecfem2D"]
        self._adj_simulation_executables = ["bin/xspecfem2D"]
        self._absorb_wildcard = "absorb_*_*"
        self._forward_array_wildcard = ""

        # Empty variables that will need to be overwritten by SPECFEM3D/3D_GLOBE
        self._regions = None
        self._export_vtk = False

    def check(self):
        """
        Checks parameter validity for SPECFEM input files and model parameters
        """
        from glob import glob
        if isinstance(self.materials, str):
            assert(self.materials.upper() in self._available_materials), (
            f"Although `self.materials` is a valid material type, it is not an "
            f"available material type for this solver choice. Please re-choose")

            if self.materials.upper() == "ANISOTROPIC":
                logger.warning("the 'ANISOTROPIC' material parameter is an "
                               "experimental feature. Use at your "
                               "own risk - NOT guaranteed to work")

                anisotropic_kl = getpar(key="ANISOTROPIC_KL", 
                                        file=os.path.join(
                                            self.path.specfem_data,  
                                            "Par_file"))[1]
                
                assert(anisotropic_kl == ".true."), (
                    f"SPECFEM3D Par_file parameter 'ANISOTROPIC_KL' must be "
                    f"set to '.true.' for ANISOTROPIC parameters")

        # Check that we have set parameters correctly. Since `rho` can be added
        # separately we need to account for that when checking length of array
        if len(self._parameters) < int(self.update_density):
            raise NotImplementedError(
                f"Invalid material: {self.materials}. `materials` must be as "
                f"list of parameter names or a pre-defined label. See "
                f"parameter file docstring for more information")

        logger.debug(f"solver parameters to be updated are: {self._parameters}")

        if self.syn_data_format.upper() not in self._syn_available_data_formats:
            raise NotImplementedError(
                f"solver.syn_data_format must be "
                f"{self._syn_available_data_formats}"
            )

        # Check that User has provided appropriate binary files to run SPECFEM
        assert(self.path.specfem_bin is not None and
               os.path.exists(self.path.specfem_bin)), (
            f"`path_specfem_bin` must exist and must point to directory " 
            f"containing SPECFEM executables"
        )
        for fid in self._required_binaries:
            assert(os.path.exists(os.path.join(self.path.specfem_bin, fid))), (
                f"`path_specfem_bin`/{fid} does not exist but is required by "
                f"SeisFlows solver module"
            )

        # Make sure mpiexec is defined. We can get into a tricky situation where 
        # mpiexec gets default defined by System but is not defined for solver
        # leading to binaries getting unexpectedly run in serial mode 
        if self.nproc > 1:
            assert(self._mpiexec is not None), (
                f"Multi-core workflows (`nproc`>1) require an MPI executable "
                f"`mpiexec`"
            ) 

        # Check that SPECFEM/DATA directory exists
        assert(self.path.specfem_data is not None and
               os.path.exists(self.path.specfem_data)), (
            f"`path_specfem_data` must exist and must point to directory " 
            f"containing SPECFEM input files"
        )
        for fid in ["STATIONS", "Par_file"]:
            assert(os.path.exists(os.path.join(self.path.specfem_data, fid))), (
                f"DATA/{fid} does not exist but is required by SeisFlows solver"
            )

        # Make sure source files exist and are appropriately labeled
        assert(self.source_prefix in self._acceptable_source_prefixes), (
            f"SPECFEM `source_prefix` must be in "
            f"{self._acceptable_source_prefixes}"
            )
        _sources = glob(os.path.join(self.path.specfem_data, 
                                     f"{self.source_prefix}_*"))
        assert(_sources), (f"No source files with prefix {self.source_prefix} "
                           f"found in DATA/")
        assert(len(_sources) >= self.ntask), (
            "Number of requested `ntasks` is larger than the number of "
            "available source files"
            )

        # Check that model type is set correctly in the Par_file
        model_type = getpar(key="MODEL",
                            file=os.path.join(self.path.specfem_data,
                                              "Par_file"))[1]
        assert(model_type in self._available_model_types), (
            f"SPECFEM Par_file parameter `model`='{model_type}' does not "
            f"match acceptable model types: {self._available_model_types}"
            )

        # Make sure the initial model is set and actually contains files
        assert(self.path.model_init is not None and
               os.path.exists(self.path.model_init)), \
            f"`path_model_init` is required for the solver, but does not exist"

        assert(len(glob(os.path.join(self.path.model_init, "*")))), \
            f"`path_model_init` is empty but should have model files"

        if self.path.model_true is not None:
            assert(os.path.exists(self.path.model_true)), \
                f"`path_model_true` is provided but does not exist"
            assert(len(glob(os.path.join(self.path.model_true, "*")))), \
                f"`path_model_true` is empty but should have model files"

        # Check that the number of tasks/events matches the number of events
        self._source_names = check_source_names(
            path_specfem_data=self.path.specfem_data,
            source_prefix=self.source_prefix, ntask=self.ntask
        )

        assert(isinstance(self.update_density, bool)), \
            f"solver `density` must be True (variable) or False (constant)"

        # Check the size of the DATA/ directory and let the User know if 
        # large files are present, e.g., tomo xyz files or topo/bathy
        for root, dirs, files in os.walk(self.path.specfem_data):
            for name in files:
                fullpath = os.path.join(root, name)
                if not os.path.islink(fullpath):
                    filesize = os.path.getsize(fullpath) / 1E9  # Bytes -> GB
                    if filesize > 0.5:
                        logger.warning(
                            f"SPECFEM DATA/ file '{fullpath}' is >.5GB and "
                            f"will be copied {self.ntask} time(s). Please be "
                            f"sure to check if this file is necessary for your "
                            f"workflow"
                            )

    def setup(self):
        """
        Prepares solver scratch directories for an impending workflow.

        Sets up directory structure expected by SPECFEM and copies or generates
        seismic data to be inverted or migrated.

        Exports INIT/STARTING and TRUE/TARGET models to disk (output/ dir.)
        """
        # Create the internal directory structure required for storing results
        for pathname in ["_solver_output"]:
            unix.mkdir(self.path[pathname])

        # Assign file extensions to be used for database file searching
        model_type = getpar(key="MODEL",
                            file=os.path.join(self.path.specfem_data,
                                              "Par_file"))[1]
        if "gll" in model_type:
            self._ext = ".bin"
        else:
            logger.warning("no SPECFEM model type specified to define file "
                           "extension, defaulting to '.bin'")
            self._ext = ".bin"

        self._initialize_working_directories()
        self._export_starting_models()

    def check_model_values(self, path):
        """
        Convenience function to check parameter and model validity for
        chosen Solver model. Should be called by the Workflow module

        :type path: str
        :param path: path to model file(s) that should be in the format expected
            by the Model class (FORTRAN binary, ADIOS etc.)
        """
        assert os.path.exists(path), f"Model check path does not exist: {path}"

        _model = Model(path=path, parameters=self._parameters,
                       regions=self._regions)
        try:
            _model.check()
            _model.print_stats()
        except AssertionError as e:
            logger.critical(
                msg.cli(str(e), header="model read error", border="=")
            )
            sys.exit(-1)

################ PATCH MAX ####################

    def enforce_model_bounds(self, model):
        """
        Optionally clamp vp/vs values in a Model instance in-place.
        Works for 'vp'/'vs' and region-tagged names like 'reg1_vp'.
        """
        limit_v  = bool(getattr(self, "limit_vpvs", False))
        limit_nu = bool(getattr(self, "limit_poisson", False))
        if not (limit_v or limit_nu):
            return model


        import numpy as np
        from seisflows import logger

        def _clip_param(param_key, vmin, vmax):
            if vmin is None and vmax is None:
                return 0
            affected = 0
            for key in list(model.model.keys()):
                if key.split("_")[-1] != param_key:
                    continue
                arrs = model.model[key]
                for i in range(len(arrs)):
                    arr = arrs[i]
                    orig = arr.copy()
                    lo = -np.inf if vmin is None else float(vmin)
                    hi =  np.inf if vmax is None else float(vmax)
                    mask = (orig < lo) | (orig > hi)
                    np.clip(arr, lo, hi, out=arr)
                    affected += int(mask.sum())
            return affected

        if limit_v:
            n_vp = _clip_param("vp", self.vp_min, self.vp_max)
            n_vs = _clip_param("vs", self.vs_min, self.vs_max)
            
        # --- Poisson nach dem Basis-Clipping erzwingen ---
        if limit_nu:
            c_vp, c_vs = self.enforce_poisson_ratio(model)
            if limit_v:
                # harte vp/vs-Bounds final erneut anwenden
                n_vp += _clip_param("vp", self.vp_min, self.vp_max)
                n_vs += _clip_param("vs", self.vs_min, self.vs_max)
            if c_vp or c_vs:
                logger.info(f"poisson ratio enforced: changed {c_vp} vp and {c_vs} vs elements")

        if limit_v:
            if n_vp or n_vs:
                logger.info(
                    f"velocity clipping applied: "
                    f"{n_vp} vp values and {n_vs} vs values clipped to "
                    f"[{self.vp_min},{self.vp_max}] / [{self.vs_min},{self.vs_max}]"
                )
            else:
                logger.debug("velocity clipping enabled but no values out of bounds")



        return model

    def enforce_poisson_ratio(self, model):
        """
        Erzwingt nu_min <= nu <= nu_max, indem das Verhältnis R = vp/vs
        zwischen Rmin und Rmax begrenzt wird. Standard: passe vp an (vs fix).
        Fluide/Hohlraum (vs <= nu_skip_vs_below) werden übersprungen.
        """
        if not getattr(self, "limit_poisson", False):
            return 0, 0  # keine Änderungen
    
        import numpy as np
        from seisflows import logger
    
        # R(ν) = sqrt(2(1-ν)/(1-2ν))
        def R(nu):
            return np.sqrt(2.0*(1.0-nu) / np.maximum(1e-12, 1.0-2.0*nu))
    
        Rmin = R(self.nu_min)
        Rmax = R(self.nu_max)
        if not np.isfinite(Rmin) or not np.isfinite(Rmax) or Rmin <= 0 or Rmax <= 0 or Rmin > Rmax:
            logger.warning(f"invalid Poisson bounds: nu_min={self.nu_min}, nu_max={self.nu_max} -> skip")
            return 0, 0
    
        changed_vp = 0
        changed_vs = 0
    
        # finde passende vp/vs-Schlüssel (unterstützt reg?-vp/vs)
        keys_vp = [k for k in model.model.keys() if k.split("_")[-1] == "vp"]
        for k_vp in keys_vp:
            k_vs = k_vp[:-2] + "vs"  # '..._vp' -> '..._vs'
            if k_vs not in model.model:
                continue
    
            vp_list = model.model[k_vp]
            vs_list = model.model[k_vs]
            for i in range(len(vp_list)):
                vp = vp_list[i]
                vs = vs_list[i]
    
                # Solids: vs > Schwelle
                solid = vs > float(self.nu_skip_vs_below)
                if not np.any(solid):
                    continue
    
                if self.nu_strategy.lower() == "vp_from_vs":
                    lo = Rmin * vs
                    hi = Rmax * vs
                    before = vp.copy()
                    # nur dort clippen, wo solid
                    vp[solid] = np.minimum(np.maximum(vp[solid], lo[solid]), hi[solid])
                    changed_vp += int(np.count_nonzero(vp != before))
    
                elif self.nu_strategy.lower() == "vs_from_vp":
                    # Vs-Band aus vp ableiten
                    lo = vp / Rmax
                    hi = vp / Rmin
                    before = vs.copy()
                    vs[solid] = np.minimum(np.maximum(vs[solid], lo[solid]), hi[solid])
                    changed_vs += int(np.count_nonzero(vs != before))
    
                else:
                    # Fallback: wie vp_from_vs
                    lo = Rmin * vs
                    hi = Rmax * vs
                    before = vp.copy()
                    vp[solid] = np.minimum(np.maximum(vp[solid], lo[solid]), hi[solid])
                    changed_vp += int(np.count_nonzero(vp != before))
    
        return changed_vp, changed_vs


##############################################


    def set_parameters(self, keys, vals, file, delim, **kwargs):
        """
        Public API that allows other modules modify solver-specific files with
        paths relative to the `cwd` attribute.

        Primarily used to modify locations or force vector direction for
        the generation of different kernels in noise workflows.

        Only works if file exists, otherwise raises FileNotFoundError
        Kwargs are passed to `seisflows.tools.specfem.setpar()`

        :type key: str
        :param key: case-insensitive key to match in par_file. must match EXACT
        :type val: str
        :param val: value to OVERWRITE to the given key
        :raises FileNotFoundError: if `file` does not exist within the solver's
            working directory
        """
        os.chdir(self.cwd)
        if os.path.exists(file):
            for key, val in zip(keys, vals):
                setpar(key=key, val=val, file=file, delim=delim, **kwargs)
        else:
            raise FileNotFoundError(f"solver/{file} not found, cannot set "
                                    f"parameters")

    @property
    def source_names(self):
        """
        Returns list of source names which should be stored in PAR.SPECFEM_DATA
        Source names are expected to match the following wildcard,
        'PREFIX_*' where PREFIX is something like 'CMTSOLUTION' or 'FORCE'

        .. note::
            Dependent on environment variable 'SEISFLOWS_TASKID' which is
            assigned by system.run() to each individually running process.

        :rtype: list
        :return: list of source names
        """
        if self._source_names is None:
            self._source_names = check_source_names(
                path_specfem_data=self.path.specfem_data,
                source_prefix=self.source_prefix, ntask=self.ntask
            )
        return self._source_names

    @property
    def source_name(self):
        """
        Returns name of source currently under consideration

        .. note::
            Dependent on environment variable 'SEISFLOWS_TASKID' which is
            assigned by system.run() to each individually running process.

        :rtype: str
        :return: given source name for given task id
        """
        return self.source_names[get_task_id()]

    @property
    def cwd(self):
        """
        Returns working directory currently in use by a running solver instance

        .. note::
            Dependent on environment variable 'SEISFLOWS_TASKID' which is
            assigned by system.run() to each individually running process.

        :rtype: str
        :return: current solver working directory
        """
        return os.path.join(self.path.scratch, self.source_name)

    def data_wildcard(self, comp="?"):
        """
        Returns a wildcard identifier for synthetic data based on SPECFEM2D
        file naming schema. Allows formatting dcomponent e.g.,
        when called by solver.data_filenames.

        Some example SPECFEM2D ASCII seismogram file names for reference:
        - AA.S000000.BXY.semd: Membrane wave displacement
        - AA.S000000.BXY.semp: Membrane wave pressure
        - AA.S000000.PRE.semp: P-SV pressure seismogram

        .. note::

            SPECFEM3D/3D_GLOBE versions must overwrite this function

        :type comp: str
        :param comp: component formatter, defaults to wildcard '?'
        :rtype: str
        :return: wildcard identifier for channels
        """
        if self.syn_data_format.upper() == "SU":
            return f"U{comp}*.su"  # e.g., Up_file_single_p.su
        elif self.syn_data_format.upper() == "ASCII":
            return f"*.??{comp}.sem?"  # e.g., AA.S000000.BXY.semd

    def model_wildcard(self, par="*", kernel=False):
        """
        Returns a wildcard identifier to search for models kernels generated by
        the solver. An example SPECFEM2D/3D kernel filename (in 
        FORTRAN binary file format) is: 'proc000001_rho_kernel.bin'
        Whereas the corresponding model would be 'proc000001_rho.bin'

        Allows dynamically searching for specific files when renaming, moving
        or copying files. Also allows for different wildcard for 3D_GLOBE 
        version

        :type par: str
        :param par: parameter formatter, defaults to wildcard '?'
        :type kernel: bool
        :param kernel: wildcarding a kernel file. If True, adds the 'kernel' 
            tag. If not, assuming we are wildcarding for a model file
        :rtype: str
        :return: wildcard identifier for channels
        """
        if kernel:
            _ker = "_kernel"
        else:
            _ker = ""
        return f"proc??????_{par}{_ker}{self._ext}"

    def data_filenames(self, choice="obs"):
        """
        Returns the filenames of SPECFEM2D data, either by the requested
        components or by all available files in the directory.

         .. note::
            SPECFEM3D/3D_GLOBE versions must overwrite this function

        .. note::
            If the glob returns an  empty list, this function exits the
            workflow because filenames should not be empty is they're being
            queried

        :rtype: list
        :return: list of data filenames
        """
        assert(choice in ["obs", "syn", "adj"]), \
            f"choice must be: 'obs', 'syn' or 'adj'"

        if self.components:
            comp_glob = f"[{self.components}]"  # 'NEZ' -> '[NEZ]' for wildcard
        else:
            comp_glob = "?"
        data_wildcard = self.data_wildcard(comp=comp_glob)
        file_glob = os.path.join(self.cwd, "traces", choice, data_wildcard)
        filenames = glob(file_glob)

        if not filenames:
            logger.critical(
                msg.cli("The property `solver.data_filenames`, used to search "
                        "for waveform files, is empty and should not be. "
                        "Please check solver parameters: ",
                        items=[f"failed wildcard: {file_glob}"],
                        header="data filenames error", border="=")
            )
            sys.exit(-1)

        return filenames

    @property
    def model_databases(self):
        """
        The location of model inputs and outputs as defined by SPECFEM2D.
        This is RELATIVE to a SPECFEM2D working directory.

         .. note::
            This path is SPECFEM version dependent so SPECFEM3D/3D_GLOBE
            versions must overwrite this function

        :rtype: str
        :return: path where SPECFEM2D database files are stored, relative to
            `solver.cwd`
        """
        return "DATA"

    @property
    def model_files(self):
        """
        Return a list of paths to model files that match the internal parameter
        list. Used to generate model vectors of the same length as gradients.

        :rtype: list
        :return: a list of full paths to model files that matches the internal
            list of solver parameters
        """
        _model_files = []
        for par in self._parameters:
            _model_files += glob(os.path.join(self.path._mainsolver,
                                              self.model_databases,
                                              self.model_wildcard(par=par))
                                              )
        return _model_files

    @property
    def kernel_databases(self):
        """
        The location of kernel inputs and outputs as defined by SPECFEM2D
        This is RELATIVE to a SPECFEM2D working directory.

         .. note::
            This path is SPECFEM version dependent so SPECFEM3D/3D_GLOBE
            versions must overwrite this function

        :rtype: str
        :return: path where SPECFEM2D database files are stored, relative to
            `solver.cwd`
        """
        return "OUTPUT_FILES"

    def forward_simulation(self, save_traces=False,
                           export_traces=False, save_forward_arrays=False,
                           flag_save_forward=True, **kwargs):
        """
        Wrapper for SPECFEM binaries: 'xmeshfem?D' 'xgenerate_databases',
                                      'xspecfem?D'

        Calls SPECFEM2D forward solver, exports solver outputs to traces dir

         .. note::
            SPECFEM3D/3D_GLOBE versions must overwrite this function

        :type save_traces: str
        :param save_traces: move files from their native SPECFEM output location
            to another directory. This is used to move output waveforms to
            'traces/obs' or 'traces/syn' so that SeisFlows knows where to look
            for them, and so that SPECFEM doesn't overwrite existing files
            during subsequent forward simulations
        :type export_traces: str
        :param export_traces: export traces from the scratch directory to a more
            permanent storage location. i.e., copy files from their original
            location
        :type save_forward_arrays: str
        :param save_forward_arrays: relative path (relative to 
            /scratch/solver/<source_name>/<model_database>) to move the forward 
            arrays which are used for adjoint simulations. Mainly used for 
            ambient noise adjoint tomography which requires multiple forward 
            simulations prior to adjoint simulations, putting forward arrays 
            at the risk of overwrite. Normal Users can leave this default.
        :type flag_save_forward: bool
        :param flag_save_forward: whether to turn on the flag for saving the 
            forward arrays which are used for adjoint simulations. Not required 
            if only running forward simulations.
        """
        unix.cd(self.cwd)
        setpar(key="SIMULATION_TYPE", val="1", file="DATA/Par_file")
        setpar(key="SAVE_FORWARD", val=f".{str(flag_save_forward).lower()}.",
               file="DATA/Par_file")

        # Calling subprocess.run() for each of the binary executables listed
        for exc in self._fwd_simulation_executables:
            # e.g., fwd_mesher.log
            stdout = f"fwd_{self._exc2log(exc)}.log"
            self._run_binary(executable=exc, stdout=stdout)

        # Error check to ensure that mesher and solver have been run succesfully
        _solv = bool(glob(os.path.join("OUTPUT_FILES", self.data_wildcard())))
        if not _solv:
            logger.critical(msg.cli(f"solver failed to produce expected files",
                            header="external solver error", border="="))
            sys.exit(-1)

        # Work around SPECFEM's version dependent file names
        if self.syn_data_format.upper() == "SU":
            for tag in ["d", "v", "a", "p"]:
                unix.rename(old=f"single_{tag}.su", new="single.su",
                            names=glob(os.path.join("OUTPUT_FILES", "*.su")))
        # Exporting traces to disk (output/) for more permanent storage
        if export_traces:
            if not os.path.exists(export_traces):
                unix.mkdir(export_traces)
            unix.cp(
                src=glob(os.path.join("OUTPUT_FILES", self.data_wildcard())),
                dst=export_traces
            )
        # Save traces somewhere else in the scratch/ directory for easier access
        if save_traces:
            if not os.path.exists(save_traces):
                unix.mkdir(save_traces)
            unix.mv(
                src=glob(os.path.join("OUTPUT_FILES", self.data_wildcard())),
                dst=save_traces
            )
        # Save forward arrays to disk for later adjoint simulations. This is
        # primarily used for ambient noise adjoint tomography when other
        # forward simulations are required prior to the adjoint simulation,
        # which would overwrite existing forward arrays
        if save_forward_arrays:
            # NOTE: Relative path naming convention used, not absolute
            # scratch/solver/<source_name>/<save_forward_arrays>
            save_forward_arrays = os.path.join(self.cwd, save_forward_arrays)

            # Overwrites any existing forward arrays, for the case when we 
            # run a thrifty line search and run multiple fwd sims consecutively 
            unix.rm(save_forward_arrays)
            unix.mkdir(save_forward_arrays)

            for glob_key in [self._forward_array_wildcard, 
                             self._absorb_wildcard]:                                   
                unix.mv(src=glob(os.path.join(self.model_databases, glob_key)),
                        dst=save_forward_arrays)

        # Delete unncessary visualization files which may be large. This is 
        # only relevant for SPECFEM3D/3D_GLOBE, but will not throw errors for 2D
        if self.prune_scratch:
            logger.debug("prune scratch: removing '*.vt?' files from database")
            unix.rm(glob(os.path.join(self.model_databases, 
                                      "proc??????_*.vt?")))

        logger.info(f"FINISH FORWARD SIMULATION: {self.source_name}")

    def adjoint_simulation(self, save_kernels=False, export_kernels=False,
                           load_forward_arrays=False, 
                           del_loaded_forward_arrays=False, **kwargs):
        """
        Wrapper for SPECFEM binary 'xspecfem?D'

        Calls SPECFEM2D adjoint solver, creates the `SEM` folder with adjoint
        traces which is required by the adjoint solver. Renames kernels
        after they have been created from 'alpha' and 'beta' to 'vp' and 'vs',
        respectively.

         .. note::
            SPECFEM3D/3D_GLOBE versions must overwrite this function

        :type save_kernels: str
        :param save_kernels: move the kernels from their native SPECFEM output
            location to another path. This is used to move kernels to another
            SeisFlows scratch directory so that they are discoverable by
            other modules. The typical location they are moved to is
            path_eval_grad
        :type export_kernels: str
        :param export_kernels: export/copy/save kernels from the scratch
            directory to a more permanent storage location. i.e., copy files
            from their original location. Note that kernel file sizes are LARGE,
            so exporting kernels can lead to massive storage requirements.
        :type load_forward_arrays: str
        :param load_forward_arrays: relative path (relative to solver.cwd) to 
            load previously generated forward arrays which are used for adjoint 
            simulations. Mainly used for ambient noise adjoint tomography. Will 
            OVERWRITE any forward array files already located in the database 
            directory.
        :type del_loaded_forward_arrays: bool
        :param del_loaded_forward_arrays: only used if `load_forward_arrays` is
            set. After adjoint simulation completes nominally, delete the 
            forward arrays that were used to run the adjoint simulation to 
            save space. Usually
        """
        unix.cd(self.cwd)

        setpar(key="SIMULATION_TYPE", val="3", file="DATA/Par_file")
        setpar(key="SAVE_FORWARD", val=".false.", file="DATA/Par_file")

        unix.rm("SEM")
        unix.ln("traces/adj", "SEM")

        # Pre-load forward arrays if necessary
        if load_forward_arrays:
            logger.info(f"loading forward arrays: '{load_forward_arrays}'")
            
            # scratch/solver/<source_name>/<load_forward_arrays>
            load_forward_arrays = os.path.join(self.cwd, load_forward_arrays)

            # Few sanity checks to make sure something is actually loaded
            if not os.path.exists(load_forward_arrays):
                logger.critical(f"forward arrays not found: "
                                f"{load_forward_arrays}")
                sys.exit(-1)
            if not glob(os.path.join(load_forward_arrays, "*")):
                logger.critical(f"forward array's empty {load_forward_arrays}")
                sys.exit(-1)
            
            # 'cp' command will OVERWRITE existing forward arrays in the dir.
            for fwd_arr in glob(os.path.join(load_forward_arrays, "*")):
                fid = os.path.basename(fwd_arr)
                unix.cp(src=fwd_arr, dst=os.path.join(self.cwd, 
                                                      self.model_databases, 
                                                      fid)
                        )

        # Calling subprocess.run() for each of the binary executables listed
        for exc in self._adj_simulation_executables:
            # e.g., adj_solver.log
            stdout = f"adj_{self._exc2log(exc)}.log"
            logger.info(f"running SPECFEM executable {exc}, log to '{stdout}'")
            self._run_binary(executable=exc, stdout=stdout)

        # Rename 'alpha' -> 'vp' and 'beta' -> 'vs' for consistency. 
        # Wait a few seconds before doing this to avoid race condition of
        # kernel file creation and renaming
        self._rename_kernel_parameters()

        # Kernel export and saving must take place within the kernel directory
        unix.cd(os.path.join(self.cwd, self.kernel_databases))

        # Export kernels: copy them to some external directory for storage
        if export_kernels:
            unix.mkdir(export_kernels)
            for par in self._parameters:
                kernel_files = glob(self.model_wildcard(par=par, kernel=True))
                if kernel_files:
                    logger.debug(f"copying '{par}' kernels to {export_kernels}")
                    unix.cp(src=kernel_files, dst=export_kernels)
                else:
                    logger.warning(f"no kernel files for '{par}', cant export")

        # Save kernels: move kernels to an internal directory for later steps
        # so they don't get overwritten by future adjoint simulations
        if save_kernels:
            unix.mkdir(save_kernels)
            for par in self._parameters:
                kernel_files = glob(self.model_wildcard(par=par, kernel=True))
                if kernel_files:
                    logger.debug(f"moving '{par}' kernels to {save_kernels}")
                    unix.mv(src=kernel_files, dst=save_kernels)
                else:
                    logger.critical(f"no kernel files found for '{par}', "
                                    f"please check adjoint solver log for "
                                    f"{self.source_name}")
                    sys.exit(-1)

        # Working around fact that `absorb_buffer` files have diff naming w.r.t
        # SPECFEM3D. Will also remove `save_forward_arrays` to free up space
        # since we no longer need these
        if self.prune_scratch:                                                   
            for glob_key in [self._forward_array_wildcard, 
                             self._absorb_wildcard]:
                logger.debug(f"prune scratch: removing '{glob_key}' files"
                             f"from database ")                                  
                unix.rm(glob(os.path.join(self.model_databases, glob_key)))

        if load_forward_arrays and del_loaded_forward_arrays:
            logger.debug(f"removing loaded forward arrays: "
                         f"{load_forward_arrays}")
            unix.rm(load_forward_arrays)

        logger.info(f"FINISH ADJOINT SIMULATION: {self.source_name}")

    def _rename_kernel_parameters(self):
        """
        Rename kernels to work w/ conflicting name conventions.
        - alpha -> vp
        - beta -> vs
        
        Performed directly inside the directory so that rename won't affect
        any strings in the full path. Deals with both SPECFEM3D and 3D_GLOBE.
        GLOBE version adds in the 'reg?' tag that needs to be considered.

        Kept as a separate function so it can be called outside the adjoint
        simulation task for debugging purposes.
        """
        unix.cd(os.path.join(self.cwd, self.kernel_databases))

        for tag in ["alpha", "alpha[hv]", "reg?_alpha", "reg?_alpha[hv]"]:
            names = glob(self.model_wildcard(par=tag, kernel=True))
            if names:
                logger.info(f"renaming {len(names)} kernels: '{tag}' -> 'vp'")
                unix.rename(old="alpha", new="vp", names=names)

        for tag in ["beta", "beta[hv]", "reg?_beta", "reg?_beta[hv]"]:
            names = glob(self.model_wildcard(par=tag, kernel=True))
            if names:
                logger.info(f"renaming {len(names)} kernels: '{tag}' -> 'vs'")
                unix.rename(old="beta", new="vs", names=names)
        # --- acoustic filename conventions -> SeisFlows conventions ---
        ############PATCH MAX############# AKUSTISCHE SIMULATIONEN
        # Vp (Schallgeschwindigkeit)
        names = glob(self.model_wildcard(par="c_acoustic", kernel=True))
        if names:
            logger.info(f"renaming {len(names)} kernels: 'c_acoustic' -> 'vp'")
            unix.rename(old="c_acoustic", new="vp", names=names)
        
        # Bulk modulus (kappa)
        names = glob(self.model_wildcard(par="kappa_acoustic", kernel=True))
        if names:
            logger.info(f"renaming {len(names)} kernels: 'kappa_acoustic' -> 'kappa'")
            unix.rename(old="kappa_acoustic", new="kappa", names=names)
        
        # Dichte (rho)
        names = glob(self.model_wildcard(par="rho_acoustic", kernel=True))
        if names:
            logger.info(f"renaming {len(names)} kernels: 'rho_acoustic' -> 'rho'")
            unix.rename(old="rho_acoustic", new="rho", names=names)
        
        # (optional) rhop -> rhop (falls du das je nutzen willst)
        names = glob(self.model_wildcard(par="rhop_acoustic", kernel=True))
        if names:
            logger.info(f"renaming {len(names)} kernels: 'rhop_acoustic' -> 'rhop'")
            unix.rename(old="rhop_acoustic", new="rhop", names=names)
        ############################################

   
    def combine(self, input_paths, output_path, parameters=None):
        
        
        # --- Guard: Kernel pro-Proc auf Modellgröße trimmen ---
        model_dir = Path(self.path._mainsolver) / "specfem2d_workdir" / "OUTPUT_FILES_INIT"
    
        kdir = Path(self.path.eval_grad) / "misfit_kernel"   # <— self.path, nicht self.paths
        klist = sorted(kdir.glob("proc*_vp_kernel.bin")) + sorted(kdir.glob("proc*_vs_kernel.bin"))
        for kfile in klist:
            _crop_kernel_to_model_size(str(kfile), model_dir=model_dir, dtype="float32",
                                       backup=True, logger=logger)
        # ------------------------------------------------------
    
        # Im mainsolver ausführen (falls vorhanden), sonst self.cwd
        workdir = self.path._mainsolver if os.path.isdir(self.path._mainsolver) else self.cwd
        logger.debug(f"[combine] using workdir={workdir}")
        unix.cd(workdir)
        logger.debug(f"[combine] cwd={os.getcwd()}")
    
        if parameters is None:
            parameters = self._parameters
            
         # Stelle sicher, dass alle Namen auf *_kernel enden
        parameters = [
            p if p.endswith("_kernel") else f"{p}_kernel"
            for p in parameters
        ]   
        if not os.path.exists(output_path):
            unix.mkdir(output_path)
    
        # Pfade der Events an xcombine_sem übergeben
        with open("kernel_paths", "w") as f:
            for p in input_paths:
                f.write(f"{p}\n")
    
        exe = os.path.join(self.path.specfem_bin, "xcombine_sem")
        if not (os.path.isfile(exe) and os.access(exe, os.X_OK)):
            logger.critical(f"xcombine_sem not found or not executable: {exe}")
            sys.exit(1)
    
        import shutil
        n_tasks = int(os.environ.get("SLURM_NTASKS", "1"))
        if shutil.which("mpirun"):      launcher = f"mpirun -n {n_tasks}"    # mpirun bevorzugt
        elif shutil.which("mpiexec"):   launcher = f"mpiexec -n {n_tasks}"
        elif shutil.which("srun"):      launcher = f"srun -n {n_tasks}"
        else:
            logger.critical("No MPI launcher found (srun/mpirun/mpiexec).")
            sys.exit(1)
        logger.debug(f"[combine] using MPI launcher: {launcher}")
    
        for name in parameters:  # erwartet z.B. 'vp_kernel', 'vs_kernel'
            cmd = f"{launcher} {exe} {name} kernel_paths {output_path}"
            stdout = f"{self._exc2log(exe)}_{name}.log"     # <— auf Binary loggen
            self._run_binary(executable=cmd, stdout=stdout, with_mpi=False)
    
        # Danach: Masking auf dem kombinierten Kernel
        self._debug_sizes_summary(output_path, parameters=parameters)
        try:
            if getattr(self, "mask_sr", False) or getattr(self, "mask_top_layer", False):
                logger.info(
                    f"applying source/receiver mask to '{output_path}' "
                    f"(src_r={self.mask_src_radius_m} m, "
                    f"rec_r={self.mask_rec_radius_m} m, "
                    f"taper={self.mask_taper_m} m)"
                )
                self._mask_gradient_near_sr(input_path=output_path, parameters=parameters)
        except Exception as e:
            logger.warning(f"source/receiver kernel mask failed: {e}")

    
        # --- Größenbilanz für Debugging (hilft bei shape-Mismatches) --------------
        self._debug_sizes_summary(output_path, parameters=parameters)


    ############# --- PATCH MAX FUER DIE NÄCHSTEN FUNKTIONEN
    ############# GRADIENTEN ZU NULL SETZEN AN SENDERN UND EMPFAENGERN
    
    def _debug_sizes_summary(self, kernel_dir, parameters=None):
        """Schreibt eine kompakte Größenbilanz ins Log:
           - Kernel-Dateigrößen (Summe -> erwartete Gradient-Länge)
           - Rohgrößen der Modelldateien an verschiedenen Orten
           - Vektorlängen laut Model-Klasse (ob Ghosts verworfen werden)
        """
        if parameters is None:
            parameters = self._parameters
    
        logger.info("[mask][debug] ===== SIZE SUMMARY (start) =====")
        # --- Kernel-Dateien
        total_kernel = 0
        for par in parameters:
            # Stelle sicher, dass wir wirklich nach *_kernel suchen
            par_tag = par if par.endswith("_kernel") else f"{par}_kernel"
            matched = []
            for pat in self._kernel_file_patterns(kernel_dir, par_tag):
                files = glob(pat)
                logger.info(f"[debug] kernel pattern '{pat}' -> {len(files)} files")
                matched.extend(files)
            for f in sorted(set(matched)):
                n = os.path.getsize(f) // 4  # float32
                total_kernel += n
                logger.info(f"[debug] kernel file {os.path.basename(f)} : {n} float32")
        logger.info(f"[debug] total kernel elements (sum over files) = {total_kernel}")
    
        # --- Rohgrößen der Modelldateien an typischen Orten
        model_places = [
            ("mainsolver/DATA", os.path.join(self.path._mainsolver, self.model_databases)),
            ("cwd/DATA",        os.path.join(self.cwd, self.model_databases)),
            ("output/MODEL_INIT", os.path.join(self.path.output, "MODEL_INIT")),
            ("path_model_init",   self.path.model_init or ""),
        ]
        for name, base in model_places:
            if base and os.path.exists(base):
                tot = 0
                for par in parameters:
                    par_tag = par.replace("_kernel", "")
                    for f in glob(os.path.join(base, f"proc??????_{par_tag}{self._ext}")):
                        tot += os.path.getsize(f) // 4
                logger.info(f"[debug] model raw length sum [{name}] = {tot} float32")
            else:
                logger.info(f"[debug] model path missing [{name}] -> {base}")
    
        # --- Vektorlängen, wie sie die Model-Klasse sieht (entscheidend für scaling)
        try:
            from seisflows.tools.model import Model
            m_ms = Model(path=os.path.join(self.path._mainsolver, self.model_databases),
                         parameters=[p.replace("_kernel","") for p in parameters],
                         regions=self._regions)
            logger.info(f"[debug] Model.vector length (mainsolver/DATA) = {m_ms.vector.size}")
        except Exception as e:
            logger.warning(f"[debug] Model(...) mainsolver failed: {e}")
        try:
            if self.path.model_init and os.path.exists(self.path.model_init):
                m_init = Model(path=self.path.model_init,
                               parameters=[p.replace("_kernel","") for p in parameters],
                               regions=self._regions)
                logger.info(f"[debug] Model.vector length (path_model_init) = {m_init.vector.size}")
        except Exception as e:
            logger.warning(f"[debug] Model(...) path_model_init failed: {e}")
    
        logger.info("[mask][debug] ===== SIZE SUMMARY (end) =====")
    
    def _kernel_file_patterns(self, dirpath, par):
        """
        Liefert eine Liste möglicher Pattern für Kernel-Dateien,
        egal ob 'vs' oder 'vs_kernel' übergeben wurde.
        """
        base = par
        has_kernel = par.endswith("_kernel")
        patterns = []
        if has_kernel:
            # z.B. 'vs_kernel' -> genau so suchen
            patterns.append(os.path.join(dirpath, f"proc??????_{par}{self._ext}"))
            # einige Workflows lassen das zweite '_kernel' trotzdem drin:
            patterns.append(os.path.join(dirpath, f"proc??????_{par}_kernel{self._ext}"))
        else:
            # z.B. 'vs' -> Standard 'vs_kernel'
            patterns.append(os.path.join(dirpath, f"proc??????_{par}_kernel{self._ext}"))
            # Fallback falls ohne Suffix geschrieben wurde
            patterns.append(os.path.join(dirpath, f"proc??????_{par}{self._ext}"))
        return patterns


    def _radial_cosine_taper(self, x, z, xc, zc, r0, rt):
        """
        1D-Vektorversion: gibt Gewichte in [0,1] zurück.
        r0: Voll-Mute-Radius, rt: Taperbreite (additiv).
        """
        if r0 <= 0.0:
            return np.ones_like(x, dtype=np.float32)
        r = np.sqrt((x - xc)**2 + (z - zc)**2)
        w = np.ones_like(r, dtype=np.float32)
        # Voll-Mute innerhalb r0
        w[r <= r0] = 0.0
        if rt > 0.0:
            # Kosinus-Taper in [r0, r0+rt]
            m = (r > r0) & (r < (r0 + rt))
            w[m] = 0.5 * (1.0 - np.cos(np.pi * (r[m] - r0) / rt))
            # außerhalb r0+rt bleibt 1.0
        return w.astype(np.float32)

    def _vertical_top_taper(self, z, z_top, thick, taper):
        """
        Kosinus-Taper von unten nach oben auf die oberste 'thick' Schicht.
        Voll-Mute in [z0, z_top], weicher Übergang in [z0 - taper, z0].
        """
        if thick <= 0.0:
            return np.ones_like(z, dtype=np.float32)
    
        z0 = z_top - float(thick)  # Unterkante der voll gemuteten Schicht
        w = np.ones_like(z, dtype=np.float32)
    
        # Voll-Mute in der Top-Schicht
        w[z >= z0] = 0.0
    
        # Optionaler Taper darunter
        taper = float(taper)
        if taper > 0.0:
            m = (z >= (z0 - taper)) & (z < z0)
            # 1 -> 0 über die Taperstrecke
            w[m] = 0.5 * (1.0 + np.cos(np.pi * (z[m] - (z0 - taper)) / taper))
    
        return w.astype(np.float32)

    def _horizontal_side_taper(self, x, x_left, x_right, thick, taper):
        """
        Kosinus-Taper an linkem und rechtem Modellrand (analog zu _vertical_top_taper).
        Voll-Mute in [x_left, x_left+thick] und [x_right-thick, x_right],
        weicher Uebergang jeweils ueber 'taper' nach innen.
        """
        if thick <= 0.0:
            return np.ones_like(x, dtype=np.float32)

        w = np.ones_like(x, dtype=np.float32)
        taper = float(taper)

        # linker Rand: Voll-Mute in [x_left, x0], x0 = x_left + thick
        x0 = x_left + float(thick)
        w[x <= x0] = 0.0
        if taper > 0.0:
            m = (x > x0) & (x <= (x0 + taper))
            w[m] = 0.5 * (1.0 - np.cos(np.pi * (x[m] - x0) / taper))

        # rechter Rand: Voll-Mute in [x1, x_right], x1 = x_right - thick
        x1 = x_right - float(thick)
        w[x >= x1] = 0.0
        if taper > 0.0:
            m = (x < x1) & (x >= (x1 - taper))
            w[m] = 0.5 * (1.0 - np.cos(np.pi * (x1 - x[m]) / taper))

        return w.astype(np.float32)

    def _read_source_coords(self):
        """xs,zs aus dem aktuell verlinkten DATA/SOURCE lesen."""
        srcfile = os.path.join(self.cwd, "DATA", self.source_prefix)
        # getpar ist im Modul bereits verfügbar
        xs = float(getpar(key="xs", file=srcfile)[1])
        zs = float(getpar(key="zs", file=srcfile)[1])
        return xs, zs
    
    def _read_station_coords(self):
        """Stationsdatei im mainsolver lesen -> Liste [(x,z), ...]."""
        stas = []
        stfile = os.path.join(self.cwd, "DATA", "STATIONS")
        if os.path.exists(stfile):
            with open(stfile, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    # SPECFEM2D: NET STAX X Z (optional weitere Spalten)
                    if len(parts) >= 4:
                        try:
                            x = float(parts[2]); z = float(parts[3])
                            stas.append((x, z))
                        except ValueError:
                            continue
        return stas
    
    def _find_grid_bins(self, proc):
        """
        Suche x/z-Binärgrids für einen MPI-Proc.
        Bevorzugt output/MODEL_INIT, fällt dann auf klassische Solver-Pfade zurück.
        """
        base_dirs = [
            os.path.join(self.path.output, "MODEL_INIT"),
            os.path.join(self.cwd, "output", "MODEL_INIT"),
            os.path.join(self.cwd, "OUTPUT_FILES", "DATABASES_MPI"),
            os.path.join(self.cwd, "DATA", "DATABASES_MPI"),
            os.path.join(self.cwd, "DATA"),
        ]
        candidates = []
        for d in base_dirs:
            candidates += [
                (d, f"{proc}_x.bin", f"{proc}_z.bin"),            # proc000000_x.bin
                (d, "x.bin", "z.bin"),                             # x.bin / z.bin
                (os.path.join(d, proc), "x.bin", "z.bin"),         # proc000000/x.bin
            ]
        for d, xname, zname in candidates:
            xbin = os.path.join(d, xname)
            zbin = os.path.join(d, zname)
            if os.path.exists(xbin) and os.path.exists(zbin):
                logger.info(f"[mask] grid for {proc}: {xbin} | {zbin}")
                return xbin, zbin
        logger.warning(f"[mask] no grid dumps found for {proc} in {base_dirs}")
        return None, None


    def _mask_gradient_near_sr(self, input_path, parameters=None):
        """
        Maskiert kombinierte Kernel in `input_path` pro Partition (proc).
        Vor dem Maskieren wird jede Kerneldatei auf die procspezifische Modell-Länge
        (OUTPUT_FILES_INIT) gecroppt, damit SPECFEM-konforme Längen garantiert sind.
        Nutzt für die Top-Layer-Maske einen GLOBALEN z_top (aus output/MODEL_INIT),
        damit innere Partitionen NICHT versehentlich als "Oberfläche" gemutet werden.
        """
        logger.info(f"[mask] ENTER _mask_gradient_near_sr(input_path={input_path}, ext='{self._ext}', cwd={self.cwd}) [IMPLEMENTATION_ID=topfix+precrop+nopad]")
    
        if not (getattr(self, "mask_sr", False) or getattr(self, "mask_top_layer", False)
                or getattr(self, "mask_side_layer", False)
                or float(getattr(self, "mask_rec_radius_m", 0)) > 0):
            logger.info("[mask] no masking requested -> early return")
            return
    
        unix.cd(self.cwd)
    
        # Parameter sicher auf *_kernel normalisieren
        if parameters is None:
            parameters = getattr(self, "_parameters", None) or []
        parameters = [p if p.endswith("_kernel") else f"{p}_kernel" for p in parameters]
    
        # Quelle/Stationen lesen (best effort)
        try:
            xs, zs = self._read_source_coords()
            logger.info(f"[mask] SOURCE coords: xs={xs:.6f}, zs={zs:.6f}")
        except Exception as e:
            logger.warning(f"[mask] could not read SOURCE coords: {e}")
            xs, zs = None, None
    
        stations = self._read_station_coords()
        logger.info(f"[mask] STATIONS: n={len(stations)} (first 3: {stations[:3] if stations else []})")
        
        model_init_dir = os.path.join(self.path.output, "MODEL_INIT")
        z_top_global = _find_global_surface_z(model_init_dir, zs_hint=zs, logger=logger)
        if z_top_global is not None:
            logger.info(f"[mask] GLOBAL z_top={z_top_global:.6f} (from {model_init_dir})")
        else:
            logger.warning("[mask] GLOBAL z_top could not be determined; top-layer mask will be skipped.")

        x_left_global, x_right_global = _find_global_x_bounds(model_init_dir, logger=logger)
        if x_left_global is not None:
            logger.info(f"[mask] GLOBAL x bounds: left={x_left_global:.6f}, right={x_right_global:.6f} "
                        f"(from {model_init_dir})")
        else:
            logger.warning("[mask] GLOBAL x bounds could not be determined; side-layer mask will be skipped.")
    
        # Modell-Länge pro Proc/Feld ermitteln (bevorzugt OUTPUT_FILES_INIT)
        def _target_size_for_proc(proc, par_model):
            candidates = [
                os.path.join(self.path._mainsolver, "specfem2d_workdir", "OUTPUT_FILES_INIT",
                             f"{proc}_{par_model}{self._ext}"),
                os.path.join(self.cwd, "specfem2d_workdir", "OUTPUT_FILES_INIT",
                             f"{proc}_{par_model}{self._ext}"),
                os.path.join(self.path.output, "MODEL_INIT",
                             f"{proc}_{par_model}{self._ext}"),
            ]
            sizes = []
            for pth in candidates:
                if pth and os.path.exists(pth):
                    try:
                        sizes.append(os.path.getsize(pth) // 4)  # float32
                    except OSError:
                        pass
            return min(sizes) if sizes else None
    
        total_files = masked_files = 0
        missing_grids = []
    
        # Fester Pfad zu OUTPUT_FILES_INIT fürs Precrop
        try:
            model_dir_precrop = Path(self.path._mainsolver) / "specfem2d_workdir" / "OUTPUT_FILES_INIT"
        except Exception:
            model_dir_precrop = Path(self.cwd) / "specfem2d_workdir" / "OUTPUT_FILES_INIT"
    
        for par in parameters:
            # alle Kernel-Dateien zu diesem Parameter einsammeln
            kfiles = []
            for pat in self._kernel_file_patterns(input_path, par):
                kfiles.extend(sorted(glob(pat)))
            if not kfiles:
                logger.warning(f"[mask] par={par}: no kernel files matched (ext='{self._ext}') -> skip")
                continue
    
            par_model = par.replace("_kernel", "")
    
            for kfile in kfiles:
                total_files += 1
                basename = os.path.basename(kfile)
                proc = basename.split("_")[0]  # 'proc000123'
                logger.info(f"[mask] par={par} {proc}: kfile={kfile}")
    
                # --- vorab Dateilänge sicherstellen (gegen Model croppen) ---
                try:
                    _crop_kernel_to_model_size(
                        kfile,
                        model_dir=model_dir_precrop,
                        dtype="float32",
                        backup=False,
                        logger=logger,
                    )
                except Exception as e:
                    logger.warning(f"[mask] par={par} {proc}: precrop failed -> {e}")
    
                # Grids suchen
                xbin, zbin = self._find_grid_bins(proc)
                logger.debug(f"[mask] grids for {proc}: xbin={xbin} zbin={zbin}")
                if not (xbin and zbin):
                    missing_grids.append(proc)
                    continue
    
                # x/z lesen (Fortran-Binary-aware: entfernt Record-Marker korrekt)
                try:
                    x = read_fortran_binary(xbin)
                    z = read_fortran_binary(zbin)
                except Exception as e:
                    logger.warning(f"[mask] par={par} {proc}: failed reading x/z -> {e}")
                    continue
                if x.size != z.size:
                    logger.warning(f"[mask] par={par} {proc}: x/z size mismatch {x.size} vs {z.size} -> skip")
                    continue

                # Kernel lesen (Fortran-Binary-aware, nach Precrop)
                try:
                    k = read_fortran_binary(kfile)
                except Exception as e:
                    logger.warning(f"[mask] par={par} {proc}: failed reading kernel -> {e}")
                    continue
    
                # procspezifische Zielgröße
                target_size = _target_size_for_proc(proc, par_model)
                if target_size is None:
                    target_size = k.size
                    logger.warning(f"[mask] {par} {proc}: no model size found -> using kernel size={target_size}")
    
                # Längen angleichen: immer nur croppen (kein Padding!)
                n = min(x.size, z.size, k.size, target_size)
                if (x.size, z.size, k.size) != (n, n, n) or n != target_size:
                    logger.info(f"[mask] {par} {proc}: align sizes x/z/k/target -> {x.size}/{z.size}/{k.size}/{target_size} -> {n}")
                x = x[:n]; z = z[:n]; k = k[:n]  # nur kürzen
    
                # Maske bauen
                w = np.ones_like(k, dtype=np.float32)
    
                # Quellen-Maske
                if xs is not None and float(self.mask_src_radius_m) > 0.0:
                    w *= self._radial_cosine_taper(
                        x, z, xs, zs,
                        r0=float(self.mask_src_radius_m),
                        rt=float(self.mask_taper_m),
                    )
    
                # Empfänger-Masken
                if float(self.mask_rec_radius_m) > 0.0 and stations:
                    for (xr, zr) in stations:
                        w *= self._radial_cosine_taper(
                            x, z, xr, zr,
                            r0=float(self.mask_rec_radius_m),
                            rt=float(self.mask_taper_m),
                        )
    
                # Top-Layer-Maske mit GLOBALER Oberfläche
                if getattr(self, "mask_top_layer", False) and float(self.mask_top_thickness_m) > 0.0 and (z_top_global is not None):
                    w *= self._vertical_top_taper(
                        z=z,
                        z_top=float(z_top_global),
                        thick=float(self.mask_top_thickness_m),
                        taper=float(self.mask_top_taper_m),
                    )

                # Side-Layer-Maske (links/rechts) mit GLOBALEN Modellraendern
                if getattr(self, "mask_side_layer", False) and float(self.mask_side_thickness_m) > 0.0 and (x_left_global is not None):
                    w *= self._horizontal_side_taper(
                        x=x,
                        x_left=float(x_left_global),
                        x_right=float(x_right_global),
                        thick=float(self.mask_side_thickness_m),
                        taper=float(self.mask_side_taper_m),
                    )

                # Statistik & Anwendung
                frac_taper = float((w < 1.0).sum()) / float(w.size) if w.size else 0.0
                logger.info(f"[mask] {par} {proc}: weight stats min={w.min():.3f} mean={w.mean():.3f} "
                            f"max={w.max():.3f} muted={frac_taper*100:.1f}%")
    
                before = float(np.linalg.norm(k)) if k.size else 0.0
                k *= w
                after  = float(np.linalg.norm(k)) if k.size else 0.0
    
                try:
                    write_fortran_binary(k.astype(np.float32), kfile)
                    masked_files += 1
                    logger.info(f"[mask] {par} {proc}: |k|2 {before:.3e} -> {after:.3e} | wrote {k.size} floats "
                                f"(Fortran-Binary mit Record-Markern)")
                except Exception as e:
                    logger.warning(f"[mask] {par} {proc}: write failed -> {e}")
                    continue
    
        # Nachlauf / Fehlerberichte
        if missing_grids:
            procs = ", ".join(sorted(set(missing_grids)))
            raise RuntimeError(f"[mask] Missing x/z grid bins for procs: {procs}")
    
        logger.info(f"[mask] EXIT _mask_gradient_near_sr: total={total_files}, masked={masked_files}")
        if masked_files == 0:
            logger.warning("[mask] completed but masked_files == 0 (no files written)")

##################################################################


    def smooth(self, input_path, output_path, parameters=None, span_h=None,
               span_v=None, use_gpu=None):
        """
        Wrapper for SPECFEM smoothing binaries: 
        xsmooth_sem, xsmooth_sem_pde, xsmooth_laplacian_sem
        User chooses which underlying function they want with the `smooth_type` 
        parameter

        .. note::
            It is ASSUMED that this function is being called by
            system.run(single=True) so that we can use the main solver
            directory to perform the kernel smooth task

        :type input_path: str
        :param input_path: path to data
        :type output_path: str
        :param output_path: path to export the outputs of xcombine_sem
        :type parameters: list
        :param parameters: optional list of parameters,
            defaults to `self._parameters`
        :type span_h: float
        :param span_h: horizontal smoothing length in meters
        :type span_v: float
        :param span_v: vertical smoothing length in meters
        :type use_gpu: bool
        :param use_gpu: whether to use GPU acceleration for smoothing. Requires
            GPU compiled binaries and GPU compute node.
        """ 
        # PATCH MAX--- begin: guard to ensure per-proc sizes match before processing ---
        from pathlib import Path
        
        # Guard: pro-Proc Kernel auf Modellgröße trimmen
        model_dir = Path(self.cwd) / "specfem2d_workdir" / "OUTPUT_FILES_INIT"
        pars = parameters or self._parameters
        
        for par in pars:
            par_tag = par if par.endswith("_kernel") else f"{par}_kernel"
            for kfile in sorted(Path(input_path).glob(f"proc??????_{par_tag}{self._ext}")):
                _crop_kernel_to_model_size(
                    str(kfile),
                    model_dir=model_dir,
                    dtype="float32",
                    backup=True,
                    logger=logger
                )
        # 



        unix.cd(self.cwd)

        # Assign some default parameters from class attributes if not given
        if parameters is None:
            parameters = self._parameters
        if span_h is None:
            span_h = self.smooth_h
        if span_v is None:
            span_v = self.smooth_v

        logger.debug(f"{self.smooth_type} smoothing {parameters} horizontal "
                     f"span {span_h}m and vertical span {span_v}m")

        if not os.path.exists(output_path):
            unix.mkdir(output_path)

        # Ensure trailing '/' character, required by xsmooth_sem
        input_path = os.path.join(input_path, "")
        output_path = os.path.join(output_path, "")
        if use_gpu is None:
            use_gpu = self.smooth_use_gpu
        if use_gpu:
            use_gpu = ".true"
        else:
            use_gpu = ".false"
            
        # Determine which smoothing function we are using
        if self.smooth_type.lower() == "gaussian":  # 2D/3D/3D_GLOBE
            cmd = "bin/xsmooth_sem"
        elif self.smooth_type.lower() == "laplacian":  # 3D_GLOBE ONLY
            cmd = "bin/xsmooth_laplacian_sem"
            # laplacian smoothing expects span values in km, not m
            span_h *= 1E-3  # m -> km
            span_v *= 1E-3  # m -> km
        elif self.smooth_type == "pde":  # 3D ONLY
            cmd = "bin/xsmooth_sem_pde"

        # mpiexec ./bin/xsmooth_sem SMOOTH_H SMOOTH_V name input output use_gpu
        for name in parameters:
            exc = (f"{cmd} {str(span_h)} {str(span_v)} {name}_kernel "
                   f"{input_path} {output_path} {use_gpu}")
            # e.g., combine_vs.log
            stdout = f"{self._exc2log(exc)}_{name}.log"
            self._run_binary(executable=exc, stdout=stdout, with_mpi=True)

        # Rename output files to remove the '_smooth' suffix which SeisFlows
        # will not recognize
        files = glob(os.path.join(output_path, "*"))
        unix.rename(old="_smooth", new="", names=files)

        ####PATCH MAX: lokal begrenztes Smoothing um Void-Waende####
        buffer_m = getattr(self, "local_void_smooth_buffer_m", 0.0)
        if buffer_m and buffer_m > 0:
            _blend_local_void_smoothing(
                input_path=input_path, output_path=output_path,
                parameters=parameters, ext=self._ext,
                buffer_m=buffer_m, logger=logger
            )
        #### PATCH MAX ####

    def _run_binary(self, executable, stdout="solver.log", with_mpi=True):
        """
        Calls MPI solver executable to run solver binaries, used by individual
        processes to run the solver on system. If the external solver returns a
        non-zero exit code (failure), this function will return a negative
        boolean.

        .. note::
            This function ASSUMES it is being run from a SPECFEM working
            directory, i.e., that the executables are located in ./bin/

        .. note::
            This is essentially an error-catching wrapper of subprocess.run()

        :type executable: str
        :param executable: executable function to call. May or may not start
            E.g., acceptable calls for the solver would './bin/xspecfem2D'.
            Also accepts additional command line arguments such as:
            'xcombine_sem alpha_kernel kernel_paths...'
        :type stdout: str
        :param stdout: where to redirect stdout
        :type with_mpi: bool
        :param with_mpi: If `mpiexec` is given, use MPI to run the executable.
            Some executables (e.g., combine_vol_data_vtk) must be run in
            serial so this flag allows them to turn off MPI running.
        :raises SystemExit: If external numerical solver return any failure
            code while running
        """
        # Executable may come with additional sub arguments, we only need to
        # check that the actually executable exists
        exc_check = executable.split(" ")[0]
        if not unix.which(exc_check):
            logger.critical(msg.cli(f"executable '{exc_check}' does not exist",
                            header="external solver error", border="="))
            sys.exit(-1)

        # Prepend with `mpiexec` if we are running with MPI
        # looks something like: `mpirun -n 4 ./bin/xspecfem2d`
        if self._mpiexec and with_mpi:
            executable = f"{self._mpiexec} -n {self.nproc} {executable}"
        logger.debug(f"running executable with cmd: '{executable}'")

        try:
            with open(stdout, "w") as f:
                subprocess.run(executable, shell=True, check=True, stdout=f,
                               stderr=f)
        except (subprocess.CalledProcessError, OSError) as e:
            logger.critical(
                msg.cli("The external numerical solver has returned a "
                        "nonzero exit code (failure). Consider stopping any "
                        "currently running jobs to avoid wasted "
                        "computational resources. Check 'scratch/solver/"
                        f"mainsolver/{stdout}' for the solvers stdout log "
                        "message. The failing command and error message are:",
                        items=[f"exc: {executable}", f"err: {e}"],
                        header="external solver error",
                        border="=")
            )
            sys.exit(-1)

    @staticmethod
    def _exc2log(exc):
        """
        Very simple conversion utility to get log file names based on binaries.
        e.g., binary 'xspecfem2D' will return 'solver'. Helps keep log file
        naming consistent and generalizable

        TODO add a check here to see if the log file exists, and then use
            `number_fid` to increment so that we keep all the output logs

        :type exc: str
        :param exc: specfem executable, e.g., xspecfem2D, xgenerate_databases
        :rtype: str
        :return: logfile name that matches executable name
        """
        convert_dict = {"specfem": "solver", "meshfem": "mesher",
                        "generate_databases": "database", "smooth": "smooth",
                        "combine": "combine"}
        for key, val in convert_dict.items():
            if key in exc:
                return val
        else:
            return "logger"

    def import_model(self, path_model):
        """
        Copy files from given `path_model` into the current working directory
        model database. Used for grabbing starting models (e.g., MODEL_INIT)
        and models that have been perturbed by the optimization library.

        :type path_model: str
        :param path_model: path to an existing starting model
        """
        assert(os.path.exists(path_model)), f"model {path_model} does not exist"
        unix.cd(self.cwd)

        # Copy the model files (ex: proc000023_vp.bin ...) into database dir
        src = glob(os.path.join(path_model, f"*{self._ext}"))
        dst = os.path.join(self.cwd, self.model_databases, "")
        unix.cp(src, dst)

    def _initialize_working_directories(self, max_workers=None):
        """
        Serial or parallel task used to initialize working directories for
        each of the available sources

        :type max_workers: int
        :param max_workers: number of concurrent tasks to use when creating 
            working directories. Defaults to using all available cores on 
            the machine since this is a lightweight task
        """
        if max_workers is None:
            max_workers = unix.nproc() - 1  # use all available cores

        # Full path each source in the scratch directory for directories that
        # do not exist, otherwise this function gets skipped
        source_paths = [os.path.join(self.path.scratch, source_name)
                        for source_name in self.source_names]
        source_paths = [p for p in source_paths if not os.path.exists(p)]

        if source_paths:
            logger.info(f"initializing {self.ntask} solver directories")
        else:
            return

        if max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._initialize_working_directory, cwd)
                    for cwd in source_paths
                ]
            wait(futures)
            # If any of the jobs, calling the result will raise the Exception
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.critical(f"directory initialization error: {e}")
                    sys.exit(-1)
        else:
            for source_name in self.source_names:
                cwd = os.path.join(self.path.scratch, source_name)
                if os.path.exists(cwd):
                    continue
                self._initialize_working_directory(cwd=cwd)

    def _initialize_working_directory(self, cwd=None):
        """
        Creates scratch directory structure expected by SPECFEM
        (i.e., bin, DATA, OUTPUT_FILES). Copies executables (bin) and
        input data (DATA) directories, prepares simulation input files.

        Each directory will act as completely independent Specfem working dir.
        This allows for embarrassing parallelization while avoiding the need
        for intra-directory communications, at the cost of temporary disk space.

        .. note::
            path to binary executables must be supplied by user as SeisFlows has
            no mechanism for automatically compiling from source code.

        :type cwd: str
        :param cwd: optional scratch working directory to intialize. If None,
            will set based on current running seisflows task (self.taskid)
        """
        # Define a constant list of required SPECFEM dir structure, relative cwd
        _required_structure = {"bin", "DATA", "OUTPUT_FILES", "traces/obs", 
                               "traces/syn", "traces/adj", self.model_databases,
                               self.kernel_databases}

        # Allow this function to be called on system or in serial
        if cwd is None:
            cwd = self.cwd
            source_name = self.source_name
        else:
            source_name = os.path.basename(cwd)

        _idx = self.source_names.index(source_name)
        logger.debug(f"source {_idx}: {source_name}")
        # Starting from a fresh working directory
        unix.rm(cwd)
        unix.mkdir(cwd)
        for dir_ in _required_structure:
            unix.mkdir(os.path.join(cwd, dir_))

        # Copy existing SPECFEM exectuables into the bin/ directory
        src = glob(os.path.join(self.path.specfem_bin, "*"))
        dst = os.path.join(cwd, "bin", "")
        unix.cp(src, dst)

        # Copy in all input DATA/ file that are not '{source_prefix}*'
        src = glob(os.path.join(self.path.specfem_data, "*"))
        src = [_ for _ in src if not
               os.path.basename(_).startswith(self.source_prefix)]
        dst = os.path.join(cwd, "DATA", "")
        unix.cp(src, dst)

        # Symlink event source specifically, only retain source prefix
        src = os.path.join(self.path.specfem_data,
                           f"{self.source_prefix}_{source_name}")
        dst = os.path.join(cwd, "DATA", self.source_prefix)
        unix.ln(src, dst)

        # Symlink TaskID==0 as mainsolver in solver directory for convenience
        if self.source_names.index(source_name) == 0:
            if not os.path.exists(self.path._mainsolver):
                logger.debug(f"linking source '{source_name}' as 'mainsolver'")
                unix.ln(cwd, self.path._mainsolver)

    def _export_starting_models(self, parameters=None):
        """
        Export the initial and target models to the SeisFlows output/ directory.
        These are not used for actual simulations, just to keep track of models
        that were used during the workflow.

        :type parameters: list
        :param parameters: list of parameters to export. If None, will default
            to `self._parameters`
        """
        if parameters is None:
            parameters = self._parameters

        # Export the initial and target models to the SeisFlows output directory
        for name, model in zip(["MODEL_INIT", "MODEL_TRUE"],
                               [self.path.model_init, self.path.model_true]):

            # Skip over if user has not provided model path (e.g., real data
            # inversion will not have `model_true`)
            if not model:
                continue
            
            # e.g., output/MODEL_INIT/*
            dst = os.path.join(self.path.output, name, "")

            if not os.path.exists(dst):
                unix.mkdir(dst)
            for par in parameters:
                # Do not try to export over existing files
                if glob(os.path.join(dst, f"*{par}{self._ext}")):
                    continue
                src = glob(os.path.join(model, f"*{par}{self._ext}"))
                unix.cp(src, dst)

    def make_output_vtk_files(self, input_path, output_path=None, 
                              parameters=None, hi_res=False, tag=None, 
                              kernel=False):
        """
        A warpper on `combine_vol_data_vtk()` that automatically tries to 
        generate .vtk files using the SPECFEM binary xcombine_vol_data_vtk, 
        and rename the output files to not be so generic. Files will be stored
        in the `output_path` directory, and will be named based on the `tag` 
        unless overwritten by the User.

        :type input_path: str
        :param input_path: path to database files to be summed.
        :type output_path: strs
        :param output_path: path to export the outputs of the binary
        :type parameters: list
        :param parameters: optional list of parameters, defaults to 
            `self._parameters` if None provided (e.g., ['vp', 'vs'])
        :type tag: str
        :param tag: optional tag to rename output vtk files. If not provided,
            will use the name of the directory holding the files
        :type kernel: bool
        :param kernel: whether the files being converted are kernel files or
            model files. This changes the file naming convention
        :type hi_res: bool
        :param hi_res: Set the high resolution flag to 1 or True, which will
            generate .vtk files with data at EACH GLL point, rather than at each
            nodal vertex. These files are LARGE, and we discourage using
            `hi_res`==True unless you know you want these files.
        """
        # Check that we are using the correct Solver type (3D, 3D_GLOBE)
        if not hasattr(self, "combine_vol_data_vtk"):
            logger.warning("solver does not have the capability to generate "
                           "VTK files, skipping")
            return
        elif not os.path.exists(os.path.join(self.path.specfem_bin, 
                                             "xcombine_vol_data_vtk")):
            logger.warning("solver does not have the required binary "
                           "'xcombine_vol_data_vtk', please compile this "
                           "binary to make VTK files. Skipping ")
            return
        
        # Set some default parameters if not overwritten by User
        if not output_path:
            output_path = os.path.join(self.path._solver_output, "VTK")

        # Determine how to rename files after creation
        if not tag:
            tag = os.path.basename(input_path)
        
        # Set which parameters will be made into VTK files. Do not re-create
        # files that already exist. Add '_kernel' for kernel and gradient files
        if not parameters:
            parameters = []
            for par in self._parameters:
                if kernel:
                    par = f"{par}_kernel"
                # Strip reg?_ from SPECFEM3D_GLOBE parameter names
                # e.g., reg1_vsh -> vsh
                if self._regions:
                    par = par[5:]
                # File naming should follow a standard format that we validate
                check = glob(os.path.join(output_path, f"{tag}*{par}.vtk"))
                if not check:
                    parameters.append(par)
                else:
                    continue
        if not parameters:
            return
        
        self.combine_vol_data_vtk(
            input_path=input_path, output_path=output_path, 
            parameters=parameters, hi_res=hi_res
            )
        # Wait for the process to finish before trying to rename files
        time.sleep(5 * len(parameters))
        
            # SPECFEM3D_GLOBE will tag files based on region
        for par in parameters:
            if self._regions is not None:
                for region in self._regions:
                    src = os.path.join(output_path, f"reg_{region}_{par}.vtk")
                    dst = os.path.join(
                        output_path, f"{tag}_reg_{region}_{par}.vtk")
                    if os.path.exists(src):
                        unix.mv(src, dst)
            else:
                src = os.path.join(output_path, f"{par}.vtk")
                dst = os.path.join(output_path, f"{tag}_{par}.vtk")
                if os.path.exists(src):
                    unix.mv(src, dst)

    def finalize(self):
        """
        General finalization procedures for SPECFEM-based solver activities
        """
        # Generate VTK files for everything in output path
        if self._export_vtk:
            for name in ["MODEL", "GRADIENT"]:
                for fid in glob(os.path.join(self.path.output, f"{name}_*")):
                    self.make_output_vtk_files(
                        input_path=fid, kernel=bool(name=="GRADIENT")
                        )
        

