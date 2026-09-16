#!/usr/bin/env python3
"""
The Pyaflowa preprocessing module for waveform gathering, preprocessing and
misfit quantification. We use the name 'Pyaflowa' to avoid any potential
name overlaps with the actual Pyatoa package.
"""
import os
import sys
import logging
import time
import traceback
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait
from glob import glob
from pyasdf import ASDFDataSet

from pyatoa import Config, Manager, Inspector, ManagerError
from pysep.utils.io import read_events_plus, read_stations

from seisflows import logger
from seisflows.tools import unix
from seisflows.tools.config import Dict
from seisflows.tools.graphics import imgs_to_pdf
from seisflows.tools.specfem import return_matching_waveform_files
from seisflows.preprocess.default import read, initialize_adjoint_traces


class Pyaflowa:
    """
    Pyaflowa Preprocess [Preprocess Base]
    -------------------------------------
    Preprocessing and misfit quantification using Python's Adjoint Tomography
    Operations Assistant (Pyatoa)

    Parameters
    ----------
    :type min_period: float
    :param min_period: Minimum filter corner in unit seconds. Bandpass
        filter if set with `max_period`, highpass filter if set without
        `max_period`, no filtering if not set and `max_period also not set
    :type preproc_toggles: dict
    :param preproc_toggles: A dictionary with keys that represent toggles that
        allow the User to turn on/off the default preprocessing steps. 
        Corresponding values should be 'True' to toggle on, and 'False' for off.
        All toggles are set 'True' by default.
        - 'standardize': resamples and trims time series to match. Turn off if 
          your 'obs' and 'syn' data are already the same length. See 
          `Pyatoa.manager.standardize()`
        - 'preprocess': Detrend, taper, filter and normalize (optional). Turn 
          off if your data are synthetics that do not need filtering, or if
          your data are already preprocessed
        - 'window': misfit windowing using PyFlex. Turn off if you want to 
          compute adjoint sources on the entire trace.
    :type pyflex_parameters: dict
    :param pyflex_parameters: overwrite for Pyflex parameters defined
        in the Pyflex.Config object. Incorrectly defined argument names
        will raise a TypeError. See Pyflex docs for detailed parameter defs:
        http://adjtomo.github.io/pyflex/#config-object
    :type pyadjoint_parameters: dict
    :param pyadjoint_parameters: overwrite for Pyadjoint parameters defined
        in the Pyadjoint.Config object for the given `adj_src_type`.
        Incorrectly defined argument names will raise a TypeError. See
        Pyadjoint docs for detailed parameter definitions:
        https://adjtomo.github.io/pyadjoint/
    :type fix_windows: bool or str
    :param fix_windows: How to address misfit window evaluation at each
        evaluation. Options to re-use misfit windows collected during an
        inversion, available options:
        [True, False, 'ITER', 'ONCE', 'OFF']
        - True: Re-use windows after first evaluation (i01s00);
        - False: Calculate new windows each evaluation;
        - 'ITER': Calculate new windows at first evaluation of
          each iteration (e.g., i01s00... i02s00...
        - 'ONCE': Calculate new windows at first evaluation of
          the workflow, i.e., at self.par.BEGIN
    :type revalidate: bool
    :param revalidate: Only used if `fix_windows` is True, ITER or ONCE. Windows
        that are retrieved from datasets will be revalidated against parameters
        in the Config object (ccshift, dlna, etc.), and rejected if they fall 
        outside defined bounds. If False, windows will not be revalidated. 
        Caution is advised with this parameter as event misfit may be 
        artificially reducedby removing a significant number of windows, which
        is possible when `revalidate`==True
    :type adj_src_type: str
    :param adj_src_type: Adjoint source type to evaluate misfit, defined by
        Pyadjoint. See `pyadjoint.config.ADJSRC_TYPES` for detailed options list
        - 'waveform': waveform misfit function
        - 'convolution': convolution misfit function
        - 'exponentiated_phase': exponentiated phase from Yuan et al. 2020
        - 'cc_traveltime': cross-correlation traveltime misfit
        - 'multitaper': multitaper misfit function
    :type plot_waveforms: bool
    :param plot_waveforms: plot waveform figures and source receiver maps during
        the preprocessing stage. Maps require metadata, and if they are not
        provided then only waveforms + windows + adjoint sources will be plotted
    :type preprocess_log_level: str
    :param preprocess_log_level: Log level to set Pyatoa, Pyflex, Pyadjoint.
        Available: ['null': no logging, 'warning': warnings only,
        'info': task tracking, 'debug': log all small details (recommended)]
    :type export_datasets: bool
    :param export_datasets: periodically save the output ASDFDataSets which
        contain data, metadata and results collected during the
        preprocessing procedure
    :type export_figures: bool
    :param export_figures: periodically save the output basemaps and
        data-synthetic waveform comparison figures
    :type export_log_files: bool
    :param export_log_files: periodically save log files created by Pyatoa

    Paths
    -----
    :type path_preprocess: str
    :param path_preprocess: scratch path for preprocessing related steps
    ***
    """
    def __init__(self, min_period=1., max_period=10., preproc_toggles=None,
                 pyflex_parameters=None, pyadjoint_parameters=None,
                 fix_windows=False, revalidate=False, 
                 adj_src_type="cc_traveltime", plot_waveforms=True, 
                 preprocess_log_level="DEBUG",
                 export_datasets=True, export_figures=True,
                 export_log_files=True, workdir=os.getcwd(),
                 path_preprocess=None, path_solver=None, path_data=None,
                 path_output=None, obs_data_format="SAC",
                 syn_data_format="ASCII", data_case="data", components=None,
                 start=None, ntask=1, nproc=1, source_prefix=None,
                 # >>> NEU PATCH MAX:
                 window_starttime=None, window_endtime=None,
                 # --- Geometry-Fensterung (optional) ---
                 win_mode=None,              # 'off' | 'absolute' | 'relative_t0'
                 t0_source=None,             # 'geometry'
                 vp_ref=None,                # m/s
                 t0_add_s=None,              # globaler Zeitoffset (s)
                 dt_start_s=None,            # Start relativ t0 (s)
                 dt_end_s=None,              # Ende relativ t0 (s)
                 win_taper_frac=None,        # 0..1
                 adj_band_code=None,         # Kanal-Bandcode fuer geschriebene .adj-Dateien
                 **kwargs):

        """
        Pyatoa preprocessing parameters

        .. note::
            Paths and parameters listed here are shared with other modules and
            so are not included in the main class docstring.

        :type syn_data_format: str
        :param syn_data_format: data format for reading synthetic traces into
            memory. Shared with solver module. Pyatoa only works with 'ASCII'
            currently.
        :type data_case: str
        :param data_case: How to address 'data' in the workflow, options:
            'data': real data will be provided by the user in
            `path_data/{source_name}` in the same format that the solver will
            produce synthetics (controlled by `solver.format`) OR
            synthetic': 'data' will be generated as synthetic seismograms using
            a target model provided in `path_model_true`. If None, workflow will
            not attempt to generate data.
        :type components: str
        :param components: components to search for synthetic data with. None by
            default which uses a wildcard when searching for synthetics. If
            provided, User only wants to use a subset of components generated by
            SPECFEM. In that case, `components` should be string of letters such
            as 'ZN' (for up and north components)
        :type workdir: str
        :param workdir: working directory in which to look for data and store
            results. Defaults to current working directory
        :type path_solver: str
        :param path_solver: scratch path for all solver related tasks
        :type path_data: str
        :param path_data: path to any externally stored data required by the 
            solver
        """
        
        #NEU PATCH MAX
        self.window_starttime = window_starttime  # in Sekunden relativ zum ersten Sample
        self.window_endtime   = window_endtime    # in Sekunden relativ zum ersten Sample
        #########
        # --- PATCH MAX - WINDOWING ---
        # --- Geometry-t0 Windowing (Defaults = None, damit YAML übersichtlich bleibt) ---
        self.win_mode       = (win_mode or "off").lower()  # 'off' bewahrt FULLTRACE-Verhalten
        self.t0_source      = None if t0_source is None else str(t0_source).lower()
        # PATCH MAX: Bandcode fuer geschriebene .adj-Dateien. `_form.channel_code(dt)`
        # rekonstruiert einen SEED-Bandcode rein aus der Abtastrate; bei unserer
        # extrem hohen Abtastrate (dt~8e-08s) schlaegt die SEED-Lookup-Tabelle fehl
        # und der Fallback griff (Default "B", passend zu CMTSOLUTION/2D-Kanaelen
        # wie "BXZ"). Unsere 3D-FORCESOLUTION-Quelle erzeugt aber "FXX/FXY/FXZ" --
        # SPECFEM3D's Adjoint-Solver erwartet .adj-Dateien mit GENAU demselben
        # Kanalcode wie die Vorwaertssynthetik, sonst liest er (stillschweigend)
        # Null-Adjointquellen ein -> Nullkernel trotz echtem Misfit. Default "B"
        # bewahrt 2D-Verhalten; wir setzen "F" explizit in parameters.yaml.
        self.adj_band_code = (adj_band_code or "B").upper()[:1]
        
        self.vp_ref         = None if vp_ref         is None else float(vp_ref)
        self.t0_add_s       = None if t0_add_s       is None else float(t0_add_s)
        self.dt_start_s     = None if dt_start_s     is None else float(dt_start_s)
        self.dt_end_s       = None if dt_end_s       is None else float(dt_end_s)
        self.win_taper_frac = None if win_taper_frac is None else float(win_taper_frac)
        ##########################################
        # Pyatoa related parameters
        self.min_period = min_period
        self.max_period = max_period
        self.fix_windows = fix_windows
        self.revalidate = revalidate
        self.adj_src_type = adj_src_type
        self.plot_waveforms = plot_waveforms
        self.preprocess_log_level = preprocess_log_level
        if preproc_toggles is None: 
            self.preproc_toggles = Dict(standardize=True, preprocess=True, window=True)
        else:
            self.preproc_toggles = Dict(preproc_toggles)

        # Set the Pyflex and Pyadjoint external parameters
        _cfg = Config(adj_src_type=adj_src_type,
                      pyflex_parameters=pyflex_parameters,
                      pyadjoint_parameters=pyadjoint_parameters)
        self.pyflex_parameters = {
            key: val for key, val in _cfg.pfcfg.items() if key not in
            ["min_period", "max_period"]}
        # Ignore meta-config parameters that are hardcoded in lower levels
        self.pyadjoint_parameters = {
            key: val for key, val in _cfg.pacfg.items() if key not in
            ["min_period", "max_period", "adjsrc_type", "double_difference"]
        }

        # How to handle saving output data to disk
        self.export_datasets = export_datasets
        self.export_figures = export_figures
        self.export_log_files = export_log_files

        # Preprocessing path variables that allow the module to keep track of IO
        self.path = Dict(
            scratch=path_preprocess or os.path.join(workdir, "scratch",
                                                    "preprocess"),
            solver=path_solver or os.path.join(workdir, "scratch", "solver"),
            output=path_output or os.path.join(workdir, "output"),
            data=path_data,
        )

        # Pyatoa-specific internal directory structure for storing data within
        # the SeisFlows scratch/ directory
        self.path["_logs"] = os.path.join(self.path.scratch, "logs")
        self.path["_tmplogs"] = os.path.join(self.path._logs, "tmp")
        self.path["_datasets"] = os.path.join(self.path.scratch, "datasets")
        self.path["_figures"] = os.path.join(self.path.scratch, "figures")
        self.path["_preproc_output"] = os.path.join(self.path.output, 
                                                    "preprocess")

        # Parameters that are defined by other modules but accessed by Pyaflowa
        self.syn_data_format = syn_data_format.upper()
        self.obs_data_format = obs_data_format.upper()
        self.source_prefix = source_prefix
        self._data_case = data_case.lower()
        self._start = start
        self._ntask = ntask
        self._nproc = nproc
        self._source_prefix = source_prefix
        if components is not None:
            self._components = list(components)  # e.g. 'RTZ' -> ['R', 'T', 'Z']
        else:
            self._components = components

        # Internal acceptable definitions to check against User-set parameters
        self._syn_acceptable_data_formats = ["ASCII"]
        self._acceptable_source_prefixes = ["SOURCE", "FORCESOLUTION",
                                            "CMTSOLUTION"]
        self._acceptable_fix_windows = ["ITER", "ONCE", "FULLTRACE", True, False]

        # Internal bookkeeping attributes to be filled in by self.setup()
        self._inv = None
        self._config = None
        self._fix_windows = False

    def check(self):
        """ 
        Checks Parameter and Path files, will be run at the start of a Seisflows
        workflow to ensure that things are set appropriately.
        """
        assert(self.syn_data_format.upper() == "ASCII"), \
            "Pyaflowa preprocess requires `syn_data_format`=='ASCII'"

        assert(self._source_prefix in self._acceptable_source_prefixes), (
            f"Pyaflowa can only accept `source_prefix` in " 
            f"{self._acceptable_source_prefixes}, not '{self._source_prefix}'"
        )
        assert(self._fix_windows in self._acceptable_fix_windows), \
            f"Pyaflowa `fix_windows` must be in {self._acceptable_fix_windows}"
        
        for key in ["standardize", "preprocess", "window"]:
            assert(key in self.preproc_toggles), \
                f"Pyaflowa `preproc_toggles` missing key {key}"

        # --- sanity checks für Windowing ---
        if self.win_mode == "relative_t0":
            # Quelle der t0-Schätzung: aktuell nur 'geometry' vorgesehen
            assert self.t0_source in ["geometry"], \
                f"t0_source must be 'geometry' for win_mode='relative_t0' (got {self.t0_source!r})"
        
            # Pflicht-Parameter müssen gesetzt sein (keine None mehr erlaubt)
            for name, val in [
                ("vp_ref",         self.vp_ref),
                ("dt_start_s",     self.dt_start_s),
                ("dt_end_s",       self.dt_end_s),
                ("win_taper_frac", self.win_taper_frac),
            ]:
                assert val is not None, f"{name} must be set for win_mode='relative_t0'"
        
            assert self.vp_ref > 0.0, "vp_ref must be > 0.0 m/s"
            assert self.dt_end_s > self.dt_start_s, \
                f"dt_end_s ({self.dt_end_s}) must be > dt_start_s ({self.dt_start_s})"
            assert 0.0 <= self.win_taper_frac < 1.0, \
                f"win_taper_frac ({self.win_taper_frac}) must be in [0,1)"
        
            # t0_add_s darf optional None sein → dann 0.0 verwenden
            if self.t0_add_s is None:
                self.t0_add_s = 0.0



    def setup(self):
        """
        Sets up data preprocessing machinery by establishing an internally
        defined directory structure that will be used to store the outputs 
        of the preprocessing workflow
        """
        # Create the internal directory structure required for storing results
        for pathname in ["scratch", "_logs", "_tmplogs", "_datasets",
                         "_figures", "_preproc_output"]:
            unix.mkdir(self.path[pathname])

        if self._data_case == "synthetic":
            st_obs_type = "syn"
        else:
            st_obs_type = "obs"

        # Convert SeisFlows user parameters into Pyatoa config parameters
        self._config = Config(
            min_period=self.min_period, max_period=self.max_period,
            adj_src_type=self.adj_src_type, component_list=self._components,
            st_obs_type=st_obs_type, st_syn_type="syn",
            pyflex_parameters=self.pyflex_parameters,
            pyadjoint_parameters=self.pyadjoint_parameters
        )

    @staticmethod
    def ftag(config):
        """
        Create a re-usable file tag from the Config object as multiple functions
        will use this tag for file naming and file discovery.

        :type config: pyatoa.core.config.Config
        :param config: Configuration object that must contain the 'event_id',
            iteration and step count
        """
        return f"{config.event_id}_{config.iter_tag}{config.step_tag}"

    ########### PATCH MAX - WINDOWING ##############
    def _parse_specfem2d_stations(self, stations_fid):
        """
        Robust für 2D-STATIONS:
        Spalten können (STA, NET, X, Z, ..) oder (NET, STA, X, Z, ..) sein.
        X/Z in Metern erwartet.
        """
        mp = {}  # (net, sta) -> (x, z)
        if not os.path.isfile(stations_fid):
            return mp
        with open(stations_fid, "r") as fr:
            for line in fr:
                s = line.strip().split()
                if len(s) < 4:
                    continue
                # Erkennen, ob s[0],s[1] textuell sind
                def _isfloat(tok):
                    try:
                        float(tok); return True
                    except Exception:
                        return False
                # zwei Varianten: (STA, NET, X, Z) oder (NET, STA, X, Z)
                if not _isfloat(s[0]) and not _isfloat(s[1]) and _isfloat(s[2]) and _isfloat(s[3]):
                    sta, net, x, z = s[0], s[1], float(s[2]), float(s[3])
                elif not _isfloat(s[0]) and not _isfloat(s[1]) and _isfloat(s[2]):
                    # fallback
                    sta, net, x, z = s[0], s[1], float(s[2]), float(s[3])
                else:
                    # (NET, STA, X, Z)
                    net, sta, x, z = s[0], s[1], float(s[2]), float(s[3])
                mp[(net, sta)] = (x, z)
        return mp
    
    def _parse_specfem2d_source(self, source_fid):
        """
        Liest xs/zs (m) + optional tshift aus SPECFEM2D SOURCE/CMTSOLUTION/FORCESOLUTION.
        Robust gegen Fortran-Zahlen (…d0 / …D+03) und unterschiedliche Zeilenformate.
        """
        import re
    
        xs = zs = None
        tshift = 0.0
        if not os.path.isfile(source_fid):
            return xs, zs, tshift
    
        # Zahl mit optionalem Exponent, erlaubt d/D anstelle von e/E
        num = re.compile(r'([-+]?\d+(?:\.\d+)?(?:[eEdD][+-]?\d+)?|\d+(?:\.\d+)?)')
    
        def _grab_float(line):
            m = num.search(line)
            if not m:
                return None
            s = m.group(1).replace('D', 'e').replace('d', 'e')
            try:
                return float(s)
            except Exception:
                return None
    
        with open(source_fid, "r") as fr:
            for raw in fr:
                low = raw.lower()
                # xs
                if ("x_source" in low) or re.match(r'^\s*xs\b', low):
                    val = _grab_float(raw)
                    if val is not None:
                        xs = float(val)
                # zs
                if ("z_source" in low) or re.match(r'^\s*zs\b', low):
                    val = _grab_float(raw)
                    if val is not None:
                        zs = float(val)
                # optionaler Zeitversatz
                if any(k in low for k in ["t0", "tshift", "time_shift", "time shift"]):
                    val = _grab_float(raw)
                    if val is not None:
                        tshift = float(val)
    
        return xs, zs, tshift

    
    def _geometry_t0_seconds(self, event_id, net, sta):
        """
        t0 aus Geometrie: Distanz/ vp_ref + t0_add_s (+ optional SOURCE-tshift).
        """
        stations_fid = os.path.join(self.path.solver, event_id, "DATA", "STATIONS")
        mp = getattr(self, "_stations_cache", None)
        if mp is None or getattr(self, "_stations_cache_fid", "") != stations_fid:
            mp = self._parse_specfem2d_stations(stations_fid)
            self._stations_cache = mp
            self._stations_cache_fid = stations_fid
    
        source_fid = os.path.join(self.path.solver, event_id, "DATA", self._source_prefix)
        xs, zs, tshift_src = self._parse_specfem2d_source(source_fid)
    
        if xs is None or zs is None or (net, sta) not in mp:
            return None  # kein t0 möglich
    
        xi, zi = mp[(net, sta)]
        dist = float(((xi - xs) ** 2 + (zi - zs) ** 2) ** 0.5)  # m
        t0 = dist / max(self.vp_ref, 1e-9) + float(self.t0_add_s) + float(tshift_src)
        return max(t0, 0.0)
 #######################################################   


    def quantify_misfit(self, source_name=None, save_residuals=None,
                        export_residuals=None, save_adjsrcs=None,
                        components=None, iteration=1, step_count=0,
                        _serial=False, **kwargs):
        """
        Main processing function to be called by Workflow module. Generates
        total misfit and adjoint sources for a given event with name 
        `source_name`.

        .. note::

            Meant to be called by `workflow.evaluate_objective_function` and
            run on system using system.run() to get access to compute nodes.

        :type source_name: str
        :param source_name: name of the event to quantify misfit for. If not
            given, will attempt to gather event id from the given task id which
            is assigned by system.run()
        :type save_residuals: str
        :param save_residuals: if not None, path to write misfit/residuls to
        :type export_residuals: str
        :param export_residuals: export all residuals (data-synthetic misfit)
            that are generated by the external solver to `path_output`. If
            False, residuals stored in scratch may be discarded at any time in
            the workflow
        :type save_adjsrcs: str
        :param save_adjsrcs: if not None, path to write adjoint sources to
        :type components: list
        :param components: optional list of components to ignore preprocessing
            traces that do not have matching components. The adjoint sources for
            these components will be 0. E.g., ['Z', 'N']. If None, all available
            components will be considered.
        :type iteration: int
        :param iteration: current iteration of the workflow, information should
            be provided by `workflow` module if we are running an inversion.
            Defaults to 1 if not given (1st iteration)
        :type step_count: int
        :param step_count: current step count of the line search. Information
            should be provided by the `optimize` module if we are running an
            inversion. Defaults to 0 if not given (1st evaluation)
        :type _serial: bool
        :param _serial: debug function to turn preprocessing to a serial task
            whereas it is normally a multiprocessed parallel task
        """
        # Set a unique Config object to specify which Source we want to process
        config = self._config.copy()
        config.event_id = source_name
        config.iteration = iteration
        config.step_count = step_count
        config.fix_windows = self.fix_windows
        if components is not None:
            config.component_list = components

        # Generate empty adjoint sources and return a matching list of files
        # that will be fed into the misfit quantification machinery
        obs, syn = self._setup_quantify_misfit(source_name, save_adjsrcs,
                                               components)

        # Process each pair in serial.
        if _serial:
            total_misfit, total_windows, total_raw_misfit = 0, 0, 0.0
            for o, s in zip(obs, syn):
                misfit, nwin, raw = self._quantify_misfit_single(o, s, config,
                                                            save_adjsrcs)

                total_misfit += misfit or 0
                total_windows += nwin or 0
                total_raw_misfit += raw or 0.0
        # Process each pair in parallel. Max workers is total num. of cores
        else:
            # Pre-fetch cartopy Natural Earth data single-threaded to prevent
            # concurrent workers from corrupting the downloaded shapefiles
            if self.plot_waveforms:
                try:
                    import cartopy.io.shapereader as shpreader
                    shpreader.natural_earth(resolution='10m', category='physical',
                                           name='coastline')
                except Exception:
                    pass
            with ProcessPoolExecutor(max_workers=unix.nproc()) as executor:
                futures = [
                    executor.submit(self._quantify_misfit_single, o, s, config,
                                    save_adjsrcs) for o, s in zip(obs, syn)
                ]
            wait(futures)

            # Initialize empty values to store statistics on entire misfit quant
            total_misfit, total_windows, total_raw_misfit = 0, 0, 0.0
            for future in futures:
                misfit, nwin, raw = future.result()

                total_misfit += misfit or 0
                total_windows += nwin or 0
                total_raw_misfit += raw or 0.0

        logger.info(f"{source_name}; misfit={total_misfit:.2E}; "
                    f"number of windows={total_windows}")

        # --- PATCH: klassischen (un-normierten) Misfit persistent speichern ---
        # Es werden ZWEI Werte pro Quelle geloggt:
        #   s_j_raw   = total_raw_misfit = 0.5*Integral(syn-obs)^2 dt auf den
        #               ROHEN (standardisiert, ungefiltert/ungetapert) Spuren.
        #               -> KONSISTENT mit der Forward-Nachrechnung
        #               (compute_misfit_curve.py) und step2 (roh). DIES ist die
        #               Groesse fuer die Publikationskurve. Wird waehrend der
        #               Inversion berechnet -> keine teure Nachrechnung noetig
        #               (wichtig fuer 3D!).
        #   s_j_filt  = total_misfit = Pyadjoints Misfit auf den gefilterten/
        #               gefensterten Spuren (das, was die Inversion minimiert).
        #               Nur als Cross-Check.
        # Rein additives Logging - die eigentliche Inversion ist NICHT betroffen.
        try:
            _cm_file = os.path.join(self.path.output, "classical_misfit.txt")
            if not os.path.exists(_cm_file):
                # Header nur einmal, race-sicher via exklusivem Anlegen ('x')
                try:
                    with open(_cm_file, "x") as _f:
                        _f.write("# iteration step_count source_name "
                                 "s_j_raw(=0.5*int_diff^2_dt_roh) "
                                 "s_j_filtered n_windows\n")
                except FileExistsError:
                    pass
            # Eine kurze Zeile pro Quelle -> atomarer Append (POSIX, < PIPE_BUF),
            # damit parallele Sigma-/Quellprozesse sich nicht ins Gehege kommen.
            with open(_cm_file, "a") as _f:
                _f.write(f"{iteration} {step_count} {source_name} "
                         f"{total_raw_misfit:.6e} {total_misfit:.6e} "
                         f"{total_windows}\n")
        except Exception as _e:
            logger.warning(f"[classical_misfit] konnte Wert nicht speichern: {_e}")
        # --- ENDE PATCH ---

        # Save residuals to external file for Workflow to calculate misfit `f`
        # Slightly different than Default preprocessing because we need to
        # normalize by the total number of windows
        if save_residuals:
            # Normalize the raw misfit by the number of measurements
            if total_windows != 0:
                summed_misfit = total_misfit / total_windows
            # Edge case where number of windows is 0 (or we didn't pick windows)
            else:
                logger.warning("0 windows found, will not normalize raw misfit")
                summed_misfit = total_misfit
            # PATCH MAX: war ".2E" (nur 3 signifikante Stellen). Diese Datei
            # wird von workflow.inversion.sum_residuals() zurückgelesen, um
            # den eigentlichen Line-Search-Zielfunktionswert `f_new`/`f_try`
            # zu bilden. Bei einem vorsichtigen ersten Line-Search-Schritt
            # (kleine `step_len_init`-skalierte Modelländerung) liegt die
            # echte Misfit-Aenderung leicht unterhalb dieser 3-stelligen
            # Aufloesung -> f_new und f_try wurden auf denselben gerundeten
            # Text abgeschnitten, obwohl Modell und Synthetik sich nachweislich
            # unterschieden ("LINE SEARCH STALLED" trotz echter Aenderung).
            # Praezision an die an anderer Stelle im Optimizer bereits
            # verwendete Konvention angleichen (siehe optimize/gradient.py
            # ":.16E"-Formatierungen).
            with open(save_residuals, "w") as f:
                f.write(f"{summed_misfit:.16E}\n")
            if export_residuals:
                unix.cp(src=save_residuals, dst=export_residuals)

        # Combine all the individual .png files created into a single PDF for
        # easier scrolling convenience
        if self.plot_waveforms:
            # Merge all .png files to a .pdf
            fids = sorted(glob(os.path.join(self.path._figures,
                                            f"{source_name}*.png")))
            fid_out = os.path.join(self.path._figures,
                                   f"{self.ftag(config)}.pdf")
            imgs_to_pdf(fids, fid_out, remove_fids=True)

        # Collect all temp log files into a single log file
        self._finalize_logging(config, total_windows, total_misfit)

        logger.info(f"FINISH QUANTIFY MISFIT: {source_name}")

    def _setup_quantify_misfit(self, source_name, save_adjsrcs=None,
                               components=None):
        """
        Gather a list of filenames of matching waveform IDs that can be
        run through the misfit quantification step, and generate empty adjoint
        sources so that Solver knows which components are zero'd out.

        :type source_name: str
        :param source_name: the name of the source to process
        :type components: list
        :param components: optional list of components to ignore preprocessing
            traces that do not have matching components. The adjoint sources for
            these components will be 0. E.g., ['Z', 'N']. If None, all available
            components will be considered.
        :rtype: list of tuples
        :return: [(observed filename, synthetic filename)]. tuples will contain
            filenames for matching stations + component for obs and syn
        """
        obs_path = os.path.join(self.path.solver, source_name, "traces", "obs")
        syn_path = os.path.join(self.path.solver, source_name, "traces", "syn")

        # Initialize empty adjoint sources for all synthetics that may or may
        # not be overwritten by the misfit quantification step
        if save_adjsrcs is not None:
            unix.mkdir(save_adjsrcs)
            syn_filenames = glob(os.path.join(syn_path, "*"))
            initialize_adjoint_traces(data_filenames=syn_filenames,
                                      fmt=self.syn_data_format,
                                      path_out=save_adjsrcs)

        # Return a matching list of observed and synthetic waveform filenames
        observed, synthetic = return_matching_waveform_files(
            obs_path, syn_path, obs_fmt=self.obs_data_format,
            syn_fmt=self.syn_data_format, components=components
        )

        # Get station metadata from the STATIONS file to be used for processing
        # We are assuming the directory structure of SPECFEM here
        try:
            self._inv = read_stations(
                os.path.join(self.path.solver, source_name, "DATA", "STATIONS")
            )
        except Exception as e:
            logger.warning(f"cannot read STATIONS file, Pyaflowa will not "
                           f"be able to plot maps: '{e}' ")

        return observed, synthetic

    def _instantiate_manager(self, obs_fid, syn_fid, config):
        """
        Convenience function to return a Manager object that is filled with
        the required data and metadata. This is defined as it's own function
        primarily for debuggin purposes as it allows the User to quickly
        retrieve an object ready for processing
        """
        # Event metadata from SPECFEM DATA/ (CMTSOLUTION, FORCESOLUTION etc.)
        # e.g., scratch/solver/<SOURCE_NAME>/DATA/CMTSOLUTION
        source_fid = os.path.join(self.path.solver, config.event_id, 
                                  "DATA", self._source_prefix)
        cat = read_events_plus(fid=source_fid, format=self._source_prefix)

        # Waveform input; `origintime` will only be applied if format=='ASCII'
        obs = read(fid=obs_fid, data_format=self.obs_data_format,
                   origintime=cat[0].preferred_origin().time)
        syn = read(fid=syn_fid, data_format=self.syn_data_format,
                   origintime=cat[0].preferred_origin().time)

        # Use synthetics to select inventory because it's assumed more stable
        # If none available, default to None meaning no maps can be plotted
        if self._inv:
            inv = self._inv.select(network=syn[0].stats.network,
                                   station=syn[0].stats.station)
        else:
            inv = None

        mgmt = Manager(st_obs=obs, st_syn=syn, event=cat[0], inv=inv,
                       config=config)

        return mgmt

    def _quantify_misfit_single(self, obs_fid, syn_fid, config,
                                save_adjsrcs=False):
        """
        Main Pyatoa processing function to quantify misfit + generation adjsrc.

        Run misfit quantification for a single event-station pair. Gather data,
        preprocess, window and measure data, save adjoint source if
        requested, and then returns the total misfit and the collected
        windows for the station.

        :type obs_fid: str
        :param obs_fid: filename for the observed waveform to be processed
        :type syn_fid: str
        :param syn_fid: filename for the synthetic waveform to be procsesed
        :type config: pyatoa.core.config.Config
        :param config: Config object that defines all the processing parameters
            required by the Pyatoa workflow
        :type save_adjsrcs: str
        :param save_adjsrcs: path to directory where adjoint sources should be
            saved. Filenames will be generated automatically by Pyatoa to fit
            the naming schema required by SPECFEM. If False, no adjoint sources
            will be saved. They of course can be saved manually later using
            Pyatoa + PyASDF
        """
        # Retrieve a Manager object that is alraedy filled with data
        mgmt = self._instantiate_manager(obs_fid, syn_fid, config)

        # SET UP LOGGER
        # Tag is a unique identifier for logs like: 001_i01_s00_XX_XYZ
        tag = f"{self.ftag(config)}_{mgmt.st_syn[0].id.replace('.', '_')}"
        station_logger = self._config_auxiliary_logger(
            fid=os.path.join(self.path._tmplogs, f"{tag}.log")
        )
        # Write a log header to make it easier to sort through logs
        station_logger.info(f"\n{'/' * 80}\n"
                            f"{mgmt.st_syn[0].id:^80}\n"
                            f"{'/' * 80}")

        # Check whether or not we want to use misfit windows from last eval.
        fix_windows, _msg = self._check_fixed_windows(
            iteration=config.iteration, step_count=config.step_count
        )
        station_logger.info(_msg)

        # --- PATCH: klassischer ROHER Misfit (fuer konsistente Publikationskurve).
        # Wird unten aus st_obs_raw/st_syn_raw berechnet (standardisiert=zeitlich
        # aligniert, aber UNgefiltert/UNgetapert). Das ist exakt dieselbe Basis wie
        # die Forward-Nachrechnung (compute_misfit_curve.py) und wie step2 (roh):
        #   s = 1/2 * Integral (syn - obs)^2 dt   pro Kanal, hier ueber die
        # Kanaele dieses Paares summiert. Default 0.0, falls Verarbeitung scheitert.
        raw_misfit = 0.0

        # If any part of this processing fails for whatever reason, move on to
        # plotting and don't let it affect the other tasks
        try:
            if self.preproc_toggles.standardize:
                mgmt.standardize()

            # Copy the standardized version of the waveform to save into the
            # ASDFDataSet because we don't want to save processed versions
            st_obs_raw = mgmt.st_obs.copy()
            st_syn_raw = mgmt.st_syn.copy()

            # --- PATCH: rohen Misfit auf den standardisierten, UNgefilterten
            # Spuren berechnen (Trapez-Integration, wie compute_misfit_curve.py).
            try:
                for _tr_syn in st_syn_raw:
                    _sel = st_obs_raw.select(component=_tr_syn.stats.component)
                    if not _sel:
                        continue
                    _d_syn = _tr_syn.data
                    _d_obs = _sel[0].data
                    _n = min(len(_d_syn), len(_d_obs))
                    _dt = float(_tr_syn.stats.delta)
                    _diff = _d_syn[:_n] - _d_obs[:_n]
                    raw_misfit += 0.5 * float(np.trapz(_diff * _diff, dx=_dt))
            except Exception as _e:
                station_logger.warning(f"[classical_misfit raw] Berechnung "
                                       f"fehlgeschlagen: {_e}")

            # Filter waveforms
            if self.preproc_toggles.preprocess:
                mgmt.preprocess(remove_response=False)

            if self.preproc_toggles.window:
                if not fix_windows:
                    mgmt.window()  
                else:
                    # Determine components from waveforms on the fly so that we 
                    # can use that information to only select windows we need
                    components = [tr.stats.component for tr in mgmt.st_syn]

                    # Retrieve windows from the last available evaluation. Wrap 
                    # in try-except block incase file lock is in place by other 
                    # procs.
                    while True:
                        try:
                            with ASDFDataSet(
                                    os.path.join(self.path["_datasets"],
                                                f"{config.event_id}.h5"),
                                    mode="r") as ds:
                                mgmt.retrieve_windows_from_dataset(
                                        ds=ds, components=components,
                                        revalidate=self.revalidate
                                        )
                            break
                        except (BlockingIOError, FileExistsError):
                            # Random sleep time [0,1]s to decrease chances of 
                            # two processes attempting to access at exactly the 
                            # same time
                            time.sleep(random.random())
                    del ds

            # ---PATCH MAX: MANUELLES WINDOWING: relative t0 aus Geometrie ---
            if (self.preproc_toggles.window
                and self.win_mode == "relative_t0"
                and self.t0_source == "geometry"):
            
                from pyflex.window import Window
            
                windows = {}
                # Komponenten aus tatsächlichen Spuren ableiten (robust)
                comps = sorted({tr.stats.component for tr in mgmt.st_obs})
            
                n_overridden = 0
                for comp in comps:
                    trsel = mgmt.st_obs.select(component=comp)
                    if not trsel:
                        continue
                    tr = trsel[0]
                    n  = tr.stats.npts
                    dt = tr.stats.delta
            
                    net = getattr(tr.stats, "network", "")
                    sta = getattr(tr.stats, "station", "")
            
                    t0 = self._geometry_t0_seconds(config.event_id, net, sta)
                    station_logger.info(f"[geom] net.sta={net}.{sta} t0={None if t0 is None else t0*1e3:.3f} ms "
                    f"n={n} dt={dt*1e6:.1f} µs")
                    if t0 is None:
                        continue  # keine Geometrie -> kein Fenster
            
                    # Start/Ende relativ zu t0
                    wstart = t0 + float(self.dt_start_s)
                    wend   = t0 + float(self.dt_end_s)
            
                    # in Sample-Indizes (inkl. Clipping) umrechnen
                    left_idx  = max(0,            int(np.floor(wstart / dt)))
                    right_idx = min(n - 1,        int(np.ceil (wend   / dt)))
                    
                    station_logger.info(f"[geom-win] {net}.{sta}.{comp} idx=[{left_idx},{right_idx}] "
                    f"t_start={(left_idx*dt):.6f}s t_end={(right_idx*dt):.6f}s")
                    
                    if right_idx <= left_idx:
                        continue  # leeres/ungültiges Fenster
            
                    # Fensterobjekt erstellen
                    w = Window(
                        left=left_idx,
                        right=right_idx,
                        center=(left_idx + right_idx) // 2,
                        time_of_first_sample=tr.stats.starttime,
                        dt=dt,
                        min_period=mgmt.config.min_period,
                        channel_id=f"{tr.stats.network}.{tr.stats.station}..{tr.stats.channel}",
                    )
                    # für Plot/Annos
                    w.max_cc_value = 1.0
                    w.cc_shift = 0
                    w.dlnA = 0.0
            
                    # Taper-Fraktion (Pyflex wertet die Grenzen; Cosine-Taper sitzt in AdjSrc)
                    # Wir speichern sie im Window-Objekt als Meta (Pyflex nutzt 'taper_percentage')
                    try:
                        w.taper_percentage = float(self.win_taper_frac)
                    except Exception:
                        pass
            
                    windows.setdefault(comp, []).append(w)
                    n_overridden += 1
            
                if windows:
                    mgmt.windows = windows
                    mgmt.stats.nwin = sum(len(v) for v in windows.values())
                    station_logger.info(
                        f"[manual-window geometry] event={config.event_id} "
                        f"vp_ref={self.vp_ref:.1f} m/s, dt_start={self.dt_start_s*1e3:.2f} ms, "
                        f"dt_end={self.dt_end_s*1e3:.2f} ms, taper={self.win_taper_frac:.2f}, "
                        f"assigned_windows={mgmt.stats.nwin}"
                    )
                else:
                    station_logger.warning("[manual-window geometry] no windows assigned; falling back to existing windows")

            ########################################################################################################
                    
            # Calculate adjoint source
            
                        # --- FULLTRACE in genau den Fällen, wo wir es brauchen -----------------------
            # Aktiv, wenn:
            #   - cfg.fix_windows == "FULLTRACE"   (explizit), ODER
            #   - windowing global aus ist, ODER
            #   - Pyflex 0 Fenster gefunden hat
            from pyflex.window import Window
            
            cfg = mgmt.config
            ### Patch Max logger
            logger.info("[win-debug] win_mode=%s t0_source=%s preproc_window=%s "
            "cfg.fix_windows=%s nwin_after_manual=%s",
            self.win_mode, self.t0_source, self.preproc_toggles.window,
            str(getattr(cfg, "fix_windows", "")),
            str(getattr(mgmt.stats, "nwin", None)))

            want_fulltrace = (
                str(getattr(cfg, "fix_windows", "")).upper() == "FULLTRACE"
                or not self.preproc_toggles.window
                or getattr(mgmt.stats, "nwin", 0) == 0
            )
            
            ############ PATCH MAX FULLTRACE STARTTIME ENDTIME
            if want_fulltrace:
                windows = {}
                # Komponenten aus Config, sonst aus tatsächlichen Spuren ableiten
                comps = list(getattr(cfg, "components", "") or "")
                if not comps:
                    comps = sorted({tr.stats.component for tr in mgmt.st_obs})
            
                # Custom-Zeiten NUR verwenden, wenn FULLTRACE EXPLIZIT gesetzt ist
                custom_bounds_allowed = (str(getattr(cfg, "fix_windows", "")).upper() == "FULLTRACE")
            
                for comp in comps:
                    trsel = mgmt.st_obs.select(component=comp)
                    if not trsel:
                        continue
                    tr = trsel[0]
                    n  = tr.stats.npts
                    dt = tr.stats.delta
            
                    # Default: ganze Spur
                    left_idx, right_idx = 0, n - 1
            
                    # Falls explizit FULLTRACE UND Zeiten gesetzt: in Indizes umrechnen
                    if custom_bounds_allowed and (self.window_starttime is not None or self.window_endtime is not None):
                        wstart = 0.0 if self.window_starttime is None else float(self.window_starttime)
                        wend   = (n - 1) * dt if self.window_endtime is None else float(self.window_endtime)
            
                        # in Sample-Indizes (inkl. Clipping) umrechnen
                        left_idx  = max(0,            int(np.floor(wstart / dt)))
                        right_idx = min(n - 1,        int(np.ceil (wend   / dt)))
            
                        # Falls ungültig, auf Vollspur zurückfallen
                        if right_idx <= left_idx:
                            left_idx, right_idx = 0, n - 1
                            logger.warning("window_starttime/window_endtime ergaben ein leeres Fenster – falle auf Vollspur zurück.")
            
                    w = Window(
                        left=left_idx, right=right_idx, center=(left_idx + right_idx)//2,
                        time_of_first_sample=tr.stats.starttime, dt=dt,
                        min_period=cfg.min_period,
                        channel_id=f"{tr.stats.network}.{tr.stats.station}..{tr.stats.channel}",
                    )
                    # Dummy-Werte für Plot/Annos
                    w.max_cc_value = 1.0
                    w.cc_shift = 0
                    w.dlnA = 0.0
            
                    windows[comp] = [w]
            
                if windows:
                    mgmt.windows = windows
                    mgmt.stats.nwin = sum(len(v) for v in windows.values())
                    logger.info("FULLTRACE → %d Fenster (Vollspur) für %s", mgmt.stats.nwin, ",".join(windows.keys()))

                    if custom_bounds_allowed and (self.window_starttime is not None or self.window_endtime is not None):
                        logger.info("FULLTRACE (custom) → %d Fenster [%gs, %gs] für %s",
                                    mgmt.stats.nwin,
                                    0.0 if self.window_starttime is None else self.window_starttime,
                                    (n - 1) * dt if self.window_endtime is None else self.window_endtime,
                                    ",".join(windows.keys()))
                    else:
                        logger.info("FULLTRACE → %d Fenster (Vollspur) für %s",
                                    mgmt.stats.nwin, ",".join(windows.keys()))
            # PATCH MAX ENDE --------------------------------------------------------------------------
            
            from pyatoa.utils import form as _form
            import pyatoa.core.manager as _mgr
            _orig = _form.channel_code
            _fallback_band_code = self.adj_band_code
            def _safe(dt):
                try:
                    return _orig(dt)
                except Exception:
                    return _fallback_band_code   # s. adj_band_code (PATCH MAX)
            _form.channel_code = _safe
            _mgr.channel_code  = _safe
            
            mgmt.measure()
            
        except Exception as e:
            station_logger.critical(f"FLOW FAILED:")
            # Get the full traceback and push to logger
            exc_str = traceback.format_exc()
            station_logger.critical(exc_str)
            pass

        # Plot waveform + map figure. Map may fail if we don't have appropriate
        # metadata, in which case we fall back to plotting waveform only
        if self.plot_waveforms:
            # e.g., 001_i01_s00_XX_ABC.png
            save = os.path.join(self.path["_figures"], f"{tag}.png")
            try:
                mgmt.plot(choice="both", show=False, save=save)
            except ManagerError as e:
                station_logger.warning(e)
                mgmt.plot(choice="wav", show=False, save=save)

        # Write out the .adj adjoint source files for solver to discover.
        if mgmt.stats.misfit is not None and save_adjsrcs:
            mgmt.write_adjsrcs(path=save_adjsrcs, write_blanks=False)

        # Write Manager data directly to an ASDFDataSet for data storage and
        # later assessments using the Inspector
        _waited = 0
        while True:
            try:
                ds = None
                with ASDFDataSet(os.path.join(self.path["_datasets"],
                                              f"{config.event_id}.h5"),
                                 mode="a") as ds:
                    # Write the raw, standardized version of the waveforms into
                    # the ASDFDataSet so that it's possible to re-process later
                    mgmt.st_obs = st_obs_raw
                    mgmt.st_syn = st_syn_raw
                    mgmt.write_to_dataset(ds=ds)
                break
            except (OSError, BlockingIOError, FileExistsError):
                # Random sleep time [0,1]s to decrease chances of two processes
                # attempting to access at exactly the same time
                _wait = 10 * random.random()  # (0,10]s
                _waited += _wait
                if _waited > 10 * 60:  # 10m of waiting
                    sys.exit(-1)

                time.sleep(_wait)

        return mgmt.stats.misfit, mgmt.stats.nwin, raw_misfit

    def finalize(self):
        """
        Run serial finalization tasks at the end of a given iteration. These 
        tasks are specific to Pyatoa, used to store figures and data in the
        more permanent output/ directory. Scratch files are deleted during 
        this operation to free up disk space.
        """
        # Create or overwrite the Inspector CSVs used for inversion review
        insp = Inspector()
        insp.discover(path=self.path._datasets)
        insp.save(path=self.path._preproc_output)
    
        # Move datasets/CSVs
        if self.export_datasets:
            src = glob(os.path.join(self.path._datasets, "*.h5"))
            src += glob(os.path.join(self.path._datasets, "*.csv"))
            dst = os.path.join(self.path._preproc_output, "datasets", "")
            unix.mkdir(dst)
            unix.cp(src, dst)
    
        # Organize waveform figures
        if self.plot_waveforms and self.export_figures:
            evaluations = []
            for fid in glob(os.path.join(self.path._figures, "*_i??s??*.pdf")):
                for part in os.path.basename(fid).split(".")[0].split("_"):
                    if part.startswith("i") and len(part) == 6:
                        if part not in evaluations:
                            evaluations.append(part)
    
            for eval_ in set(evaluations):
                dst = os.path.join(self.path._figures, eval_)
                src = glob(os.path.join(self.path._figures, f"*_{eval_}*.pdf"))
                unix.mkdir(os.path.join(self.path._figures, eval_))
                unix.mv(src, dst)
    
            src = glob(os.path.join(self.path._figures, "i??s??"))
            dst = os.path.join(self.path._preproc_output, "figures", "")
            unix.mkdir(dst)
            unix.mv(src, dst)
    
            unix.rm(self.path._figures)
            unix.mkdir(self.path._figures)
    
        # Move logs
        if self.export_log_files:
            evaluations = []
            for fid in glob(os.path.join(self.path._logs, "*_i??s??*.log")):
                for part in os.path.basename(fid).split(".")[0].split("_"):
                    if part.startswith("i") and len(part) == 6:
                        if part not in evaluations:
                            evaluations.append(part)
    
            for eval_ in set(evaluations):
                dst = os.path.join(self.path._logs, eval_)
                src = glob(os.path.join(self.path._logs, f"*_{eval_}*.log"))
                unix.mkdir(os.path.join(self.path._logs, eval_))
                unix.mv(src, dst)
    
            src = glob(os.path.join(self.path._logs, "i??s??"))
            dst = os.path.join(self.path._preproc_output, "logs", "")
            unix.mkdir(dst)
            unix.mv(src, dst)
    
            unix.rm(self.path._logs)
            unix.mkdir(self.path._logs)
            unix.mkdir(os.path.join(self.path._logs, "tmp"))


    def _check_fixed_windows(self, iteration, step_count):
        """
        Determine how to address re-using misfit windows during an inversion
        workflow. Throw some log messages out to let the User know whether or
        not misfit windows will be re used throughout an inversion.

            True: Always fix windows except for i01s00 because we don't have any
                  windows for the first function evaluation
            False: Don't fix windows, always choose a new set of windows
            Iter: Pick windows only on the initial step count (0th) for each
                  iteration. WARNING - does not work well with Thrifty Inversion
                  because the 0th step count is usually skipped
            Once: Pick new windows on the first function evaluation and then fix
                  windows. Useful for when parameters have changed, e.g. filter
                  bounds

        :type iteration: int
        :param iteration: The current iteration of the SeisFlows3 workflow,
            within SeisFlows3 this is defined by `optimize.iter`
        :type step_count: int
        :param step_count: Current line search step count within the SeisFlows3
            workflow. Within SeisFlows3 this is defined by
            `optimize.line_search.step_count`
        :rtype: tuple (bool or None, str)
        :return: (bool on whether to use windows from the previous step or None 
                  if fix window turned off, a message that can be sent to the 
                  logger)
        """
        fix_windows = False
        msg = ""

        # First function evaluation never fixes windows
        if iteration == 1 and step_count == 0:
            fix_windows = False
            msg = "first evaluation of workflow, selecting new windows"
        elif isinstance(self.fix_windows, str):
            # By 'iter'ation only pick new windows on the first step count
            if self.fix_windows.upper() == "ITER":
                if step_count == 0:
                    fix_windows = False
                    msg = "first step of line search, will select new windows"
                else:
                    fix_windows = True
                    msg = "mid line search, fix windows from last evaluation"
            # 'Once' picks windows only for the first function evaluation of
            # the current set of iterations.
            elif self.fix_windows.upper() == "ONCE":
                if iteration == self._start and step_count == 0:
                    fix_windows = False
                    msg = "first evaluation of workflow, selecting new windows"
                else:
                    fix_windows = True
                    msg = "mid workflow, fix windows from last evaluation"
        # Bool fix windows simply sets the parameter
        elif isinstance(self.fix_windows, bool):
            fix_windows = self.fix_windows
            msg = f"fixed windows flag set constant: {self.fix_windows}"

        return fix_windows, msg

    def _config_auxiliary_logger(self, fid):
        """
        Create a log file to track processing of a given source-receiver pair.
        Because each station is processed asynchronously, we don't want them to
        log to the main file at the same time, otherwise we get a random mixing
        of log messages. Instead we have them log to temporary files, which
        are combined at the end of the processing script in serial.

        :type fid: str
        :param fid: full path and filename for logger that will be configured
        :rtype: logging.Logger
        :return: a logger which does NOT log to stdout and only logs to
            the given file defined by `fid`
        """
        handler = logging.FileHandler(fid, mode="w")
        logfmt = "%(asctime)s [%(name)s %(levelname).4s] | %(message)s"
        formatter = logging.Formatter(logfmt, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        for log in ["pyflex", "pyadjoint", "pysep", "pyatoa"]:
            # Set the overall log level
            logger = logging.getLogger(log)
            # Turn off any existing handlers (stream and file)
            while logger.hasHandlers():
                logger.removeHandler(logger.handlers[0])
            # Log to new temporary file
            logger.setLevel(self.preprocess_log_level)
            logger.addHandler(handler)

        return logger

    def _finalize_logging(self, config, total_windows, total_misfit):
        """
        Each source-receiver pair has made its own log file. This function
        collects these files and writes their content back into the main log.
        This is a lot of IO but should be okay since the files are small.

        .. note::

            This was the most foolproof method for having multiple parallel
            processes write to the same file. I played around with StringIO
            buffers and file locks, but they became overly complicated and
            ultimately did not work how I wanted them to. This function trades
            filecount and IO overhead for simplicity.

        .. warning::

            The assumption here is that the number of source-receiver pairs
            is manageable (in the thousands). If we start reaching file count
            limits on the cluster then this method for logging may have to be
            re-thought. See link for example:
            https://stackless.readthedocs.io/en/3.7-slp/howto/
              logging-cookbook.html#using-concurrent-futures-processpoolexecutor

        :type config: pyatoa.core.config.Config
        :param config: Config object that will be queried for iteration, step
            count and event ID information
        :type total_windows: int
        :param total_windows: total number of windows collected for a given
            source. this will be written to the final log message
        :type total_misfit: float
        :param total_misfit: total misfit for a given source. this will be
            written to the final log message
        """
        pyatoa_logger = self._config_auxiliary_logger(
            fid=os.path.join(self.path._logs, f"{self.ftag(config)}.log")
        )
        # Summary log message so that User can quickly assess each source
        pyatoa_logger.info(
            f"\n{'=' * 80}\n{'SUMMARY':^80}\n{'=' * 80}\n"
            f"SOURCE NAME: {config.event_id}\n"
            f"WINDOWS: {total_windows}\n"
            f"RAW MISFIT: {total_misfit:.4f}\n"
            f"\n{'=' * 80}\n{'RAW LOGS':^80}\n{'=' * 80}"
            )

        # Give ample time to ensure logs are available and written
        time.sleep(30)

        # Collect each station's log file and write them into the main file
        tmp_logs = sorted(glob(os.path.join(self.path._tmplogs,
                                            f"{config.event_id}_*.log")))

        with open(pyatoa_logger.handlers[0].baseFilename, "a") as fw:
            for tmp_log in tmp_logs:
                try:
                    with open(tmp_log, "r") as fr:
                        fw.writelines(fr.readlines())
                    unix.rm(tmp_log)  # delete after writing
                except FileNotFoundError:
                    logger.warning(f"error reading {tmp_log}")
                    continue



