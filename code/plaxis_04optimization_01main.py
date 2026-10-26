# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 09:20:29 2026

@author: annal
"""

"""
PLAXIS optimisation. Requires pymoo >= 0.6 and the existing project modules.
Every candidate, including rejected/failed models, is appended to a semicolon
separated CSV immediately. One run directory per algorithm and seed.
"""

import csv
import json
import math
import os
import time
import uuid
import numpy as np
import pandas as pd
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from itertools import product

from plaxis_00calling_02helpers import (open_plaxisData,
                                        close_plaxis)
from plaxis_04optimization_02helpers import (get_buildingData,
                                             fct_calculateCost)

#                       scale   fixed     lower bound   upper bound
DEFAULTSET = {"slab":      [10,     1.00,        0.3,   6.0],
              "ltp":       [10,     0.50,        0.0,   6.0],
              "si_method": ['str',  'VR',         '',    ''],
              "si_num":    ['',    15.00,        1,    30.0],
              "si_L":      [4,      7.00,        2.0,  20.0],
              "si_R":      [100,    0.35,        0.1,   1.4],
              "si_E":      ['',    100e3,         '',    ''],
              }

FIELDS          = tuple(DEFAULTSET)
SCALES          = {key: spec[0]
                   for key, spec in DEFAULTSET.items()
                   if isinstance(spec[0], (int, float))}
DEFAULT_FIXED   = {key: int(spec[1])
                   if key == "si_num" else spec[1]
                   for key, spec in DEFAULTSET.items()}
DEFAULT_BOUNDS  = {key: (spec[2], spec[3])
                   for key, spec in DEFAULTSET.items()
                   if isinstance(spec[2], (int, float))
                   and isinstance(spec[3], (int, float))}

@dataclass
class StudyConfig:
    pathFolder:         str     = "Optimization_study"
    dimension:          str     = "unitcell2"
    device:             int     = 0
    user_methods:       tuple   = ("VR",)
    user_load:          float   = 100.0
    user_slab:          float   = 24.0
    user_soil:          int     = 5
    algorithm:          str     = "GA" # GA, DE, ISRES, MixedVariableGA, grid
    output:             tuple   = ("run", "console")
    vary:               tuple   = ("si_R", "si_L")
    scales:             dict    = field(default_factory=lambda:
                                        SCALES.copy())
    fixed:              dict    = field(default_factory=lambda:
                                        DEFAULT_FIXED.copy())
    bounds:             dict    = field(default_factory=lambda:
                                        DEFAULT_BOUNDS.copy())
    limits:             dict    = field(default_factory=lambda: dict(
                                                             u=0.04,
                                                             u_diff=np.inf,
                                                             distortion=np.inf,
                                                             Mperh=np.inf
                                                             ))
    constraint_mode:    str     = "constraints"  # or "penalty"
    penalty_weights:    dict    = field(default_factory=lambda: dict(
                                                             u=1e7,
                                                             u_diff=1e7,
                                                             distortion=1e5,
                                                             Mperh=1e3
                                                             ))
    calcSize:           int     = 100
    pop_size:           int     = 10
    seed:               int     = 1
    stop_window:        int     = 0 # eval. number (0 disables stagnation stop)
    stop_cost_rel:      float   = 0.001
    stop_cv_abs:        float   = 0.0
    METHOD_LIMITS:      dict    = None
    METHOD_PARAMS:      dict    = None
    METHOD_COSTS:       dict    = None
    algorithm_kwargs:   dict    = field(default_factory=dict)

def _quiet_call(fn, *args, **kwargs): # for open_plaxis_call
    with open(os.devnull, "w") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            return fn(*args, **kwargs)

def _output_limit(error
                  ): # for open_plaxis_call for opt_algo
    message = str(error).lower()
    return ("10001" in message and
            any(phrase in message
                for phrase in ("too many", "maximum number",
                               "max number", "zu viele",
                               "window", "windows", "fenster",
                               "open views", "open plots"
                               )
                )
            )

def open_plaxis_call(fn,
                     *args,
                     **kwargs
                     ): #for opt_algo
   # Close the default Output server & retry once on its window-limit error.
    try:
        return _quiet_call(fn, *args, **kwargs)
    except Exception as error:
        if not _output_limit(error): raise
        _quiet_call(close_plaxis, 10001)
        return _quiet_call(fn, *args, **kwargs)

def _values(name, specification, scales=None):
    if name == "si_method":
        return list(specification)
    if name == "si_E":
        return list(specification)
    if (isinstance(specification, (tuple, list)) and
        len(specification) == 2):
        lo, hi = map(float, specification)
        effective_scales = (SCALES
                            if scales is None
                            else scales)
        if name in effective_scales:
            scale = effective_scales[name]
            a, b = math.ceil(lo * scale - 1e-8), math.floor(hi * scale + 1e-8)
            return [i / scale
                    for i in range(a, b + 1)]
        if name == "si_num":
            return list(range(math.ceil(lo), math.floor(hi) + 1))
    return list(specification)

def _limits(cfg): # for StudyEvaluator
    return (cfg.METHOD_LIMITS or get_buildingData("limits"),
            cfg.METHOD_PARAMS or get_buildingData("params"),
            cfg.METHOD_COSTS  or get_buildingData("costs"))

def _range_value(value,
                 context
                 ): # for _hard_error for _simulate for StudyEvaluator
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("inf",
                 "inf.",
                 "infty",
                 "infinite"):
            return math.inf
        if "*" in s:
            out = 1.0
            for part in s.split("*"):
                part = part.strip()
                out *= context[part] if part in context else float(part)
            return out
    return float(value)

def _hard_error(x, cfg, limits,
                soil_depth  = None,
                si_uk       = None
                ): # for _simulate for StudyEvaluator for opt_algo
    method = x["si_method"]
    if method not in limits:
        return "unknown_method"
    n = int(x["si_num"])
    if (n            <  1
        or x["si_R"] <= 0
        or x["si_L"] <= 0
        or x["slab"] < 0):
        return "invalid_geometry"
    spacing = cfg.user_slab / (n + 1)
    diameter = 2 * x["si_R"]
    context = dict(diameter      = diameter,
                   length        = x["si_L"],
                   ltp_thickness = x["ltp"])
    if spacing <= diameter:
        return "diameter_spacing"
    for key, actual in (("diameter",  diameter),
                        ("spacing",   spacing),
                        ("ltp",       x["ltp"]),
                        ("stiffness", x["si_E"])):
        spec = limits[method].get(key)
        if spec is None:
            continue
        if isinstance(spec, (tuple, list)):
            lo, hi = (_range_value(v, context)
                      for v in spec)
        else:
            lo = hi = _range_value(spec, context)
        if not lo - 1e-8 <= actual <= hi + 1e-8:
            return key + "_limit"
    if "spacing_factor_min" in limits[method]:
        if spacing < _range_value(limits[method]["spacing_factor_min"],
                                  context) * diameter:
            return "spacing_factor"
    if "length_ratio_max" in limits[method]:
        if x["si_L"] / diameter > float(limits[method]["length_ratio_max"]):
            return "slenderness"
    if (soil_depth is not None
        and si_uk is not None
        and float(soil_depth) + 0.5 > float(si_uk)):
        return "depth_boundary"
    return None

def _soft_values(last, h, cfg): # for _simulate for StudyEvaluator
    measures = dict(u           = abs(float(last["u"])),
                    u_diff      = (abs(float(last["u_diff"]))
                                   if np.isfinite(cfg.limits["u_diff"])
                                   else np.nan),
                    distortion  = (abs(float(last["distortion"]))
                                   if np.isfinite(cfg.limits["distortion"])
                                   else np.nan),
                    Mperh       = (abs(float(last["Mglobal"])) / h
                                   if np.isfinite(cfg.limits["Mperh"])
                                   else np.nan)
                    )
    g = {key: measures[key] - cfg.limits[key]
         for key in measures
         if np.isfinite(cfg.limits[key])}
    if not all(np.isfinite(v)
               for v in g.values()):
        raise ValueError("Non-finite active PLAXIS response")
    return measures, g

def _load_response(path_data, dimension): # for _simulate for StudyEvaluator
    suffix = (""
              if dimension == "unitcell2"
              else "_Middle")
    file = os.path.join(path_data,
                        "summary_value" + suffix,
                        "Fsu_phases"    + suffix + ".csv")
    df = pd.read_csv(file, sep=";")
    phase = df.loc[df["Name"].astype(str).str.contains("Load step",na=False)]
    if phase.empty: raise ValueError("No load step in " + file)
    return phase.iloc[-1]

def _simulate(x, cfg,
              path, name,
              method_params,
              method_costs,
              method_limits
              ): # for StudyEvaluator
    # p = method_params[x["si_method"]]
    p = method_params.get(x["si_method"], {})
    spacing = cfg.user_slab / (int(x["si_num"]) + 1) # hardcoded quadratic
    from plaxis_01model_01main import fct_modelParams
    # print(f'simulate {x["slab"]}')
    params = fct_modelParams(disx           = spacing,
                             disy           = spacing,
                             si_method      = x["si_method"],
                             si_length      = x["si_L"],
                             si_radius      = x["si_R"],
                             si_numberx     = int(x["si_num"]),
                             si_numbery     = int(x["si_num"]),
                             si_stiffness   = x["si_E"],
                             si_gamma = (float(np.mean(p["gamma"]))
                                         if p.get("gamma") is not None
                                         else None),
                             si_Rinter = (float(np.mean(p["Rinter"]))
                                          if p.get("Rinter") is not None
                                          else None),
                             soil           = cfg.user_soil,
                             ltp_height     = x["ltp"],
                             slab_height    = x["slab"],
                             intf           = False,
                             line_load      = cfg.user_load,
                             loadsteps      = max(1,
                                                  math.ceil(cfg.user_load
                                                            / 50)))
    error = _hard_error(x,
                        cfg,
                        method_limits,
                        params[0].get("depth"),
                        params[5].get("UK")
                        )
    if error:
        raise ValueError("hard constraint: "
                         + error)
    path_data = os.path.join(path, name + "_" + cfg.dimension)
    from plaxis_01model_01main import fct_model
    open_plaxis_call(fct_model,
                     path_data,
                     "Model",
                     cfg.dimension,
                     cfg._port_ip,
                     cfg._port_pw,
                     *params)
    # print('model done')
    from plaxis_02eval_01main import fct_eval
    open_plaxis_call(fct_eval,
                     path_data,
                     "Model",
                     cfg.dimension,
                     cfg._port_ip,
                     cfg._port_pw,
                     output = ['settlement','tilt'], # add moment if needed
                     accuracy = "bare"
                     )
    # print('eval done')
    last = _load_response(path_data, cfg.dimension)
    h = float(params[2]["height"])
    measures, g = _soft_values(last, h, cfg)
    cost, breakdown = fct_calculateCost(path_data, method_costs)
    if not np.isfinite(cost):
        raise ValueError("Non-finite cost")
    return (cost,
            measures,
            g,
            breakdown)

class StudyEvaluator:

    def __init__(self, cfg, path,
                 simulate=None
                 ): # class StudyEvaluator
        self.cfg, self.path = cfg, path
        self.methods, self.params, self.costs = _limits(cfg)
        for m in cfg.user_methods:
            if (m not in self.methods
                # or m not in self.params
                or m not in self.costs["improvement"]):
                raise ValueError("Incomplete method data: " + m)
        self.methods = {m: self.methods[m]
                        for m in cfg.user_methods}
        self.space = {}
        for key in cfg.vary:
            spec = (cfg.user_methods
                    if key == "si_method"
                    else cfg.bounds.get(key))
            if spec is None:
                if key == "si_E":
                    vals = set()
                    for m in cfg.user_methods:
                        v = self.methods[m].get("stiffness",
                                                self.params[m]["Eref"])
                        vals.update([v]
                                    if np.isscalar(v)
                                    else [v[0], np.mean(v), v[1]])
                    spec = sorted(vals)
                else:
                    raise ValueError("Missing bound for " + key)
            self.space[key] = _values(key, spec, cfg.scales)
            if not self.space[key]:
                raise ValueError("Empty bounds for " + key)
        self.n = 0
        self.simulate = simulate or _simulate
        self.file = os.path.join(path, "optimization_history.csv")
        os.makedirs(path, exist_ok=True)
        self.active = [k
                       for k, v in cfg.limits.items()
                       if np.isfinite(v)]
        if (cfg.constraint_mode not in ("constraints", "penalty")):
            raise ValueError("constraint_mode must be constraints or penalty")
        if (not cfg.user_methods or
            not cfg.vary):
            raise ValueError("Methods and variable list required")
        if (cfg.algorithm.lower() != "grid" and
            (cfg.calcSize < 1
                 or cfg.pop_size < 1)):
            raise ValueError("Positive calcSize and pop_size required"
                             " for an optimizer")
        if (len(set(cfg.vary)) != len(cfg.vary)
            or set(cfg.vary) - set(FIELDS)):
            raise ValueError("Unsupported or duplicate varied variable")
        if ("si_method" not in cfg.vary and
            cfg.fixed["si_method"] not in cfg.user_methods):
            raise ValueError("Fixed method must belong to user_methods")
        if (os.path.exists(self.file) and
            os.path.getsize(self.file)):
            raise FileExistsError("Run directory already has history: "
                                  + self.file)

    def from_vector(self, z): # class StudyEvaluator
        x = dict(self.cfg.fixed)
        for i, (key, vals) in enumerate(self.space.items()):
            idx = min(len(vals) - 1,
                      max(0,
                          int(np.floor(float(z[i]) * len(vals)))))
            x[key] = vals[idx]
        return x

    def evaluate(self, x): # class StudyEvaluator
        self.n += 1
        cfg = self.cfg
        x = dict(x)
        name = "opt_%05d_%s" % (self.n, uuid.uuid4().hex[:8])
        cost, measures = np.nan, {}
        g = {k: 1e6 for k in self.active}
        status, reason, breakdown = "ERROR", "", {}
        hard = _hard_error(x, cfg, self.methods)
        try:
            if hard:
                status, reason = "REJECTED", hard
            else:
                cost, measures, g, breakdown = self.simulate(x,
                                                             cfg,
                                                             self.path,
                                                             name,
                                                             self.params,
                                                             self.costs,
                                                             self.methods)
                status = "OK"
        except Exception as exc:
            reason = "%s: %s" % (type(exc).__name__, exc)
        feasible = status == "OK" and all(v <= 0
                                          for v in g.values())
        penalty = sum(cfg.penalty_weights.get(k, 0) * max(v, 0)
                      for k, v in g.items())
        objective = (cost + penalty
                     if (cfg.constraint_mode == "penalty" and
                         status == "OK")
                     else (cost
                           if status == "OK"
                           else 1e12)
                     )
        G = ([g[k]
             for k in self.active]
             if status == "OK"
             else [1e6] * len(self.active))
        cv_norm = (sum(max(g[k], 0) / max(abs(cfg.limits[k]), 1e-9)
                      for k in self.active)
                   if status == "OK"
                   else np.nan)
        # print(x["slab"])
        row = dict(time         = time.strftime("%Y-%m-%d %H:%M:%S"),
                   evaluation   = self.n,
                   algorithm    = cfg.algorithm,
                   seed         = cfg.seed,
                   name         = name,
                   slab_t       = x["slab"],
                   ltp_t        = x["ltp"],
                   si_method    = x["si_method"],
                   si_num       = x["si_num"],
                   si_L         = x["si_L"],
                   si_R         = x["si_R"],
                   si_E         = x["si_E"],
                   cost         = cost,
                   u            = measures.get("u",             np.nan),
                   u_diff       = measures.get("u_diff",        np.nan),
                   distortion   = measures.get("distortion",    np.nan),
                   Mperh        = measures.get("Mperh",         np.nan),
                   g0_settl     = g.get("u",                    np.nan),
                   g1_sdiff     = g.get("u_diff",               np.nan),
                   g2_tilt      = g.get("distortion",           np.nan),
                   g3_bend      = g.get("Mperh",                np.nan),
                   cv_norm      = cv_norm,
                   penalty      = (penalty
                                   if status == "OK" else       np.nan),
                   objective    = objective,
                   feasible     = feasible,
                   status       = status,
                   reason       = reason,
                   co2_kg       = breakdown.get("co2_kg",       np.nan))
        with open(self.file,
                  "a",
                  newline = "",
                  encoding = "utf-8") as f:
            writer = csv.DictWriter(f,
                                    fieldnames  = row.keys(),
                                    delimiter   = ";")
            if f.tell() == 0: writer.writeheader()
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        if "console" in cfg.output:
            print(f'{self.n:4d} {cfg.algorithm:15s} {status:8s} '
                  f'cost = {cost:,.2f} €, '
                  f'u = {measures.get("u", np.nan):.5f} m, '
                  f'feasible = {feasible} {reason}', flush=True)
        return (float(objective),
                np.asarray(G, dtype=float),
                row)

def _stagnated(evaluator): # used in class StudyTermination
    cfg = evaluator.cfg
    if (not cfg.stop_window or
        evaluator.n < cfg.stop_window + 1):
        return False
    df = pd.read_csv(evaluator.file, sep=";")
    ok = df.loc[df["feasible"].astype(str).str.lower() == "true",
                "cost"]
    if ok.empty:
        return False
    prior = df.iloc[:-cfg.stop_window]
    prior_ok = prior.loc[prior["feasible"].astype(str).str.lower() == "true",
                         "cost"]
    if prior_ok.empty:
        return False
    old, new = prior_ok.min(), ok.min()
    if (abs(old - new) /
        max(abs(old), 1.0)) > cfg.stop_cost_rel:
        return False
    tail = df.tail(cfg.stop_window)
    if not tail["feasible"].astype(str).str.lower().eq("true").any():
        return False
    if cfg.stop_cv_abs:
        cols = ["g0_settl", "g1_sdiff", "g2_tilt", "g3_bend"]
        v = tail[cols].clip(lower=0).fillna(0).sum(axis=1)
        if v.max() - v.min() > cfg.stop_cv_abs:
            return False
    return True


def run_algorithm(ev):
    cfg = ev.cfg

    if cfg.algorithm.lower() == "grid":
        # shapes = [len(v) for v in ev.space.values()]
        # total = math.prod(shapes)
        best = None
        best_design = None
        for vals in product(*ev.space.values()):
            design = dict(cfg.fixed,
                          **dict(zip(ev.space, vals)))
            _, _, row = ev.evaluate(design)
            if (row["feasible"] and
                (best is None or
                 row["cost"] < best["cost"])):
                best        = row
                best_design = design
        from types import SimpleNamespace
        return SimpleNamespace(X = best_design,
                               F = (float(best["cost"])
                                    if best is not None
                                    else None),
                               algorithm = "grid",
                               n_eval    = ev.n)

    else:
        from pymoo.core.problem                      import ElementwiseProblem
        from pymoo.optimize                          import minimize
        from pymoo.core.termination                  import Termination
        from pymoo.algorithms.soo.nonconvex.ga       import GA
        from pymoo.algorithms.soo.nonconvex.de       import DE
        from pymoo.algorithms.soo.nonconvex.isres    import ISRES
        from pymoo.algorithms.soo.nonconvex.pso      import PSO
        from pymoo.algorithms.soo.nonconvex.pattern  import PatternSearch
        from pymoo.core.mixed                        import MixedVariableGA
        from pymoo.core.variable                     import Choice
        n_constr = (len(ev.active)
                    if cfg.constraint_mode == "constraints"
                    else 0)

        if cfg.algorithm == "MixedVariableGA":
            class Problem(ElementwiseProblem):
                def __init__(self):
                    super().__init__(vars={key: Choice(options=vals)
                                           for key, vals in ev.space.items()},
                                     n_obj=1, n_ieq_constr=n_constr
                                     )
                def _evaluate(self, x, out, *args, **kwargs):
                    y = dict(cfg.fixed, **x)
                    out["F"], G, _ = ev.evaluate(y)
                    if n_constr: out["G"] = G
            algorithm = MixedVariableGA(pop_size        = min(cfg.pop_size,
                                                              cfg.calcSize),
                                        n_offsprings    = 1,
                                        eliminate_duplicates = False,
                                        **cfg.algorithm_kwargs
                                        )

        else:
            class Problem(ElementwiseProblem):
                def __init__(self):
                    super().__init__(n_var          = len(ev.space),
                                     n_obj          = 1,
                                     n_ieq_constr   = n_constr,
                                     xl             = np.zeros(len(ev.space)),
                                     xu             = np.ones(len(ev.space))
                                     )
                def _evaluate(self, x, out, *args, **kwargs):
                    out["F"], G, _ = ev.evaluate(ev.from_vector(x))
                    if n_constr: out["G"] = G
            pop = min(cfg.pop_size,
                      cfg.calcSize)
            if cfg.algorithm == "GA":
                algorithm = GA(pop_size             = pop,
                               n_offsprings         = 1,
                               eliminate_duplicates = False,
                               **cfg.algorithm_kwargs
                               )
            elif cfg.algorithm == "DE":
                algorithm = DE(pop_size             = pop,
                               n_offsprings         = 1,
                               **cfg.algorithm_kwargs
                               )
            elif cfg.algorithm == "ISRES":
                algorithm = ISRES(pop_size          = pop,
                                  # AssertionError: The number of offsprings
                                  # should be at least double
                                  # the population size.
                                  n_offsprings      = pop*2,
                                  **cfg.algorithm_kwargs
                                  )
            elif cfg.algorithm == "PSO":
                algorithm = PSO(pop_size=pop,
                                **cfg.algorithm_kwargs)
            elif cfg.algorithm == "PatternSearch":
                algorithm = PatternSearch(**cfg.algorithm_kwargs)
            else:
                raise ValueError("Unknown algorithm: "
                                 + cfg.algorithm)

        class StudyTermination(Termination):
            def _update(self, algorithm):
                return float(ev.n >= cfg.calcSize or _stagnated(ev))

        return minimize(Problem(),
                        algorithm,
                        termination                 = StudyTermination(),
                        seed                        = cfg.seed,
                        verbose                     = "console" in cfg.output,
                        return_least_infeasible     = False)


def write_settings(path, cfg, evaluator): # for opt_algo
    """Record the effective inputs and the actual discrete design space."""
    settings = {key: value
                for key, value in vars(cfg).items()
                if key not in ("_port_ip", "_port_pw")}
    if cfg.algorithm.lower() == "grid":
        settings["calcSize"]     = None
        settings["pop_size"]     = None
    settings["DEFAULTSET"]       = DEFAULTSET
    settings["design_variables"] = {key: ({"role":      "varied",
                                           "values":    evaluator.space[key],
                                           "count":     len(evaluator.space[key]),
                                           "bounds":    cfg.bounds.get(key),
                                           "scale":     cfg.scales.get(key),
                                           "step":      (1 / cfg.scales[key]
                                                         if key in cfg.scales
                                                         else None)
                                           }
                                          if key in evaluator.space
                                          else {"role":      "fixed",
                                                "value":     cfg.fixed[key]
                                                }
                                          )
                                    for key in FIELDS
                                    }
    settings["design_space_size"] = math.prod(map(len,
                                                  evaluator.space.values())
                                              )
    settings["active_limits"] = {key: cfg.limits[key]
                                 for key in evaluator.active
                                 }
    settings["method_data"] = {"limits": evaluator.methods,
                               "params": {key: evaluator.params.get(key)
                                          for key in cfg.user_methods
                                          }, # if not defined = None
                               "costs": evaluator.costs
                               }
    filename = os.path.join(path, "optimization_settings.json")
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(settings,
                  file,
                  indent        = 2,
                  ensure_ascii  = False,
                  default       = str)


def opt_algo(pathFolder          = "Optimization_study",
             dimension           = "unitcell2",
             device              = 0,
             user_methods        = ('VR',
                                    # 'RI', 'RAP',
                                    # 'VS', 'DSM',
                                    # 'JG',
                                    # 'RP', 'DP'
                                    ),
             user_load           = 100,
             user_slab           = 24,
             user_soil           = 5,
             algorithm           = 'GA', # grid ignores calcSize and pop_size
             output              =('run',
                                   'console',
                                   'landscape_bars', # 2D contours
                                   'landscape_surface', # 3D interpolated surf
                                   'convergence',
                                   'evolution',
                                   'cost_settlement',
                                   'parallel',
                                   # 'method_ranges',
                                   ),
             vary                =("si_R",
                                   "si_L"),
             fixed               =None,
             bounds              =None,
             limits              =None,
             constraint_mode     ="constraints",
             penalty_weights     =None,
             calcSize            =100,
             pop_size            =10,
             seed                =1,
             stop_window         =20,    # best costs with dev. after x iter.
             stop_cost_rel       =0.001, # deviation of 0.1%
             stop_cv_abs         =0.0,   # should stay zero
             METHOD_LIMITS       =None,
             METHOD_PARAMS       =None,
             METHOD_COSTS        =None,
             algorithm_kwargs    =None,
             simulate            =None,
             scales              =None
             ):
    if scales is not None:
        unknown = set(scales) - set(SCALES)
        if unknown:
            raise ValueError("Unsupported scale variables: "
                             + ", ".join(sorted(unknown)))
        if any(isinstance(v, (bool, str)) or
               not isinstance(v, (int, float)) or
               not np.isfinite(v) or
               v <= 0 for v in scales.values()):
            raise ValueError("Scales must be finite positive numbers")
    cfg = StudyConfig(pathFolder        = pathFolder,
                      dimension         = dimension,
                      device            = device,
                      user_methods      = tuple(user_methods),
                      user_load         = user_load,
                      user_slab         = user_slab,
                      user_soil         = user_soil,
                      algorithm         = algorithm,
                      output            = tuple(output),
                      vary              = tuple(vary),
                      scales            = dict(SCALES,
                                               **(scales or {})),
                      fixed             = dict(DEFAULT_FIXED,
                                               **(fixed or {})),
                      bounds            = dict(DEFAULT_BOUNDS,
                                               **(bounds or {})),
                      limits            = dict(StudyConfig().limits,
                                               **(limits or {})),
                      constraint_mode   = constraint_mode,
                      penalty_weights   = dict(StudyConfig().penalty_weights,
                                               **(penalty_weights or {})),
                      calcSize          = calcSize,
                      pop_size          = pop_size,
                      seed              = seed,
                      stop_window       = stop_window,
                      stop_cost_rel     = stop_cost_rel,
                      stop_cv_abs       = stop_cv_abs,
                      METHOD_LIMITS     = METHOD_LIMITS,
                      METHOD_PARAMS     = METHOD_PARAMS,
                      METHOD_COSTS      = METHOD_COSTS,
                      algorithm_kwargs  = algorithm_kwargs or {})
    # _, _, path = open_plaxisData(device, 'opt', pathFolder)
    _, _, path = open_plaxisData(device, 'opt', pathFolder, create=False)
    if "run" in cfg.output:
        cfg._port_ip, cfg._port_pw, _ = open_plaxisData(device,
                                                        dimension,
                                                        pathFolder
                                                       )
        ev = StudyEvaluator(cfg,
                            path,
                            simulate=simulate
                            )
        write_settings(path, cfg, ev)
        result = run_algorithm(ev)
        history = pd.read_csv(ev.file, sep=";")
        feasible = history[history.feasible.astype(str).str.lower().eq("true")]
        if not feasible.empty:
            best = feasible.loc[feasible.cost.idxmin()]
            if "console" in cfg.output:
                print("Best feasible:", best[["cost",
                                              "u",
                                              "si_R",
                                              "si_L",
                                               "si_method",
                                               "si_num"]
                                             ].to_dict())
        elif "console" in cfg.output:
            print("No feasible solution in the completed evaluations.")
    else:
        path, result = path, None
        # settings_file = os.path.join(path, "optimization_settings.json")
        # if not os.path.isfile(settings_file):
        #     raise FileNotFoundError(settings_file)
        # with open(settings_file, "r", encoding="utf-8") as f:
        #     settings = json.load(f)
        # for key in ("algorithm",
        #             "dimension",
        #             "user_methods",
        #             "user_load",
        #             "user_slab",
        #             "user_soil",
        #             "vary",
        #             "scales",
        #             "fixed",
        #             "bounds",
        #             "limits",
        #             "constraint_mode",
        #             "penalty_weights",
        #             "calcSize",
        #             "pop_size",
        #             "seed",
        #             ):
        #     if key in settings:
        #         setattr(cfg, key, settings[key])
        # cfg.vary = tuple(cfg.vary)
        # cfg.user_methods = tuple(cfg.user_methods)
    plots = {}
    requested = set(cfg.output) & {
                                   "landscape_bars",
                                   "landscape_surface",
                                    "convergence",
                                    "evolution",
                                    "cost_settlement",
                                    "parallel",
                                    # "method_ranges"
                                    }
    if requested:
        from plaxis_04optimization_03plots import opt_plot
        plots = opt_plot(path, requested, cfg)
    return result, path, plots

if __name__ == "__main__":
    res, path, plots = opt_algo(device=0)
