# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 09:20:29 2026

@author: annal
"""

###############################################################################
#---------------------------------- IMPORTS -----------------------------------
###############################################################################

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from plaxis_03plot_02helpers import fct_setPlottext
from plaxis_00calling_02helpers import (fct_setPlotstyle,
                                        fct_formatSave,
                                        open_plaxisData)
from plaxis_04optimization_02helpers import (get_buildingData,
                                             plot_methodRanges)

###############################################################################
#-------------------------------- USER-INPUT ----------------------------------
###############################################################################

LABELS = dict(si_R           = r"SI radius $r_{\mathrm{SI}}$ [m]",
              si_L           = "SI length $L$ [m]",
              slab_t         = r"Slab thickness $t_{\mathrm{slab}}$ [m]",
              ltp_t          = r"LTP thickness $t_{\mathrm{LTP}}$ [m]",
              si_num         = r"Piles per row $a_x$ [-]",
              si_E           = "SI stiffness [kPa]",
              si_method      = "SI method",
              raster         = "Grid arrangement",
              cost           = "Costs [€]",
              u              = "Settlement [m]",
              g              = "Objectives [-]",
              max_evaluation = "Max. evaluation",
              max_iteration  = "Max. iteration",
              )
PARAM_INFO = {"cost":       ("C",       "€",    "Costs"),
              "si_method":  ("M",       "",     "SI Method"),
              "si_num":     ("N",       "-",    "SI number"),
              "si_L":       ("L",       "m",    "SI length"),
              "si_R":       ("R",       "m",    "SI radius"),
              "si_E":       ("E",       "kPa",  "SI stiffness"),
              "ltp_t":      ("LTP",     "m",    "LTP thickness"),
              "slab_t":     ("Slab",    "m",    "slab thickness"),
              }
PARAMS = ["slab_t", "ltp_t", "si_num", "si_L", "si_R", "si_E",
          "si_method", "raster"]
G_COLS = ["g0_settl", "g1_sdiff", "g2_tilt", "g3_bend"]

###############################################################################
#-------------------------------- DIMENSIONS ----------------------------------
###############################################################################

DEF_cmTOinch  = 2.54

DEF_figWidth   = 15.0 / DEF_cmTOinch
DEF_figHeight  =  7.5 / DEF_cmTOinch

DEF_axisTop    = 1 - 1.2 / DEF_cmTOinch / DEF_figHeight # 84.0%
DEF_axisBottom =     1.2 / DEF_cmTOinch / DEF_figHeight # 16.0%
DEF_axisGapH   =     0.2 / DEF_cmTOinch / DEF_figHeight #  2.7%
DEF_axisSpaceH =     0.6 / DEF_cmTOinch / DEF_figHeight # x%

DEF_axisLeft   =     1.5 / DEF_cmTOinch / DEF_figWidth  # 10.0%
DEF_axisRight  = 1 - 0.5 / DEF_cmTOinch / DEF_figWidth  # 96.7%
DEF_axisGapW   =     0.2 / DEF_cmTOinch / DEF_figWidth  #  1.3%
DEF_axisSpaceW =     2.5 / DEF_cmTOinch / DEF_figWidth  # 16.7%
DEF_figLegendx = 1 - 4.5 / DEF_cmTOinch / DEF_figWidth  # 70.0%

DEF_bestGroupWidth     = 0.72
DEF_bestBarWidthFactor = 0.82
DEF_extraAxis_cm       = 1.2

###############################################################################
#------------------------------- General Helpers ------------------------------
###############################################################################

def _read(paths):
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    frames = []
    for path in paths:
        f = (path
             if str(path).endswith(".csv")
             else os.path.join(path, "optimization_history.csv"))
        if not os.path.isfile(f):
            print(f"_read: File not found, skipping: {f}")
            continue
        df = pd.read_csv(f, sep=";")
        if df.empty:
            print(f"_read: Empty CSV, skipping: {f}")
            continue
        df["source"] = os.path.basename(os.path.dirname(f))
        if "algorithm" not in df:
            df["algorithm"] = df["source"]
        df["algorithm"] = df["algorithm"].astype(str).str.strip()
        df.loc[df["algorithm"].str.lower().eq("mixedvariablega"),
               "algorithm"] = "MVGA"
        if "evaluation" not in df:
            df["evaluation"] = np.arange(1, len(df) + 1)
        df["feasible"] = (df["feasible"].astype(str)
                          .str.lower().eq("true")
                          if "feasible" in df else False
                          )
        frames.append(df)
    if not frames:
        print("_read: No valid optimisation results found.")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def _study_groups(df):
    """Keep separate run folders separate, including runs of the same algo."""
    keys = (["algorithm", "source"]
            if df.source.nunique() > 1
            else ["algorithm"])
    groups = list(df.groupby(keys, sort=False))
    labels = [str(key[0]) + ((f" ({key[1]})"
                             if (len(key) > 1 and
                                 key[1] != key[0])
                             else "")
                             if isinstance(key, tuple)
                             else str(key)
                             )
              for key, _ in groups]
    return list(zip(labels, (group
                             for _, group in groups)))

def _run_title(base, df, groups):
    if len(groups) == 1:
        return f"{base} of {groups[0][0]}"
    if df.algorithm.nunique() == 1:
        return f"{base} of {df.algorithm.iloc[0]} (different runs)"
    return f"{base} of different algorithms"

def _algorithm_tag(paths):
    """Return algorithm name(s) for use as filename prefix."""
    df = _read(paths)
    if df.empty: return
    algorithms = (df["algorithm"].dropna().astype(str)
                  .str.strip().unique())
    if len(algorithms) == 0: return "unknown"
    return "_".join(algorithms)

def _apply_layout(fig,
                  *,
                  with_legend=False,
                  nrows=1,
                  ncols=1,
                  extra_left_cm=0.0,
                  extra_right_cm=0.0):
    figW, figH  = fig.get_size_inches()
    scale_w     = DEF_figWidth / figW
    scale_h     = DEF_figHeight / figH
    left        = (DEF_axisLeft * scale_w
                   + extra_left_cm / DEF_cmTOinch / figW)
    bottom      = DEF_axisBottom * scale_h
    top         = 1 - (1 - DEF_axisTop) * scale_h
    x_legend    = 1 - (1 - DEF_figLegendx) * scale_w
    gap_w       = DEF_axisGapW * scale_w
    gap_h       = DEF_axisGapH * scale_h
    if with_legend: right = x_legend - gap_w
    else:           right = 1 - (1 - DEF_axisRight) * scale_w
    right -= extra_right_cm / DEF_cmTOinch / figW
    space_h = DEF_axisSpaceH * scale_h
    space_w = DEF_axisSpaceW * scale_w
    axis_height = (top - bottom - (nrows - 1) * space_h) / nrows
    axis_width  = (right - left - (ncols - 1) * space_w) / ncols
    if axis_height <= 0 or axis_width <= 0:
        raise ValueError("Figure is too small.")
    fig.subplots_adjust(left    = left,
                        right   = right,
                        bottom  = bottom,
                        top     = top,
                        hspace  = space_h / axis_height if nrows > 1 else 0,
                        wspace  = space_w / axis_width  if ncols > 1 else 0,
                        )
    return x_legend, top, gap_h

def _legend_layout(fig,
                   sections,
                   *,
                   nrows=1,
                   ncols=1,
                   extra_left_cm=0.0,
                   extra_right_cm=0.0,
                   legend_only=False):
    sections = [(title, handles)
                for title, handles in sections
                if handles]
    if legend_only:
        x_legend    = 0.05
        top         = 0.95
        gap_h       = 0.3 / (fig.get_figheight() * DEF_cmTOinch)

    else:
        x_legend, top, gap_h = _apply_layout(fig,
                                             with_legend=bool(sections),
                                             nrows=nrows,
                                             ncols=ncols,
                                             extra_left_cm=extra_left_cm,
                                             extra_right_cm=extra_right_cm
                                             )
    y = top
    for title, handles in sections:
        legend = fig.legend(handles=handles,
                            title=title,
                            loc="upper left",
                            bbox_to_anchor=(x_legend, y),
                            bbox_transform=fig.transFigure,
                            borderaxespad=0,
                            framealpha=1,
                            facecolor="white",
                            edgecolor="black",
                            fontsize=7,
                            title_fontsize=7,
                            )
        fig.canvas.draw()
        box = (legend.get_window_extent(fig.canvas.get_renderer())
               .transformed(fig.transFigure.inverted()))
        y = box.y0 - gap_h
    return

def _suptitle(fig, text):
    fig.suptitle(text,
                 x  = 0.5,
                 y  = 1 - DEF_axisGapH * DEF_figHeight / fig.get_figheight(),
                 va = "top",
                 )
    return

def _cost_scale(cost_step):
    if not np.isfinite(cost_step) or cost_step <= 0:
        return 1.0, 0
    exponent = int(3 * np.floor(np.log10(cost_step) / 3))
    return 10.0 ** exponent, exponent

def _nice_step(value):
    """61 -> 60; 5.1 -> 6; 0.36 -> 0.4."""
    if not np.isfinite(value) or value <= 0:
        return 1.0
    magnitude = 10.0 ** np.floor(np.log10(value))
    leading = value / magnitude
    rounded = np.ceil(leading) if value < 10 else np.floor(leading + 0.5)
    return rounded * magnitude

###############################################################################
#---------------------------------- LANDSCAPE ---------------------------------
###############################################################################

def _landscape_grid(df,
                    x,
                    y,
                    field,
                    resolution=90):
    from scipy.interpolate import CloughTocher2DInterpolator
    from scipy.spatial import QhullError
    xx = df[x].to_numpy(dtype=float)
    yy = df[y].to_numpy(dtype=float)
    zz = df[field].to_numpy(dtype=float)
    span_x, span_y = np.ptp(xx), np.ptp(yy)
    if (len(df) < 3 or
        span_x == 0 or
        span_y == 0):
        return None
    coords = np.column_stack(((xx - xx.min()) / span_x,
                              (yy - yy.min()) / span_y))
    try:
        gx = np.linspace(xx.min(), xx.max(), resolution)
        gy = np.linspace(yy.min(), yy.max(), resolution)
        X, Y = np.meshgrid(gx, gy)
        interpolator = CloughTocher2DInterpolator(coords, zz)
        Z = interpolator((X - xx.min()) / span_x,
                         (Y - yy.min()) / span_y)
    except (QhullError, ValueError): return None
    if not np.isfinite(Z).any(): return None
    # A sparse cubic fit may overshoot measured extrema; keep the display in
    # the observed response range, and leave the area outside the hull blank.
    Z = np.where(np.isfinite(Z),
                 np.clip(Z, zz.min(), zz.max()),
                 np.nan)
    return X, Y, Z

def plot_landscape(path,
                   variant  = "both",
                   x        = None,
                   y        = None,
                   method   = None,
                   value    = "cost"):
    """
    2D contours and 3D surfaces of cost/objective and settlement.
    The smooth field is interpolated, not simulated; it is confined to the
    convex hull of observed designs. Markers show actual PLAXIS evaluations.
    The historical variant='bars' now requests the 2D contour figure.
    """
    if variant not in ("bars", "surface", "both", "feasible"):
        raise ValueError("variant must be bars, surface, both, or feasible")
    if value not in ("cost", "objective"):
        raise ValueError("value must be 'cost' or 'objective'")
    df = _read(path)
    
    algorithm_tag = _algorithm_tag(path)
    df = df.loc[df.status.astype(str).str.strip().eq("OK")].copy()
    if method is not None:
        df = df[df.si_method == method]
    if df.empty:
        return []
    # find params
    if x is None and y is None:
        varying = [p for p in PARAMS
                   if (p in df.columns and
                       df[p].nunique() > 1)
                   ]
        if len(varying) < 2:
            return []
        from itertools import combinations
        xy_combinations = list(combinations(varying, 2))
    elif x is not None and y is not None:
        xy_combinations = [(x, y)]
    else:
        raise ValueError("x and y must either both be given or both be None")
    outputs = []
    for x, y in xy_combinations:
        data = df.copy()
        for col in (x, y):
            _, categories = _prepare_plot_axis(data, col)
            if categories is not None:
                print(f"Design landscape benötigt numerische Achsen; "
                      f"'{col}' ist kategorial. Nutze plot_evolution_parameter "
                      "oder plot_best_feasible_points dafür.")
                return
        for col in (x, y, value, "u"):
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(subset=[x, y, value, "u"])
        data = data[np.isfinite(data[[x, y, value, "u"]]).all(axis=1)]
        if data.empty:
            continue
        # Best feasible solution (minimum cost)
        feasible = data.loc[data["feasible"].astype(bool)
                            & np.isfinite(pd.to_numeric(data["cost"],
                                                        errors="coerce"))
                            ]
        best = (feasible.loc[pd.to_numeric(feasible["cost"],
                                           errors="coerce").idxmin()
                             ]
                if not feasible.empty
                else None)
        # A landscape is meaningful when the other design variables are fixed.
        # Resolve repeated coordinates consistently without inventing observations.
        data = data.sort_values(["feasible", value],
                                ascending=[False, True])
        data = data.drop_duplicates([x, y])
        colors = fct_setPlotstyle(case=0, flat=True, r=1)[1]
        cmap = LinearSegmentedColormap.from_list("study_style", colors[:5])
        cmap.set_bad((1, 1, 1, 0))
        outputs = []
        cost_scale = 1.0
        cost_exponent = 0
        if value == "cost":
            cost_step = (data["cost"].max() - data["cost"].min()) / 5
            cost_scale, cost_exponent = _cost_scale(cost_step)
        for kind in (["bars", "surface"]
                         if variant == "both"
                         else [variant]):
            is_3d = kind == "surface"
            is_feasible = kind == "feasible"
            fig = plt.figure(figsize = (DEF_figWidth,
                                        DEF_figHeight)
                             )
            if is_feasible:
                fields = (("cost", "Cost", colors[0]),)
            else:
                fields = ((value,
                           "Objective"
                           if value == "objective"
                           else "Cost",
                           colors[0]),
                          ("u",
                           "Settlement",
                           colors[1])
                          )
            for i, (field, title, base) in enumerate(fields):
                ax = fig.add_subplot(1,
                                     2,
                                     i + 1,
                                     projection=("3d"
                                                  if is_3d
                                                  else None)
                                     )
                xx, yy, zz = [data[k].to_numpy(dtype=float)
                              for k in (x, y, field)]
                surface = _landscape_grid(data, x, y, field)
                if is_3d:
                    zz_plot = (zz / cost_scale
                               if field == "cost"
                               else zz)
                    if surface is not None:
                        X, Y, Z = surface
                        Z_plot = (Z / cost_scale
                                  if field == "cost"
                                  else Z)
                        ax.plot_surface(X,
                                        Y,
                                        np.ma.masked_invalid(Z_plot),
                                        cmap=cmap,
                                        linewidth=0,
                                        antialiased=True,
                                        alpha=0.86,
                                        vmin=zz_plot.min(),
                                        vmax=zz_plot.max()
                                        )
                    # All evaluated designs
                    ax.scatter(xx,
                               yy,
                               zz_plot,
                               color="black",
                               s=12,
                               depthshade=False,
                               zorder=4)
                    # Best feasible solution
                    if best is not None:
                        best_z = (best[field] / cost_scale
                                  if field == "cost"
                                  else best[field])
                        ax.scatter([best[x]],
                                   [best[y]],
                                   [best_z],
                                   color="red",
                                   s=45,
                                   depthshade=False,
                                   edgecolors="black",
                                   linewidths=0.4,
                                   zorder=10
                                   )
                    zlabel = LABELS.get(field, field)
                    if field == "cost" and cost_exponent != 0:
                        zlabel = rf"Cost [$€ \times 10^{{{cost_exponent}}}$]"
                    ax.set_zlabel(zlabel, labelpad=2)
                    ax.view_init(elev=26, azim=-60)
                    ax.tick_params(labelsize=6, pad=0)
                else: # 2d
                    if surface is not None:
                        X, Y, Z = surface
                        Z_plot = (Z / cost_scale
                                  if field == "cost"
                                  else Z)
                        zz_plot = (zz / cost_scale
                                   if field == "cost"
                                   else zz)
                        im = ax.contourf(X,
                                         Y,
                                         np.ma.masked_invalid(Z_plot),
                                         levels=18,
                                         cmap=cmap,
                                         vmin=zz_plot.min(),
                                         vmax=zz_plot.max()
                                         )
                        cbar_label = LABELS.get(field, field)
                        if field == "cost" and cost_exponent != 0:
                            cbar_label = cbar_label.replace("[€]",
                                rf"[$€ \times 10^{{{cost_exponent}}}$]")
                        fig.colorbar(im,
                                     ax=ax,
                                     shrink=0.83,
                                     pad=0.03,
                                     label=cbar_label)
                    # All evaluated designs
                    ax.scatter(xx,
                               yy,
                               color="black",
                               s=12,
                               zorder=4)
                    # Best feasible solution
                    if best is not None:
                        ax.scatter(best[x],
                                   best[y],
                                   color="red",
                                   s=45,
                                   edgecolors="black",
                                   linewidths=0.4,
                                   zorder=10
                                   )
                    ax.tick_params(labelsize=6)
                ax.set(xlabel=LABELS.get(x, x),
                       ylabel=LABELS.get(y, y),
                       title=title)
                if is_3d:
                    ax.xaxis.labelpad = -2
                    ax.yaxis.labelpad = -2
            _suptitle(fig,
                      "Interpolated design landscape"
                          if not is_feasible
                          else "Cost landscape with feasible designs")
            _apply_layout(fig, nrows=1, ncols=2)
            suffix = f"{x}_{y}"
            outputs.append(fct_formatSave(True,
                                          fig,
                                          os.path.normpath(path) + "_plot",
                                          f"{algorithm_tag}"
                                          f"_optimization_landscape_{suffix}"
                                          f"_{kind}",
                                          pad = (0.28 if is_3d else None)
                                          ))
    return outputs

###############################################################################
#-------------------------------- CONVERGENCE ---------------------------------
###############################################################################

def plot_convergence(paths,
                     save_path=None,
                     metrics=("cost", "constraints"),
                     best=True,
                     group_by="algorithm",
                     reference_path=None,
                     ):
    """
    Cost and constraint violation by evaluation, colour per algorithm/run.
    Two legends: line style describes metric; colour describes algorithm/run.
    Costs use the left axis, dimensionless normalized violation the right axis.
    reference_path points to a completed grid history (folder or CSV). Its
    minimum successful feasible cost is drawn as a horizontal line.
    """
    df = _read(paths)
    if df.empty: return
    algorithm_tag = _algorithm_tag(paths)
    if save_path is None:
        save_path = (paths
                     if isinstance(paths, (str, os.PathLike))
                     else os.path.dirname(str(paths[0])))
    if isinstance(metrics, str):
        metrics = (metrics,)
    metrics = tuple(metrics)
    if not metrics or set(metrics) - {"cost", "constraints"}:
        raise ValueError("metrics must contain 'cost', 'constraints', or both")
    show_cost = "cost" in metrics
    show_constraints = "constraints" in metrics
    colors = fct_setPlotstyle(case=0, flat=True, r=1)[1]
    fig, ax = plt.subplots(figsize=(DEF_figWidth,
                                    DEF_figHeight
                                    ))
    ax2 = ax.twinx() if show_cost and show_constraints else None
    cv_ax = ax2 if ax2 is not None else ax
    color_legend = []
    groups = (_study_groups(df)
              if group_by == "algorithm"
              else list(df.groupby(group_by, sort=False))
              )
    valid_cost_values = []
    valid_cv_values = []
    for i, (name, group) in enumerate(groups):
        color = colors[i % len(colors)]
        group = group.sort_values("evaluation")
        x = pd.to_numeric(group["evaluation"], errors="coerce").to_numpy(dtype=float)
        valid = group["status"].astype(str).str.strip().eq("OK").to_numpy()
        cost = pd.to_numeric(group["cost"], errors="coerce").to_numpy(dtype=float)
        good = valid & np.isfinite(cost) & (cost < 1e11) & np.isfinite(x)

        if show_cost:
            if np.any(good):
                valid_cost_values.extend(cost[good])
                ax.plot(x, np.where(good, cost, np.nan),
                        color=color, lw=0.8, alpha=0.3, linestyle="-")
            else:
                print(f"Warning: No valid cost values for {name}")
            if best:
                feasible = group["feasible"].to_numpy(dtype=bool) & good
                running = (pd.Series(np.where(feasible, cost, np.inf))
                           .cummin().replace(np.inf, np.nan).to_numpy())
                ax.plot(x, running, color=color, lw=1.8, linestyle="-")

        if show_constraints:
            cv = (pd.to_numeric(group["cv_norm"], errors="coerce")
                  .to_numpy(dtype=float)
                  if "cv_norm" in group
                  else np.full(len(group), np.nan))
            good_cv = valid & np.isfinite(cv) & np.isfinite(x)
            if np.any(good_cv):
                valid_cv_values.extend(cv[good_cv])
            cv_ax.plot(x, np.where(good_cv, cv, np.nan),
                       color=color, lw=1.1, linestyle="--")

        color_legend.append(Line2D([], [], color=color, lw=2,
                                   label=str(name).split(" (", 1)[0]))
    reference_handle = None
    if reference_path is not None:
        if not show_cost:
            raise ValueError("The grid reference requires 'cost' in metrics")
        if not os.path.isabs(reference_path):
            first_path = (paths
                          if isinstance(paths, (str, os.PathLike))
                          else paths[0]
                          )
            base_path = (os.path.dirname(str(first_path))
                         if str(first_path).lower().endswith(".csv")
                         else os.path.dirname(os.path.normpath(str(first_path)))
                         )
            reference_path = os.path.join(base_path, reference_path)
        reference = _read(reference_path)
        if not reference.algorithm.astype(str).str.lower().eq("grid").all():
            raise ValueError("reference_path must contain a grid run")
        reference = (reference.loc[reference.status.astype(str)
                                   .str.strip().eq("OK")].copy()
                     )
        reference = reference.loc[reference.feasible].copy()
        reference["cost"] = pd.to_numeric(reference.cost,
                                          errors="coerce")
        reference = reference.loc[np.isfinite(reference.cost)]
        if reference.empty:
            raise ValueError("Grid history has no successful feasible cost")
        reference_cost = float(reference.cost.min())
        ax.axhline(reference_cost,
                   color="black",
                   linestyle=":",
                   lw=1.5,
                   zorder=3
                   )
        reference_handle = Line2D([],
                                  [],
                                  color         ="black",
                                  linestyle     =":",
                                  lw            =1.5,
                                  label         =("Best feasible grid"
                                                  # f": {reference_cost:,.2f} €"
                                                  )
                                  )
    ax.set_xlabel("Iteration [-]")
    if show_cost:
        ax.set_ylabel("Cost [€]")
        if valid_cost_values:
            cost_min = float(np.min(valid_cost_values))
            cost_max = float(np.max(valid_cost_values))
            cost_range = cost_max - cost_min
            cost_step = (_nice_step(cost_range / 5)
                         if cost_range > 0
                         else max(abs(cost_max) * 0.1, 1.0))
            cost_unit = 10.0 ** np.floor(np.log10(cost_step))
            cost_start = np.floor(cost_min / cost_unit) * cost_unit
            cost_intervals = max(5, int(np.ceil((cost_max - cost_start) / cost_step)))
        else:
            print("Warning: No valid costs found for convergence plot.")
            cost_step, cost_start, cost_intervals = 1.0, 0.0, 5
    else:
        cost_intervals = 0
    if show_constraints:
        cv_ax.set_ylabel("Normalised constraint\nviolation [-]")
        cv_max = max(0.0, float(np.max(valid_cv_values))) if valid_cv_values else 0.0
        cv_step = _nice_step(cv_max / 5)
        cv_intervals = int(np.ceil(cv_max / cv_step))
    else:
        cv_intervals = 0
    intervals = max(5, cost_intervals, cv_intervals)
    if show_cost:
        ax.set_yticks(cost_start + np.arange(intervals + 1) * cost_step)
        ax.set_ylim(cost_start,
                    cost_start + (intervals + 0.15) * cost_step
                    )
        scale, exponent = _cost_scale(cost_step)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value / scale:g}")
            )
        if exponent != 0:
            ax.text(0,
                    1.02,
                    rf"$\times 10^{{{exponent}}}$",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom"
                    )
    if show_constraints:
        cv_ax.set_yticks(np.arange(intervals + 1) * cv_step)
        cv_ax.set_ylim(0,
                       (intervals + 0.15) * cv_step
                       )
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", alpha=0.20)
    style = []
    if show_cost:
        style.append(Line2D([],
                            [],
                            color="black",
                            lw=1.8,
                            label=("Best feasible cost"
                                   if best
                                   else "Cost")
                            ))
        if best:
            style.append(Line2D([],
                                [],
                                color="black",
                                lw=0.8,
                                alpha=0.4,
                                label="Evaluated cost"
                                ))
    if show_constraints:
        style.append(Line2D([],
                            [],
                            color="black",
                            linestyle="--",
                            label="Constraint residuals"
                            ))
    if reference_handle is not None:
        style.append(reference_handle)
    one = len(groups) == 1
    _suptitle(fig, _run_title("Convergence", df, groups))
    _legend_layout(fig,
                   (([]
                     if one
                     else [("Algorithms", color_legend)])
                    + [("Quantities", style)]),
                   extra_right_cm = (1.5
                                     if ax2 is not None
                                     else 0.0),
                   )
    os.makedirs(save_path, exist_ok=True)
    metric_suffix = ("both"
                     if show_cost and show_constraints
                     else ("cost"
                           if show_cost
                           else "cons")
                     )
    existing_versions = []
    for filename in os.listdir(save_path):
        match = re.fullmatch(rf"(?:Plot_)?{re.escape(algorithm_tag)}_"
                             rf"optimization_convergence_v(\d+)"
                             rf"(?:_(?:cost|cons|both))?"
                             rf"\.(?:pdf|png|svg)",
                             filename,
                             flags=re.IGNORECASE
                             )
        if match:
            existing_versions.append(int(match.group(1)))
    version = 1 + max(existing_versions, default=0)
    image_path = fct_formatSave(
        True,
        fig,
        os.path.normpath(save_path) + "_plot",
        f"{algorithm_tag}_optimization_convergence_v{version}_{metric_suffix}"
        )
    if isinstance(paths, (str, os.PathLike)):
        input_paths = [paths]
    else:
        input_paths = paths
    documentation = os.path.join(os.path.normpath(save_path)
                                 + "_plot",
                                 "optimization_convergence_files.txt"
                                 )
    with open(documentation, "a", encoding="utf-8") as file:
        file.write(f"\nVersion v{version}: {image_path}\n")
        file.write("Plotted histories:\n")
        for path in input_paths:
            history = (str(path)
                       if str(path).lower().endswith(".csv")
                       else os.path.join(path, "optimization_history.csv")
                       )
            file.write(f"  {os.path.abspath(history)}\n")
        if reference_path is not None:
            reference_file = (str(reference_path)
                              if str(reference_path).lower().endswith(".csv")
                              else os.path.join(
                                  reference_path,
                                  "optimization_history.csv"
                                  )
                              )
            file.write(
                f"Grid reference: {os.path.abspath(reference_file)}\n"
                )
    return image_path

###############################################################################
#------------------------ COMPARE BEST FEASIBLE DESIGN ------------------------
###############################################################################

def plot_best_feasible(paths,
                       save_path,
                       params=None,
                       labels=None,
                       filename="optimization_best_feasible",
                       table_filename="optimization_best_feasible"):
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    else:
        paths = list(paths)
    algorithm_tag = _algorithm_tag(paths)
    dfs = [_read(path) for path in paths]
    present_dfs = [df for df in dfs if not df.empty]
    if not present_dfs:
        raise ValueError("Keine gültigen Optimierungsdaten gefunden.")
    all_runs = pd.concat(present_dfs, ignore_index=True)
    if params is None:
        # Only varying parameters need separate comparison plots.
        params = [p for p in PARAMS
                  if p in all_runs.columns and all_runs[p].nunique() > 1]
    params = list(params)
    # The _all LaTeX table contains ALL available design parameters,
    # including constant si_num and string-based si_method/raster.
    paramsTOT = list(dict.fromkeys(
        [p for p in PARAMS
         if p in all_runs.columns and all_runs[p].notna().any()]
        + [p for p in params if p != "cost"]
    ))
    # Maximum two additional parameters per plot because
    # cost occupies the first y-axis.
    groups = ([params]
              + [[p] for p in params]
              + [params[i:i + 2] for i in range(0, len(params), 2)])
    results = []
    for i, group in enumerate(groups, start=1):
        # Keep original filename for the first plot.
        # Add an index for additional plots.
        if len(groups) == 1:
            current_filename = filename
            current_table_filename = table_filename
        else:
            root, ext = os.path.splitext(filename)
            current_filename = f"{root}_{i}{ext}"
            root, ext = os.path.splitext(table_filename)
            current_table_filename = f"{root}_{i}{ext}"
        current_filename = f"{algorithm_tag}_{current_filename}"
        current_table_filename = f"{algorithm_tag}_{current_table_filename}"
        result = _best_feasible(paths,
                                save_path,
                                current_filename,
                                current_table_filename,
                                params=group,
                                labels=labels
                                )
        results.append(result)
    result = _best_feasible(paths,
                            save_path,
                            f"{algorithm_tag}_{filename}_all",
                            f"{algorithm_tag}_{table_filename}_all",
                            params=paramsTOT,
                            labels=labels,
                            )
    results.append(result)
    return results, groups

# Shared publication labels for both best-feasible plots.
BEST_FEASIBLE_INFO = {
    "cost": (r"$C$", "Costs [€]", "€"),
    "slab_t": (r"$t_{\mathrm{slab}}$", r"Slab thickness $t_{\mathrm{slab}}$ [m]", "m"),
    "ltp_t": (r"$t_{\mathrm{LTP}}$", r"LTP thickness $t_{\mathrm{LTP}}$ [m]", "m"),
    "si_num": (r"$a_x$", r"Piles per row $a_x$ [-]", ""),
    "si_L": (r"$L$", "SI length $L$ [m]", "m"),
    "si_R": (r"$r_{\mathrm{SI}}$", r"SI radius $r_{\mathrm{SI}}$ [m]", "m"),
    "si_E": (r"$E_{\mathrm{SI}}$", r"SI stiffness $E_{\mathrm{SI}}$ [kPa]", "kPa"),
    "si_method": (r"$M$", r"SI method $M$ [-]", ""),
}

def _param_label(param):
    if param in BEST_FEASIBLE_INFO:
        return BEST_FEASIBLE_INFO[param][1]
    name, unit, _ = PARAM_INFO.get(param, (param, "", param))
    return f"{name} [{unit}]" if unit else name

def _legend_label(param):
    return _param_label(param)

def _bar_value_text(param, original):
    symbol, _, unit = BEST_FEASIBLE_INFO.get(param, (param, param, ""))
    try:
        value = float(original)
        number = (f"{value:,.0f}".replace(",", " ") if param == "cost"
                  else f"{value:.2f}" if unit == "m" else f"{value:g}")
    except (ValueError, TypeError):
        number = str(original)
    return f"{symbol} = {number}" + (f" {unit}" if unit else "")

def _best_seed(path):
    # Supports ...seed1_GA, ...seed2_DE, ..._seed3, and seed-4.
    pathstr = str(path).replace("\\", "/")
    match = re.search(r"seed[_-]?(\d+)(?=[/_-]|$)", pathstr, re.I)
    return int(match.group(1)) if match else None

def _run_sort_key(row):
    name = str(row["algorithm"])
    preferred = {"GA": 0, "DE": 1, "ISRES": 2, "MVGA": 3, "PSO": 4}
    return (preferred.get(name.upper(), 100), name.upper(),
            float("inf") if pd.isna(row["seed"]) else int(row["seed"]),
            int(row["_order"]))

def _axis_limits(values, n_intervals=4, quantum=0.01, padding=0.45):
    """Exactly five aligned ticks, using rounded steps and enclosing the data."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        values = np.array([0.0, 1.0])
    lo, hi = float(values.min()), float(values.max())
    span = max(hi - lo, abs(hi) * 0.1, quantum)
    hi += padding * span
    required = max((hi - lo) / n_intervals, quantum)
    power = 10.0 ** np.floor(np.log10(required))
    for _ in range(30):
        for leading in (1, 2, 2.5, 5, 10):
            step = max(quantum, np.ceil(leading * power / quantum - 1e-10) * quantum)
            lower = np.floor(lo / step + 1e-10) * step
            upper = lower + n_intervals * step
            if step >= required * (1 - 1e-10) and upper >= hi - quantum * 1e-8:
                return lower, upper, lower + np.arange(n_intervals + 1) * step
        power *= 10
    raise ValueError("Unable to construct finite axis limits")


def _format_best_axis(axis, param, values, padding=0.45):
    """Format raw coordinates; scaling affects tick labels only."""
    from matplotlib.transforms import ScaledTranslation
    unit = BEST_FEASIBLE_INFO.get(param, (None, None, ""))[2]
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    magnitude = float(np.max(np.abs(finite))) if len(finite) else 0.0
    exponent = (int(3 * np.floor(np.log10(magnitude) / 3))
                if magnitude >= 1000 and unit != "m" else 0)
    scale = 10.0 ** exponent
    decimals = 2 if unit == "m" else 0
    quantum = 10.0 ** (-decimals)
    lo, hi, ticks = _axis_limits(finite / scale, quantum=quantum, padding=padding)
    axis.set_ylim(lo * scale, hi * scale)
    axis.set_yticks(ticks * scale)
    axis.yaxis.set_major_formatter(FuncFormatter(
        lambda value, _, scale=scale, decimals=decimals: f"{value / scale:.{decimals}f}"))
    axis.yaxis.offsetText.set_visible(False)
    old = getattr(axis, "_engineering_text", None)
    if old is not None:
        old.remove()
    axis._engineering_text = None
    if exponent:
        side = axis.yaxis.get_ticks_position()
        right = side == "right"
        position = axis.spines["right" if right else "left"].get_position()
        offset = position[1] if position[0] == "outward" else 0
        transform = axis.transAxes + ScaledTranslation(
            (offset if right else -offset) / 72, 0,
            axis.figure.dpi_scale_trans)
        axis._engineering_text = axis.text(
            1 if right else 0, 1.02, rf"$\times 10^{{{exponent}}}$",
            transform=transform, ha="left", va="bottom", clip_on=False)


def _best_feasible(paths,
                   save_path,
                   filename,
                   table_filename,
                   params=("si_method",
                           "si_num",
                           "si_L",
                           "si_R",
                           "ltp_t",
                           "slab_t"),
                   labels=None):
    """
    Plot and LaTeX table of the best feasible design for each optimisation run.

    The plot groups optimisation runs by algorithm. Within each algorithm,
    individual bars represent different parameter variations.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import FuncFormatter
    paths = ([paths]
             if isinstance(paths, (str, os.PathLike))
             else list(paths)
             )
    if not paths:
        return None, None, None
    if save_path is None:
        first_path = paths[0]
        save_path = (first_path
                     if isinstance(first_path, (str, os.PathLike))
                     else os.path.dirname(str(first_path))
                     )
    save_path = os.path.normpath(save_path)
    os.makedirs(save_path, exist_ok=True)
    if labels is not None and len(labels) != len(paths):
        raise ValueError("labels must have the same length as paths")
    results = []
    for i, path in enumerate(paths):
        df = _read(path)
        # print(f"PATH: {path}")
        # print(f"COLUMNS: {df.columns.tolist()}")
        if df.empty: return
        history = df.copy()
        df = df.loc[df.status.astype(str).str.strip().eq("OK")].copy()
        df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
        df = df.loc[df.feasible.astype(bool)
                    & np.isfinite(df["cost"])].copy()
        if df.empty:
            raise ValueError(f"No successful feasible design found in: {path}")
        best = df.loc[df["cost"].idxmin()]
        run_history = history.loc[history["algorithm"].eq(best["algorithm"])]
        counters = {"max_evaluation": np.nan,
                    "max_iteration": np.nan}
        counter_columns = {"max_evaluation": ("evaluation",),
                           "max_iteration": ("iteration",
                                             "generation",
                                             "n_gen",
                                             "n_iter",
                                             ),
                           }
        for target, candidates in counter_columns.items():
            for column in candidates:
                if column not in run_history: continue
                values = pd.to_numeric(run_history[column], errors="coerce")
                values = values[np.isfinite(values)]
                if not values.empty:
                    counters[target] = float(values.max())
                    break
        if labels is not None:          run_label = re.sub("mixedvariablega",
                                                           "MVGA",
                                                           str(labels[i]),
                                                           flags=re.IGNORECASE
                                                           )
        elif "algorithm" in best.index: run_label = str(best["algorithm"])
        else:                           run_label = f"Run {i + 1}"
        result = {"run": run_label, "algorithm": str(best["algorithm"]),
                  "seed": _best_seed(path), "_order": i,
                  "cost": float(best["cost"])}
        for param in params:
            result[param] = (best[param]
                             if param in best.index
                             else np.nan
                             )
        result.update(counters)
        results.append(result)
    best_df = pd.DataFrame(results)
    best_df = best_df.iloc[sorted(range(len(best_df)),
        key=lambda i: _run_sort_key(best_df.iloc[i]))].reset_index(drop=True)
    repeated = best_df["algorithm"].duplicated(keep=False).any()
    best_df["display_run"] = [
        (f"{row['algorithm']} - {int(row['seed'])}"
         if pd.notna(row['seed']) else
         (f"{row['algorithm']} - {i+1}" if repeated else str(row['algorithm'])))
        for i, (_, row) in enumerate(best_df.iterrows())]
    plot_params = ["cost"] + [param
                              for param in params
                              if param in best_df.columns
                              ]
    # if len(plot_params) > 3:
    #     raise ValueError("At most two parameters can be passed in `params`, "
    #                      "because cost uses the first of the three y-axes."
    #                      )
    n_bars = len(plot_params)
    algorithm_col = ("algorithm"
                     if "algorithm" in best_df.columns
                     else "run")
    algorithms = best_df["display_run"].astype(str).tolist()
    n_algorithms = len(algorithms)
    # fig, ax_cost = plt.subplots(figsize=(DEF_figWidth,
    #                                      DEF_figHeight)
    #                             )
    # Keep a readable plotting area and reserve physical space on the right
    # for every y-axis. Figure widths/heights are in inches.
    plot_width      = max(DEF_figWidth * 0.80,
                          n_algorithms * n_bars * 0.20)
    axis_gap        = max(DEF_extraAxis_cm / DEF_cmTOinch, 0.65)
    left_margin     = 0.35
    right_margin    = 0.90 + (n_bars - 1) * axis_gap
    fig_width       = left_margin + plot_width + right_margin
    fig, ax_cost    = plt.subplots(figsize=(fig_width, DEF_figHeight))
    fig.subplots_adjust(left    = left_margin / fig_width,
                        right   = (left_margin + plot_width) / fig_width,
                        bottom  = 0.24,
                        top     = 0.94)

    axes = [ax_cost]
    for axis_index in range(1, n_bars):
        axis = ax_cost if axis_index == 0 else ax_cost.twinx()
        axis.spines["left"].set_visible(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_position(("outward",
                                           axis_index * axis_gap * 72))
        axis.spines["right"].set_visible(True)
        axis.yaxis.set_ticks_position("right")
        axis.yaxis.set_label_position("right")
        axis.patch.set_visible(False)
        if axis_index:
            axes.append(axis)
    # twinx() can reset the host ticks to the left; set all sides afterwards.
    for axis in axes:
        axis.yaxis.set_ticks_position("right")
        axis.yaxis.set_label_position("right")
        axis.tick_params(axis       = "y",
                         left       = False,
                         right      = True,
                         labelleft  = False,
                         labelright = True)

    plot_values = {}
    category_names = {}
    for param in plot_params:
        encoded, names = _prepare_plot_axis(best_df, param)
        # Use 1-based category positions for bars (zero-height bars are hidden).
        plot_values[param] = encoded + (1.0 if names is not None else 0.0)
        if names is not None:
            category_names[param] = names
    values = {param: [] for param in plot_params}
    rows_by_algorithm = {}
    for _, row in best_df.iterrows():
        rows_by_algorithm[str(row["display_run"])] = row
        for param in plot_params:
            if param in row.index and pd.notna(row[param]):
                values[param].append(float(plot_values[param].loc[row.name]))
    for param, axis in zip(plot_params, axes):
        if not values[param]:
            continue
        if param in category_names:
            names = category_names[param]
            axis.set_ylim(0.0, len(names) + 0.8)
            axis.set_yticks([])
            axis.spines["right"].set_visible(False)
        else:
            _format_best_axis(axis, param, values[param])
    for axis, param in zip(axes, plot_params):
        axis.set_ylabel("" if param in category_names else _param_label(param),
                        rotation    = 90,
                        labelpad    = 4,
                        fontsize    = ax_cost.xaxis.label.get_size(),
                        )
    colors = fct_setPlotstyle(case=0,
                              flat=True,
                              r=1)[1] or ["0.35"]
    alphas = ([1.0]
              if n_bars == 1
              else np.linspace(1.0,
                               0.4,
                               n_bars)
              )
    group_width = 0.68
    bar_width = group_width / n_bars
    x = np.arange(n_algorithms, dtype=float)
    offsets = ((np.arange(n_bars) - (n_bars - 1) / 2) * bar_width)
    bars = []
    color_algorithms = list(dict.fromkeys(best_df["algorithm"].astype(str)))
    for i, algorithm in enumerate(algorithms):
        row = rows_by_algorithm.get(algorithm)
        if row is None: continue
        algorithm_color = colors[color_algorithms.index(str(row["algorithm"])) % len(colors)]
        for j, (param, axis) in enumerate(zip(plot_params, axes)):
            if param not in row.index or pd.isna(row[param]):
                continue
            value = float(plot_values[param].loc[row.name])
            bar = axis.bar(x[i] + offsets[j],
                           value,
                           width     = bar_width * DEF_bestBarWidthFactor,
                           color     = algorithm_color,
                           alpha     = alphas[j],
                           edgecolor = "black",
                           linewidth = 0.4,
                           zorder    = 1
                           )[0]
            bar.set_edgecolor((0, 0, 0, 1))
            bars.append((bar, value, row[param], param, axis))
    value_labels = []
    for bar, value, original, param, axis in bars:
        text = _bar_value_text(param, original)
        annotation = axis.annotate(text,
                                   xy           = (bar.get_x() +
                                                   bar.get_width() / 2,
                                                   max(value, 0.0)
                                                   ),
                                   xytext       = (0, 5),
                                   textcoords   = "offset points",
                                   rotation     = 90,
                                   ha           = "center",
                                   va           = "bottom",
                                   fontsize     = 6,
                                   clip_on      = False,
                                   annotation_clip= False,
                                   zorder       = 20,
                                   )
        value_labels.append((annotation, axis))
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels(algorithms)
    ax_cost.set_xlabel("Optimisation algorithm – seed" if best_df.seed.notna().any()
                       else "Optimisation algorithm")
    ax_cost.set_xlim(-0.55, n_algorithms - 0.45)
    # Draw the grid only once: all axes have aligned tick positions.
    ax_cost.set_axisbelow(True)
    ax_cost.grid(axis="y", alpha=0.20, zorder=0)
    for axis in axes[1:]: axis.grid(False)
    for axis in axes[1:]: axis.spines["bottom"].set_visible(False)
    for axis in axes:
        for spine in axis.spines.values():
            if spine.get_visible():
                spine.set_zorder(10)
    legend_handles = [mpatches.Patch(facecolor="0.35",
                                     edgecolor="black",
                                     alpha=alpha,
                                     label=_legend_label(param)
                                     )
                      for param, alpha in zip(plot_params, alphas)
                      ]
    ax_cost.legend(handles=legend_handles,
                   title="Parameter",
                   loc="upper center",
                   bbox_to_anchor=(0.5, -0.18),
                   ncol=min(n_bars, 6),
                   frameon=False,
                   fontsize=8)

    # Measure the rotated text at the final layout. If necessary, increase
    # the upper limits of ALL axes by the same factor to retain grid alignment.
    for _ in range(8):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        factor = 1.0
        for annotation, axis in value_labels:
            box = annotation.get_window_extent(renderer)
            y0, y1 = axis.get_ylim()
            value = annotation.xy[1]
            room = axis.bbox.height - box.height - 18.0
            if room <= 0:
                raise ValueError("Figure height is too "
                                 "small for the bar labels.")
            required_span = max(0.0, value - y0) * axis.bbox.height / room
            factor = max(factor, required_span / (y1 - y0))
        if factor <= 1.0:
            break
        factor *= 1.03
        for axis, param in zip(axes, plot_params):
            lower, upper = axis.get_ylim()
            axis.set_ylim(lower, lower + (upper - lower) * factor)
            if param in category_names:
                names = category_names[param]
                axis.set_yticks([])
            else:
                _format_best_axis(axis, param,
                                  [lower, lower + (upper - lower) * factor],
                                  padding=0.0)
    plot_path = os.path.normpath(save_path) + "_plot"
    os.makedirs(plot_path, exist_ok=True)
    pdf_name = os.path.splitext(filename)[0] + ".pdf"
    image_path = os.path.join(plot_path, pdf_name)
    fig.savefig(image_path, format="pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    # Additional 15 cm, axis-free comparison. The original figure remains.
    compact_path = _best_feasible_compact(best_df, plot_params, category_names,
                                         save_path, filename)

    table_df = best_df.drop(columns=["_variation", "_order", "display_run"],
                            errors="ignore").copy()

    def _tex_cell(value, parameter):
        if pd.isna(value):
            return "---"
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric) and np.isfinite(numeric):
            return f"{numeric:.6g}"  # si_num stays integer, radius retains decimals
        # to_latex(escape=False): escape special characters in TEXT values.
        return str(value).replace("\\", r"\textbackslash{}") \
                         .replace("_", r"\_") \
                         .replace("&", r"\&") \
                         .replace("%", r"\%")

    for col in table_df.columns:
        table_df[col] = table_df[col].map(lambda value, c=col: _tex_cell(value, c))

    table_df = table_df.rename(columns={col: LABELS.get(col, col)
                                         for col in table_df.columns})

    table_path = os.path.join(save_path,
                              f'{os.path.splitext(table_filename)[0]}.tex'
                              )
    with open(table_path, "w", encoding="utf-8") as file:
        file.write(table_df.to_latex(index=False,
                                     escape=False,
                                     column_format = ("l"
                                                      + "r"
                                                      * (len(table_df.columns)
                                                         - 1))
                                     )
                   )
    return image_path, table_path, best_df

def _best_feasible_compact(best_df, plot_params, category_names, save_path,
                           filename):
    """Separate best-feasible overview, with per-parameter normalised bar heights.

    No y axes are displayed. Numeric labels show unscaled original quantities.
    Each parameter uses its own data range: bar heights are NOT comparable
    between different parameters.
    """
    import matplotlib.patches as mpatches
    width_cm, height_cm = 15.0, 10.2
    fig, ax = plt.subplots(figsize=(width_cm / DEF_cmTOinch,
                                    height_cm / DEF_cmTOinch), dpi=600)
    params = [p for p in plot_params if p in best_df]
    if not params:
        plt.close(fig)
        return None
    n_runs, n_params = len(best_df), len(params)
    colors = fct_setPlotstyle(case=0, flat=True, r=1)[1] or ["0.35"]
    alphas = np.linspace(1, 0.42, n_params) if n_params > 1 else [1.0]
    x = np.arange(n_runs, dtype=float) * 0.89
    group_width = 0.74
    bar_step = group_width / n_params
    offsets = (np.arange(n_params) - (n_params - 1) / 2) * bar_step
    handles = []
    color_algorithms = list(dict.fromkeys(best_df["algorithm"].astype(str)))
    for j, param in enumerate(params):
        raw = best_df[param]
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().all():
            # Heights use the magnitude relative to the parameter maximum.
            # Ensures all columns start at zero and preserve their ordering.
            maximum = max(abs(numeric).max(), 1e-12)
            heights = np.maximum(numeric.to_numpy(dtype=float) / maximum, 0)
        else:
            categories = list(dict.fromkeys(raw.astype(str)))
            heights = np.array([(categories.index(str(v)) + 1) /
                                len(categories) for v in raw], dtype=float)
        handles.append(mpatches.Patch(facecolor="0.45", alpha=alphas[j],
                                      edgecolor="black", label=_legend_label(param)))
        for i, (value, height) in enumerate(zip(raw, heights)):
            bar = ax.bar(x[i] + offsets[j], height, width=bar_step * 0.82,
                         color=colors[color_algorithms.index(str(best_df.iloc[i]["algorithm"])) % len(colors)], alpha=alphas[j],
                         edgecolor="black", linewidth=0.4, zorder=3)[0]
            # Categorical method: use a short vertical marker; see legend below.
            ax.annotate(_bar_value_text(param, value),
                        (bar.get_x() + bar.get_width()/2, float(height)),
                        xytext=(0, 4), textcoords="offset points",
                        rotation=90, ha="center", va="bottom", fontsize=5.8,
                        annotation_clip=False)
    ax.set_xticks(x, best_df["display_run"].astype(str))
    ax.set_xlabel("Optimisation algorithm – seed" if best_df.seed.notna().any()
                  else "Optimisation algorithm")
    ax.set_xlim(x[0] - 0.49, x[-1] + 0.49)
    ax.set_ylim(0, 1.65)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(False)
    fig.subplots_adjust(left=0.45/width_cm, right=1-0.25/width_cm,
                        bottom=2.75/height_cm, top=1-0.45/height_cm)
    fig.legend(handles=handles, title="Parameter", ncol=min(3, n_params),
               frameon=False, fontsize=6, title_fontsize=7,
               loc="lower center", bbox_to_anchor=(0.5, 0.01))
    os.makedirs(os.path.normpath(save_path) + "_plot", exist_ok=True)
    output = os.path.join(os.path.normpath(save_path) + "_plot",
                          os.path.splitext(filename)[0] + "_CONDENSED.pdf")
    fig.savefig(output, format="pdf", dpi=600, bbox_inches=None)
    plt.close(fig)
    return output

###############################################################################
#--------------------------------- EVOLUTION ----------------------------------
###############################################################################

def _plot_evolution(paths,
                    save_path,
                    params,
                    filename,
                    title):
    df = _read(paths)
    if df.empty: return
    algorithm_tag = _algorithm_tag(paths)
    if save_path is None:
        save_path = (paths
                     if isinstance(paths, (str, os.PathLike))
                     else os.path.dirname(str(paths[0])))
    if not params:
        return None
    # Skip unavailable measurements without aborting the other plots.
    # A missing cost is not a categorical label: only finite numeric values
    # are meaningful for cost/settlement evolution.
    usable_params = []
    for param in params:
        if param not in df.columns:
            print(f"_plot_evolution: '{param}' missing; skipping it.")
            continue
        if param in ("cost", "u"):
            numeric = pd.to_numeric(df[param], errors="coerce")
            valid = np.isfinite(numeric.to_numpy(dtype=float))
            valid &= df.status.astype(str).str.strip().eq("OK").to_numpy()
            if not valid.any():
                print(f"_plot_evolution: no valid '{param}' values; skipping it.")
                continue
        elif not df[param].notna().any():
            print(f"_plot_evolution: no '{param}' values; skipping it.")
            continue
        usable_params.append(param)
    params = usable_params
    if not params:
        print("_plot_evolution: no plottable parameters found; skipping plot.")
        return None
    # Every parameter gets one consistent encoding across all run groups.
    prepared = {param: _prepare_plot_axis(df, param) for param in params}
    colors = fct_setPlotstyle(case=0, flat=True, r=1)[1]
    groups = _study_groups(df)
    fig, axes = plt.subplots(len(params),
                             1,
                             figsize=(DEF_figWidth,
                                      (3 * len(params) + 1.8) / DEF_cmTOinch
                                      ),
                             sharex=True,
                             squeeze=False
                             )
    handles = []
    cost_scale = 1.0
    cost_exponent = 0
    if "cost" in params:
        cost_values = pd.to_numeric(df["cost"],
                                    errors="coerce")
        cost_values = cost_values[np.isfinite(cost_values)
                                  & df.status.astype(str).str.strip().eq("OK")]
        if len(cost_values) > 0:
            cost_step = (cost_values.max() - cost_values.min()) / 5
            cost_scale, cost_exponent = _cost_scale(cost_step)
    for j, (label, group) in enumerate(groups):
        color = colors[j % len(colors)]
        legend_label = str(label).split("(", 1)[0].strip()
        handles.append(Line2D([],
                              [],
                              linestyle="None",
                              marker="o",
                              markersize=5,
                              color=color,
                              label=legend_label
                              )
                       )
        for i, param in enumerate(params):
            data = (group
                    if param not in ("cost", "u")
                    else group.loc[
                        group.status.astype(str)
                        .str.strip()
                        .eq("OK")
                        ].copy()
                    )
            yy = prepared[param][0].loc[data.index].astype(float)
            if param == "cost": yy = yy / cost_scale
            mask = np.isfinite(yy)
            axes[i, 0].scatter(data.loc[mask, "evaluation"],
                               yy[mask],
                               s=12,
                               color=color,
                               alpha=0.72
                               )
    for i, param in enumerate(params):
        ylabel = LABELS.get(param, param)
        if param == "cost" and cost_exponent != 0:
            ylabel += rf" [$\times 10^{{{cost_exponent}}}$]"
        axes[i, 0].set_ylabel(ylabel)
        categories = prepared[param][1]
        if categories is not None:
            axes[i, 0].set_yticks(range(len(categories)))
            axes[i, 0].set_yticklabels(categories, fontsize=7)
            axes[i, 0].set_ylim(-0.5, len(categories) - 0.5)
        axes[i, 0].grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Evaluation")
    one = len(groups) == 1
    _suptitle(fig, _run_title(title, df, groups))
    extra_left_cm = 0.0
    if any(prepared[param][1] is not None for param in params):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        widest_label_px = max( (tick.get_window_extent(renderer).width
                                for i, param in enumerate(params)
                                if prepared[param][1] is not None
                                for tick in axes[i, 0].get_yticklabels()
                                if tick.get_text()),
                              default=0.0,
                              )
        widest_label_cm = widest_label_px / fig.dpi * DEF_cmTOinch
        standard_left_cm = DEF_axisLeft * DEF_figWidth * DEF_cmTOinch
        extra_left_cm = max(0.0,
                            widest_label_cm + 1.1 - standard_left_cm
                            )
        if extra_left_cm > 0:
            fig.set_size_inches((fig.get_figwidth() + extra_left_cm
                                 / DEF_cmTOinch),
                                fig.get_figheight(),
                                forward=True,
                                )
    _legend_layout(fig,
                   [] if one else [("Algorithms", handles)],
                   nrows          = len(params),
                   extra_left_cm  = extra_left_cm,
                   )
    return fct_formatSave(True,
                          fig,
                          os.path.normpath(save_path) + "_plot",
                          f"{algorithm_tag}_{filename}")

def plot_evolution_parameter(paths,
                             save_path=None,
                             params=None):
    df = _read(paths)
    if df.empty: return
    params = (params
              or [p
                  for p in PARAMS
                  if p in df and df[p].nunique() > 1]
              )
    return _plot_evolution(paths,
                           save_path,
                           params,
                           "optimization_evolution_parameter",
                           "Parameter evolution"
                           )

def plot_evolution_cost_settlement(paths,
                                   save_path=None):
    return _plot_evolution(paths,
                           save_path,
                           ["cost", "u"],
                           "optimization_evolution_cost_settlement",
                           "Cost and settlement evolution"
                           )
            

###############################################################################
#---------------------------- Parallel COORDINATES ----------------------------
###############################################################################

def plot_parallel_coordinates(paths,
                              save_path=None,
                              params=None,
                              max_lines=100):
    df = _read(paths)
    if df.empty: return
    algorithm_tag = _algorithm_tag(paths)
    df = df.loc[df.status.astype(str).str.strip().eq("OK")].copy()
    if save_path is None:
        save_path = (paths
                     if isinstance(paths, (str, os.PathLike))
                     else os.path.dirname(str(paths[0])))
    params = (params
              or [p
                  for p in PARAMS
                  if p in df
                  and df[p].nunique() > 1]
              )
    cols = params + ["cost", "u"]
    if df.empty or not params:
        return None
    df = df.dropna(subset=cols).copy()
    if df.empty:
        return None
    best_feasible_name = None
    feasible_df = df[df.feasible].copy()
    if not feasible_df.empty:
        best_feasible_name = feasible_df.loc[feasible_df["cost"].idxmin(),
                                             "name"]
    df = pd.concat([df[df.feasible].nsmallest(max_lines // 2, "cost"),
                    df.nsmallest(max_lines // 2, "cost")
                    ]).drop_duplicates("name").head(max_lines)
    if best_feasible_name is not None:
        best_original = feasible_df[feasible_df["name"].eq(best_feasible_name)]
        if best_feasible_name not in df["name"].values:
            df = pd.concat([df, best_original],
                           ignore_index=True)
    normalized = pd.DataFrame(index=df.index)
    category_names = {}
    for col in cols:
        yy, names = _prepare_plot_axis(df, col)
        normalized[col] = yy
        if names is not None:
            category_names[col] = names
        lo, hi = normalized[col].min(), normalized[col].max()
        normalized[col] = ((normalized[col] - lo) / (hi - lo)
                           if hi > lo
                           else normalized[col] * 0 + 0.5
                          )
    fig, ax = plt.subplots(figsize=(DEF_figWidth,
                                    DEF_figHeight))
    colors = fct_setPlotstyle(case=0,
                              flat=True,
                              r=1)[1]
    coloridx_feasible = 0
    coloridx_constraint = 1
    for (_, row), (_, raw) in zip(normalized.iterrows(),
                                  df.iterrows()):
        ax.plot(range(len(cols)),
                row.to_numpy(dtype=float),
                color=(colors[coloridx_feasible]
                       if raw.feasible
                       else colors[coloridx_constraint]),
                alpha=0.22,
                lw=1
                )
    if best_feasible_name is not None:
        mask_best = df["name"].eq(best_feasible_name)
        if mask_best.any():
            best_row = normalized.loc[mask_best].iloc[0]
            ax.plot(range(len(cols)),
                    best_row.to_numpy(dtype=float),
                    color="black",
                    alpha=1.0,
                    lw=1.8,
                    zorder=10
                    )
    ax.set(xticks=range(len(cols)),
           xticklabels=[LABELS.get(c, c)
                        for c in cols],
           ylabel="Scaled value within displayed designs",
           ylim=(-0.03, 1.03)
           )
    ax.tick_params(axis="x", rotation=25)
    ax.grid(alpha=0.2)
    # The shared y-axis is normalised; annotate categorical values beside
    # their own parameter axis instead of presenting meaningless numeric codes.
    for col, names in category_names.items():
        i = cols.index(col)
        for j, name in enumerate(names):
            yy = (j / (len(names) - 1) if len(names) > 1 else 0.5)
            ax.annotate(str(name), (i, yy), xytext=(4, 2),
                        textcoords="offset points", fontsize=5)
    legend = [Line2D([],
                     [],
                     color=colors[coloridx_feasible],
                     label="Feasible"
                     ),
              Line2D([],
                     [],
                     color=colors[coloridx_constraint],
                     label="Constraint violated"
                     ),
              Line2D([],
                     [],
                     color="black",
                     lw=1.8,
                     label="Best feasible solution"
                     )
              ]
    groups = _study_groups(df)
    _suptitle(fig, _run_title("Parallel coordinates", df, groups))
    _legend_layout(fig,
                   [("Status", legend)])
    return fct_formatSave(True,
                          fig,
                          os.path.normpath(save_path) + "_plot",
                          f"{algorithm_tag}_optimization_parallel_coordinates")

###############################################################################
#------------------------------------ MAIN ------------------------------------
###############################################################################

def _find_sibling_seed_runs(path):
    """Discover every algorithm/seed run belonging to the current study.

    Examples: Opt_Yasser210_seed1_GA, Opt_Yasser210_seed2_DE,
    Opt_Yasser210_seed3_comparison. Only folders with CSV histories are used.
    """
    folder = os.path.normpath(str(path))
    parent, basename = os.path.split(folder)
    match = re.search(r"seed[_-]?(\d+)(?=[_-]|$)", basename, re.I)
    if match is None:
        return [folder] if os.path.isfile(os.path.join(folder, "optimization_history.csv")) else []
    prefix = basename[:match.start()]
    candidates = []
    for name in os.listdir(parent):
        candidate = os.path.join(parent, name)
        if not os.path.isdir(candidate) or name.lower().endswith('_plot'):
            continue
        m = re.fullmatch(re.escape(prefix) + r"seed[_-]?(\d+)(?:[_-].*)?",
                         name, re.I)
        if m is None or not os.path.isfile(os.path.join(candidate, "optimization_history.csv")):
            continue
        candidates.append((int(m.group(1)), candidate))
    return [candidate for _, candidate in sorted(candidates, key=lambda row: (row[0], row[1].lower()))]


def _best_feasible_valid_paths(paths):
    """Keep individual runs that have at least one successful feasible cost."""
    result = []
    for folder in paths:
        df = _read(folder)
        if df.empty or not {'status', 'feasible', 'cost'} <= set(df.columns):
            print('Best feasible: skipping incomplete history:', folder)
            continue
        mask = (df['status'].astype(str).str.strip().eq('OK') &
                df['feasible'].astype(bool) &
                np.isfinite(pd.to_numeric(df['cost'], errors='coerce')))
        if mask.any():
            result.append(folder)
        else:
            print('Best feasible: no valid feasible result:', folder)
    return result


def opt_plot(path, requested, cfg):
    fct_setPlottext()
    outputs = {}
    if "method_ranges" in requested:
        limits = cfg.METHOD_LIMITS or get_buildingData("limits")
        selected = {m: limits[m]
                    for m in cfg.user_methods}
        outputs["method_ranges"] = plot_methodRanges(selected,
                                                     ["diameter",
                                                      "ltp",
                                                      "stiffness"],
                                                     savePath=path
                                                     )
    if not os.path.exists(os.path.join(path, "optimization_history.csv")):
        if requested - {"method_ranges"}:
            print("No history in " + path)
        return outputs
    if ("landscape_bars" in requested
        or "landscape_surface" in requested):
        variant = ("both"
                   if {"landscape_bars",
                       "landscape_surface"} <= requested
                   else "bars"
                   if "landscape_bars" in requested
                   else "surface"
                   )
        outputs["landscape"] = plot_landscape(path, variant=variant)
    if "best_feasible" in requested:
        run_folder = os.path.normpath(str(path))
        parent, run_name = os.path.split(run_folder)
        # All algorithms and seeds share one output folder, even when the
        # selected run ends in _GA, _DE, _PSO, or _comparison.
        m = re.search(r"seed[_-]?\d+(?=[_-]|$)", run_name, re.I)
        if m:
            common_name = run_name[:m.start()] + "seed_comparison"
        else:
            common_name = run_name + "_seed_comparison"
        shared_save_path = os.path.join(parent, common_name)
        seed_paths = _best_feasible_valid_paths(_find_sibling_seed_runs(path))
        print("Best feasible runs:", *seed_paths, sep="\n  ")
        outputs["best_feasible"] = {}
        if seed_paths:
            # Individual algorithm plots and one additional combined plot.
            by_algorithm = {}
            for run_path in seed_paths:
                run_df = _read(run_path)
                for algorithm in run_df['algorithm'].dropna().astype(str).unique():
                    by_algorithm.setdefault(algorithm, []).append(run_path)
            for algorithm, algo_paths in sorted(by_algorithm.items()):
                # A CSV normally contains only one algorithm. Avoid duplicating
                # mixed-algorithm CSV folders in several individual plots.
                if all(_read(f)['algorithm'].nunique() == 1 for f in algo_paths):
                    print('Best feasible algorithm:', algorithm, len(algo_paths), 'runs')
                    outputs['best_feasible'][algorithm] = plot_best_feasible(
                        algo_paths, shared_save_path)
            print('Best feasible combined:', len(seed_paths), 'runs')
            outputs['best_feasible']['all_algorithms'] = plot_best_feasible(
                seed_paths, shared_save_path)
        else:
            print('Best feasible: no usable runs found.')
        print('Best feasible plots:', shared_save_path + '_plot')
    if ("landscape_feasible" in requested):
        outputs["landscape"] = plot_landscape(path, variant="feasible")
        plot_best_feasible(path, path)
    if "convergence" in requested:
        outputs["convergence"] = plot_convergence(path)
    if "evolution" in requested:
        outputs["evolution"] = plot_evolution_parameter(path)
    if "cost_settlement" in requested:
        outputs["cost_settlement"] = plot_evolution_cost_settlement(path)
    if "parallel" in requested:
        outputs["parallel"] = plot_parallel_coordinates(path)
    return outputs

# BEST FEASIBLE POINTS

def _short_folder_labels(names):
    """Gemeinsamen Anfang und gemeinsames Ende entfernen."""
    names = [str(name) for name in names]
    if (len(names) < 2 or
        len(set(names)) != len(names)):
        return names
    prefix = os.path.commonprefix(names)
    tails = [name[len(prefix):] for name in names]
    suffix = os.path.commonprefix([tail[::-1]
                                   for tail in tails])[::-1]
    shortened = [tail[:-len(suffix)]
                 if suffix else tail
                 for tail in tails]
    return shortened if all(shortened) else names

def read_best_feasible_tables(paths,
                              labels=None,
                              table_filename="optimization_best_feasible.tex"):
    from pathlib import Path
    paths = ([paths]
             if isinstance(paths, (str, os.PathLike))
             else list(paths))
    if not paths:
        raise ValueError("paths darf nicht leer sein.")
    if labels is not None and len(labels) != len(paths):
        raise ValueError("labels muss dieselbe Länge wie paths haben.")
    aliases = {label: parameter
               for parameter, label in LABELS.items()}
    aliases.update({"run": "algorithm",
                    "algorithm": "algorithm",
                    "Max. evaluation [-]": "max_evaluation",
                    "Max. iteration [-]": "max_iteration",
                    })
    frames = []
    for index, path in enumerate(paths):
        path = Path(path)
        if path.is_dir():
            file = path / table_filename
            if not file.is_file():
                matches = sorted(path.glob("*_" + table_filename))
                if not matches and table_filename == "optimization_best_feasible.tex":
                    matches = sorted(path.glob("*_optimization_best_feasible_all.tex"))
                if len(matches) != 1:
                    raise ValueError(
                        f"Keine eindeutige Tabelle in:\n{path}\n"
                        f"Gefundene Dateien: {len(matches)}\n"
                        f"Bitte den vollständigen .tex-Pfad {table_filename} "
                        "angeben."
                        )
                file = matches[0]
        else:
            file = path
        content = file.read_text(encoding="utf-8-sig")
        match = re.search(r"\\begin\{tabular\}(?:\[[^\]]*\])?\{[^\n]*\}"
                          r"(.*?)\\end\{tabular\}",
                          content,
                          re.DOTALL,
                          )
        if match is None:
            raise ValueError(f"Keine tabular-Umgebung gefunden:\n{file}")
        body = re.sub(r"(?<!\\)%[^\n]*",
                      "",
                      match.group(1),)
        body = re.sub(r"\\(?:toprule|midrule|bottomrule|hline)\b"
                      r"(?:\[[^\]]*\])?",
                      "",
                      body,)
        rows = []
        for row in re.split(r"\\\\(?:\[[^\]]*\])?", body):
            if not row.strip():
                continue
            cells = [cell.strip()
                     .replace(r"\_", "_")
                     .replace(r"\&", "&")
                     for cell in re.split(r"(?<!\\)&", row)]
            rows.append(cells)
        if (len(rows) < 2
            or any(len(row) != len(rows[0])
                   for row in rows)):
            raise ValueError(f"Ungültige oder leere Tabelle:\n{file}")
        columns = [
            column if column == "run" and "algorithm" in rows[0]
            else aliases.get(column, column)
            for column in rows[0]
            ]
        from collections import Counter
        duplicates = {
            name: count
            for name, count in Counter(columns).items()
            if count > 1
            }
        if duplicates:
            print(f"Datei: {file}")
            print(f"Originale Spalten: {rows[0]}")
            print(f"Spalten nach aliases: {columns}")
            print(f"Doppelte Spalten: {duplicates}")
        if len(set(columns)) != len(columns):
            raise ValueError(f"Doppelte Spaltennamen:\n{file}")
        data = pd.DataFrame(rows[1:], columns=columns,)
        if "algorithm" not in data:
            raise ValueError(f"Spalte 'run' fehlt:\n{file}")
        data["algorithm"] = data["algorithm"].str.replace("MixedVariableGA",
                                                          "MVGA",
                                                          case=False,
                                                          regex=False,
                                                          )
        if data["algorithm"].duplicated().any():
            raise ValueError(
                f"Mehrere Zeilen für denselben Algorithmus:\n{file}"
                )
        # Convert to numbers only if ALL non-missing entries are numeric.
        # Otherwise preserve the original text (e.g. si_method = "VR").
        for column in data.columns.drop("algorithm"):
            original = data[column].astype("string").str.strip()
            original = original.mask(original.isin(["", "-", "--", "---"]))
            numeric = pd.to_numeric(original, errors="coerce")
            if numeric[original.notna()].notna().all():
                data[column] = numeric.astype(float)
            else:
                data[column] = original
        data["study_index"] = index
        data["study"] = (str(labels[index])
                         if labels is not None
                         else file.parent.name
                         )
        data["table_path"] = str(file)
        frames.append(data)
    return pd.concat(frames, ignore_index=True, sort=False,)

def _prepare_plot_axis(data, parameter):
    """Numeric series or category positions; category names if categorical.

    Conversion is column-wide: all valid values must be numeric to use a
    numeric axis.  In particular, method names are never lost through coerce.
    """
    if parameter not in data.columns:
        raise ValueError(f"Parameter '{parameter}' not found.")

    values = data[parameter].astype("string").str.strip()
    values = values.mask(values.isin(["", "-", "--", "---", "nan", "NaN",
                                    "None", "<NA>"]))
    valid = values.notna()
    if not valid.any():
        raise ValueError(f"Keine gültigen Werte für '{parameter}' gefunden.")

    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if numeric[valid].notna().all():
        return numeric.astype(float), None

    categories = list(dict.fromkeys(values[valid].tolist()))
    mapping = {value: i for i, value in enumerate(categories)}
    positions = values.map(mapping).astype(float)
    return positions, categories


def plot_best_feasible_points(paths,
                              save_path=None,
                              x="si_L",
                              y="si_R",
                              labels=None,
                              table_filename="optimization_best_feasible.tex",
                              filename="optimization_best_feasible_points",
                              save=True):
    data = read_best_feasible_tables(paths=paths,
                                     labels=labels,
                                     table_filename=table_filename,
                                     )
    # Backward-compatible names used by older calling scripts.
    parameter_aliases = {"ltp": "ltp_t", "slab": "slab_t"}
    x = parameter_aliases.get(x, x)
    y = parameter_aliases.get(y, y)
    if x == y:
        raise ValueError("x und y müssen unterschiedlich sein.")
    plot_x, categories_x = _prepare_plot_axis(data, x)
    plot_y, categories_y = _prepare_plot_axis(data, y)
    algorithms = list(dict.fromkeys(data["algorithm"]))
    studies = (data[["study_index", "study"]].drop_duplicates("study_index"))
    folder_labels = (list(studies["study"])
                     if labels is not None
                     else _short_folder_labels(studies["study"]))
    colors = fct_setPlotstyle(case=0, flat=True, r=1,)[1]
    markers = ["o", "s", "^", "D", "v", "P","X", "<", ">", "h", "*", "p",]
    if len(studies) > len(markers):
        raise ValueError(f"Maximal {len(markers)} Ordner unterstützt.")
    color_map = {algorithm: colors[i % len(colors)]
                 for i, algorithm in enumerate(algorithms)
                 }
    marker_map = dict(zip(studies["study_index"], markers))
    height_cm = max(8.0,
                    2.0 + 0.45 * (len(algorithms) + len(studies) + 4),
                    )
    fig, ax = plt.subplots( figsize=(DEF_figWidth, height_cm / DEF_cmTOinch))
    for study_index, group in data.groupby("study_index", sort=False):
        for algorithm, points in group.groupby("algorithm", sort=False):
            xx = plot_x.loc[points.index]
            yy = plot_y.loc[points.index]
            valid = np.isfinite(xx) & np.isfinite(yy)
            ax.scatter(xx[valid],
                       yy[valid],
                       marker       = marker_map[study_index],
                       edgecolors   = color_map[algorithm],
                       facecolors   = "none",
                       s            = 38,
                       linewidths   = 1.1,
                       alpha        = 0.85,
                       zorder       = 3,
                       )
    extra_labels = {"max_evaluation": "Max. evaluation [-]",
                    "max_iteration": "Max. iteration [-]",
                    }
    ax.set_xlabel(LABELS.get(x, extra_labels.get(x, x)), fontsize=7)
    ax.set_ylabel(LABELS.get(y, extra_labels.get(y, y)), fontsize=7)
    if categories_x is not None:
        ax.set_xticks(range(len(categories_x)))
        ax.set_xticklabels(categories_x, rotation=45, ha="right")
        ax.set_xlim(-0.5, len(categories_x) - 0.5)
    if categories_y is not None:
        ax.set_yticks(range(len(categories_y)))
        ax.set_yticklabels(categories_y)
        ax.set_ylim(-0.5, len(categories_y) - 0.5)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.2)
    ax.margins(0.12)
    for direction, parameter in (("x", x), ("y", y)):
        if parameter == "cost":
            ax.ticklabel_format(axis        = direction,
                                style       = "sci",
                                scilimits   = (0, 0),
                                useMathText = True,
                                useOffset   = False,
                                )
    algorithm_handles = [Line2D([], [],
                                 linestyle      = "None",
                                 marker         = "o",
                                 color          = color_map[algorithm],
                                 markerfacecolor= "none",
                                 markersize     = 5,
                                 label          = algorithm,
                                 )
                         for algorithm in algorithms]
    folder_handles = [Line2D([], [],
                             linestyle      = "None",
                             marker         = marker_map[index],
                             color          = "0.25",
                             markerfacecolor= "none",
                             markersize     = 5,
                      label=re.sub(r"(?i)^Opt_\d{4}-\d{2}-\d{2}-", "", label),
                      )
                      for index, label in zip(studies["study_index"],
                                              folder_labels,
                                              )
                      ]
    _legend_layout(fig,
                   [("Algorithms", algorithm_handles),
                    ("Seeds", folder_handles),
                    ],
                   )
    saved_path = None
    if save:
        if save_path is None:
            save_path = os.path.dirname(data["table_path"].iloc[0])
        saved_path = fct_formatSave(True, fig,
                                    os.path.normpath(save_path) + "_plot",
                                    f"{filename}_{x}_{y}",
                                    )
    return saved_path, data, fig, ax

### STATISTICS

def _stat_termination(history, settings):
    """Am letzten CSV-Eintrag erfüllte Terminierungsbedingungen."""
    if not settings:
        return "unknown"

    n = len(history)
    budget = settings.get("calcSize")
    reached_budget = budget is not None and n >= int(budget)
    window = int(settings.get("stop_window") or 0)
    stagnated = False

    if window and n > window:
        feasible = history["feasible"]
        costs = pd.to_numeric(history["cost"], errors="coerce")
        old = costs.iloc[:-window][feasible.iloc[:-window]].min()
        new = costs[feasible].min()
        if (pd.notna(old) and pd.notna(new)
            and abs(old - new) / max(abs(old), 1.0)
                <= float(settings.get("stop_cost_rel", 0.001))
            and feasible.iloc[-window:].any()):
            stagnated = True
            tolerance = float(settings.get("stop_cv_abs") or 0)
            if tolerance:
                tail = history.iloc[-window:][G_COLS]
                cv = (tail.apply(pd.to_numeric, errors="coerce")
                      .clip(lower=0).fillna(0).sum(axis=1))
                stagnated = cv.max() - cv.min() <= tolerance

    reasons = []
    if reached_budget:
        reasons.append("Budget")
    if stagnated:
        reasons.append("Stagnation")
    if str(history["algorithm"].iloc[0]).lower() == "grid":
        reasons = ["Grid"]
    return " / ".join(reasons) or "unknown"

def plot_optimization_statistics(RUN_DEVICE,
                                 RUN_PATH_MAIN,
                                 show_legend_seed=True,
                                 show_legend_algo=False,
                                 testing=False
                                 ):
    import json
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    # seed1, seed2, ... -> seed
    RUN_PATH_MAIN = re.sub(r"(?<=seed)\d+$", "",
                           RUN_PATH_MAIN,
                           flags=re.IGNORECASE)
    _, _, path = open_plaxisData(RUN_DEVICE,
                                 'opt',
                                 RUN_PATH_MAIN,
                                 create=False)
    basepath = os.path.dirname(os.path.normpath(path))
    fct_setPlottext()
    records = []
    for name in sorted(os.listdir(basepath)):
        match = re.fullmatch(re.escape(RUN_PATH_MAIN) + r"(\d+)[_-](.+)",
                             name,
                             flags=re.IGNORECASE
                             )
        if not match: continue
        path = os.path.join(basepath, name)
        if not (os.path.isdir(path) and
                os.path.isfile(os.path.join(path,
                                            "optimization_history.csv"))):
            continue
        history = (_read(path).sort_values("evaluation")
                   .reset_index(drop=True))
        if history.empty:
            continue
        settings_path = os.path.join(path, "optimization_settings.json")
        settings = {}
        if os.path.isfile(settings_path):
            with open(settings_path, encoding="utf-8") as file:
                settings = json.load(file)
        history["cost"] = pd.to_numeric(history["cost"], errors="coerce")
        history["u"] = pd.to_numeric(history["u"], errors="coerce")
        valid = (history.feasible
                 & history.status.astype(str).str.strip().eq("OK")
                 & np.isfinite(history.cost)
                 & np.isfinite(history.u))
        feasible = history.loc[valid]
        best = (feasible.loc[feasible.cost.idxmin()]
                if not feasible.empty
                else None)
        timestamps = pd.to_datetime(history["time"],
                                    errors="coerce").dropna()
        runtime = ((timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
                   / 3600
                   if len(timestamps) > 1
                   else np.nan)
        u_limit = settings.get("limits", {}).get("u", np.nan)
        study = re.sub(r"^Opt_", "", RUN_PATH_MAIN, flags=re.IGNORECASE)
        study = re.sub(r"[-_]?seed$", "", study, flags=re.IGNORECASE)
        study = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", study)

        records.append(dict(Study       = study,
                            Algorithm   = history.algorithm.iloc[0],
                            Seed        = int(match.group(1)),
                            Cost        = (float(best.cost)
                                           if best is not None
                                           else np.nan),
                            Settlement  = (float(best.u) * 1000
                                           if best is not None
                                           else np.nan),
                            Limit       = (pd.to_numeric(u_limit,
                                                         errors="coerce")
                                           * 1000),
                            Runtime     = runtime,
                            Evaluations = int(history.evaluation.max()),
                            Termination = _stat_termination(history,
                                                            settings),
                            )
                       )
    if not records:
        raise ValueError(f"no optimization_history.csv found: {basepath}")
    df = pd.DataFrame(records).sort_values(["Study",
                                            "Algorithm",
                                            "Seed"])
    save_path = os.path.join(basepath,
                             f'{RUN_PATH_MAIN}_comparison')
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, "optimization_statistics.csv"),
              sep=";", index=False)
    table = df.rename(columns={"Cost":          "Cost [EUR]",
                               "Settlement":    "Settlement [mm]",
                               "Limit":         "Limit [mm]",
                               "Runtime":       "Runtime [h]",
                               }
                      )
    tex_path = os.path.join(save_path,
                            "optimization_statistics.tex")
    table.to_latex(tex_path,
                   index        = False,
                   escape       = True,
                   na_rep       = "--",
                   float_format = lambda x: f"{x:.2f}",
                   longtable    = True,
                   column_format= "llrrrrrrl",
                   caption      = "Best feasible solutions and recorded runtimes.",
                   label        = "tab:optimization_statistics")
    # Plot settings
    algorithms = df.Algorithm.unique().tolist()
    seeds = sorted(df.Seed.unique())
    colors = fct_setPlotstyle(case=0, flat=True, r=1)[1]
    cmap = {a: colors[i % len(colors)]
            for i, a in enumerate(algorithms)}
    alphas = dict(zip(seeds,
                      np.linspace(1.0, 0.4, len(seeds))))
    metrics = [("Cost",       1e-3, r"Cost [€$\times 10^3$]"),
               ("Settlement", 1,    "Settlement [mm]"),
               ("Runtime",    1,    "Runtime [h]")]
    # Separate PDF for each study and metric
    DEF_figWidth_specificLEG = 4 / DEF_cmTOinch
    DEF_figWidth_specific = DEF_figWidth - DEF_figWidth_specificLEG
    n_yintervals = 3
    y_padding = 0.08
    for study, df_study in df.groupby("Study", sort=False):
        # study_path = os.path.join(save_path,
        #                           study
        #                           )
        study_path = save_path
        os.makedirs(study_path, exist_ok=True)
        # One bar for each algorithm/seed combination
        df_plot = df_study.sort_values(["Algorithm",
                                        "Seed"]
                                       ).reset_index(drop=True)
        n_bars = len(df_plot)
        bar_width = 0.75
        x = np.arange(n_bars) * 0.85
        # X positions grouped by algorithm
        algo_centres = []
        algo_labels = []
        for algorithm in algorithms:
            indices = np.where(df_plot.Algorithm.eq(algorithm))[0]
            if len(indices):
                algo_centres.append(x[indices].mean())
                algo_labels.append(algorithm)
        for field, scale, ylabel in metrics:
            fig, ax = plt.subplots(figsize=(DEF_figWidth_specific,
                                            DEF_figHeight/2))
            # Relevant data range
            values = pd.to_numeric(df_plot[field],
                                   errors="coerce").to_numpy(dtype=float)*scale
            values = values[np.isfinite(values)]
            if len(values) == 0:
                plt.close(fig)
                continue
            # Settlement limit
            limits = np.array([], dtype=float)
            if field == "Settlement":
                limits = pd.to_numeric(df_study.Limit,
                                       errors="coerce").to_numpy(dtype=float)
                limits = limits[np.isfinite(limits)]
            # Automatic axis limits
            if field in ("Cost", "Settlement"):
                all_values = np.concatenate([values, limits])
                ymin = float(np.min(all_values))
                ymax = float(np.max(all_values))
                span = ymax - ymin
                if span <= 0: span = max(abs(ymax) * 0.05, 0.01)
                lower = ymin - y_padding * span
                upper = ymax + y_padding * span
                lower = max(0.0, lower)
            else:
                lower = 0.0
                upper = float(np.max(values)) * 1.08
                if upper <= 0: upper = 1.0
            # Bars
            for i, row in df_plot.iterrows():
                value = row[field]
                if not np.isfinite(value): continue
                ax.bar(x[i],
                       value * scale,
                       width=bar_width,
                       color=cmap[row.Algorithm],
                       alpha=alphas[row.Seed],
                       edgecolor="none",
                       zorder=3)
            # Settlement limit
            if field == "Settlement":
                for limit in np.unique(limits):
                    ax.axhline(limit,
                               color="0.4",
                               linestyle="--",
                               linewidth=0.7,
                               zorder=2
                    )

            # Y-axis formatting
            ax.set_ylim(lower, upper)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=n_yintervals,
                                                   min_n_ticks=3,
                                                   steps=[1, 2, 2.5, 5, 10]
                                                   )
                                       )
            ax.tick_params(axis="y", labelsize=7, pad=2)
            # X-axis formatting
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Algorithm")
            ax.set_xticks(algo_centres)
            ax.set_xticklabels(algo_labels)
            ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
            ax.tick_params(axis="x", labelsize=7, pad=2)
            ax.grid(axis="y", alpha=0.2, zorder=0)
            # Study label in upper-right corner
            ax.text(0.98, 0.97, study,
                    transform=ax.transAxes,
                    ha="right", va="top")
            # Separate legend PDF
            legends = []
            if show_legend_algo:
                handles_algo = [Patch(facecolor=cmap[a],
                                      label=a)
                                for a in algorithms]
                legends.append(("Algorithms", handles_algo))
            if show_legend_seed:
                handles_seed = [Patch(facecolor="black",
                                      alpha=alphas[s],
                                      label=f"Seed {s}")
                                for s in seeds]
                legends.append(("Seeds", handles_seed))
            if legends:
                fig_leg = plt.figure(figsize=(DEF_figWidth_specificLEG,
                                              DEF_figHeight)
                                     )
                _legend_layout(fig_leg, legends, nrows=2,
                               legend_only=True)
                filename = (f"optimization_statistics_{field.lower()}_legend")
                fct_formatSave(True, fig_leg, study_path, filename)
                plt.close(fig_leg)
            # Plot without legend
            fig.tight_layout()
            # Separate PDF for each metric
            filename = (f"optimization_statistics_{field.lower()}")
            pdf_path = fct_formatSave(True, fig, study_path, filename)
            if testing: print(f"Plot: {pdf_path}")
    if testing: print(f"LaTeX: {tex_path}")
    return df
