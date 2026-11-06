# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 09:20:29 2026

@author: annal
"""
"""
Created on Sun Jun 14 07:26:46 2026
@author: annaleoniesalzmann
"""

###############################################################################
###############################################################################
# Imports
###############################################################################
###############################################################################

import re
import ast
import math
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

from plaxis_00calling_02helpers import (fct_setPlotstyle,
                                        fct_formatSave,
                                        open_plaxisData)

###############################################################################
###############################################################################
# DATA
###############################################################################
###############################################################################

def get_buildingData(typ): # 'limits', 'costs', 'params'
    # VR, RI, RAP, VS, DSM, JG, RP, DP
    METHOD_LIMITS = {
                     'VR': {"name": "Vibro Replacement",
                            "diameter":  (0.55, 0.90),
                            "spacing":   (1.40, 3.00),
                            "stiffness": (60e3, 130e3)
                            },
                     'RI': {"name": "Rigid Inclusion",
                           "diameter": (0.20, 0.42),
                           "spacing":  ('4*diameter', 3.50),
                           "ltp":      (0.4, 2.0),
                           "stiffness": 30e6
                            },
                     'RAP': {"name": "Rammed Aggregate Piers",
                             "diameter": (0.55, 0.65),
                             "spacing":  (1.4, 3.5),
                             "ltp":      (0.3, 2.0),
                             "stiffness": 27e6
                             },
                     'VC': {"name": "Vibro Concrete Columns",
                            "diameter": (0.45, 0.50),
                            "stiffness": 27e6
                            },
                     'DSM': {"name": "Deep Soil Mixing",
                             "diameter": (0.60, 0.80),
                             },
                     'JG': {"name": "Jet Grouting Columns",
                            "diameter":  (0.60, 2.20),
                            "stiffness": (200*3e3, 800*10e3),
                            "ltp":       0,
                            },
                     'RP': {"name": "Replacement Piles",
                            "diameter":  (0.40, 1.50),
                            "spacing":   ('2*diameter', 'inf.'),
                            "ltp":       0,
                            "stiffness": 30e6
                            },
                     'DP': {"name": "Displacement Piles",
                            "diameter":  (0.40, 0.62),
                            "spacing":   ('6*diameter', 'inf.'),
                            "ltp":       0,
                            "stiffness": 30e6
                            },
                     'VR_Nguyen': {"name": "Vibro Replacement",
                                  "diameter":  (0.6, 1.4),
                                  "spacing":   (1.40, 3.00),
                                  },
                     'RI_Nguyen': {"name": "Rigid Inclusion",
                                   "diameter": (0.20, 0.85),
                                   "spacing":  (1.5, 3.0),
                                   },
                     'VR_Yasser': {"name":      "Vibro Replacement",
                                   "diameter":  (1.00, 1.50),
                                   "spacing":   (1.40, 3.00),
                                   "ltp":       (0.00, 2.625),
                                   "stiffness": (100e3, 100e3),
                                   },
                     'VR_quarry': {"name":      "Vibro Replacement",
                                   "spacing":   (1.5, 2),#ref parameter
                                   },
                     'VR_crushedpebble': {"name":      "Vibro Replacement",
                                          "spacing":   (1.5, 2),#ref parameter
                                          },
                     'VR_pebble': {"name":      "Vibro Replacement",
                                   "spacing":   (1.5, 2),#ref parameter
                                   },
                     'VR_Bouassida': {"name":      "Vibro Replacement",
                                      "diameter":  (0.7, 0.9),
                                      },
                     }

    METHOD_COSTS = {# table 2.1 und 2.3 in my thesis
                  "concrete": {"cost_eur_m3": 137.5,
                               "co2_kg_m3": 280.5,
                               },
                  "ltp": {"cost_eur_m3": 30,
                          "co2_kg_m3": 11.6,
                          },
                  "excavation": {"cost_eur_m3": 18.5,
                                 "co2_kg_m3": 0,
                                 },
                  "improvement": {"DSM": {"name": "Deep Soil Mixing",
                                          "cost_eur_m3": 218.3,
                                          "co2_kg_m3": 161,
                                          },
                                  "RAP": {"name": "Rammed Aggregate Piers",
                                          "cost_eur_m3": 293.3,
                                          "co2_kg_m3": 11.6,
                                          },
                                  "VC": {"name": "Concrete Vibro Compaction",
                                         "cost_eur_m3": 494.3,
                                         "co2_kg_m3": 141.2,
                                         },
                                  "JG": {"name": "Jet Grouting Columns",
                                         "cost_eur_m3": 350.8,
                                          "co2_kg_m3": 328.3
                                          },
                                  "VR": {"name": "Vibro Replacement",
                                         "cost_eur_m3": 190.1,
                                         "co2_kg_m3": 11.6,
                                         },
                                  "VR_Nguyen": {"name": "Vibro Replacement",
                                                "cost_eur_m3": 190.1,
                                                "co2_kg_m3": 11.6,
                                                },
                                  "VR_Yasser": {"name": "Vibro Replacement",
                                                "cost_eur_m3": 190.1,
                                                "co2_kg_m3": 11.6,
                                                },
        # https://hurni-gruppe.ch/wp-content/uploads/2026/01/
        # Preisliste_2026_GESAMT.pdf
        # COSTS ONLY REGARDING THE PURE MATERIAL!
        "VR_quarry": {"name": "Vibro Replacement",
                      # gebrochenes Steinbruchmaterial (sprengen + brechen)
                      "cost_eur_m3": 95.80*1.08,
                      "co2_kg_m3": 2.8*16.81/9.81,
                      },
        "VR_crushedpebble":
                    {"name": "Vibro Replacement",
                     # gebrochener Naturkies (gewinnen + brechen)
                      "cost_eur_m3": 67.30*1.08,
                      "co2_kg_m3": 2.8*17.36/9.81,
                      },
        "VR_pebble": {"name": "Vibro Replacement",
                      # Rundkies (waschen udn sieben ohne Zerkleinerung)
                      "cost_eur_m3": 67.30*1.08,
                      #Preisliste_2026_GESAMT*CHFinEUR
                      "co2_kg_m3": 1.69*18.00/9.81,
                      #Ergebnisbericht_GSV_GWP-Studie_2023.pdf*Yasser798
                      },
                                  "VR_Bouassida": {"name": "Vibro Replacement",
                                                "cost_eur_m3": 190.1,
                                                "co2_kg_m3": 11.6,
                                                },
                                  "RI": {"name": "Rigid Inclusion",
                                         "cost_eur_m3": 175.7,
                                         "co2_kg_m3": 141.2,
                                         },
                                  "RI_Nguyen": {"name": "Rigid Inclusion",
                                        "cost_eur_m3": 175.7,
                                        "co2_kg_m3": 141.2,
                                        },
                                  "RP": {"name": "Replacement Piles",
                                         "cost_eur_m3": 561,
                                         "co2_kg_m3": 248.7,
                                         },
                                  "DP": {"name": "Displacement Piles",
                                         "cost_eur_m3": 801.2,
                                         "co2_kg_m3": 248.7,
                                        },
                                  },
                  "co2": {"cost_eur_ton": 70}
                  }

    METHOD_PARAMS = {
                     'VR': {"name":      'Vibro Replacement',
                            "Model":     'MC',
                            "Eref":      (60e3, 130e3),
                            "nu":        (0.25, 0.35),
                            "gamma":     (19,   21),
                            "Rinter":    (0.8,  1.0),
                            },
                     'VR_Nguyen': {"name":   'Vibro Replacement',
                                   "Model":  'MC',
                                   "Eref":   (60e3, 130e3),
                                   "nu":     (0.25, 0.35),
                                    "gamma":  (19, 21),
                                    "Rinter": (0.8, 1.0),
                                    },
                     'VR_Yasser': {"name":   'Vibro Replacement',
                                   "Model":  'MC',
                                   "Eref":   (100e3, 100e3),
                                   "nu":     (0.30, 0.30),
                                   "gamma":  (19.0, 19.0),
                                   "Rinter": (1.0, 1.0),
                                   },
                     'RAP': {"name":     'Rammed Aggregate Piers',
                             "Model":    'MC',
                             "Eref":     (40e3, 100e3),
                             "nu":       (0.25, 0.35),
                             "gamma":    (19,   21),
                             "Rinter":   (0.8,  1.0),
                             },
                     'DSM': {"name":     'Deep Soil Mixing',
                             "Model":    'LE',
                             "Eref":     (100e3, 500e3),
                             "nu":       (0.20, 0.25),
                             "gamma":    (15,   19),
                             "Rinter":   (0.6,  0.8),
                            },
                     'JG': {"name":      'Jet Grouting Columns',
                            "Model":     'LE',
                            "Eref":      (200e3, 1000e3),
                            "nu":        (0.15,  0.25),
                            "gamma":     (17,    21),
                            "Rinter":    (0.6,   0.8),
                            },
                     'VC': {"name":      'Vibro Concrete Columns',
                            "Model":     'LE',
                            "Eref":      (10e6, 30e6),
                            "nu":        (0.15, 0.20),
                            "gamma":     (23,   25),
                            "Rinter":    (0.6,  0.8),
                            },
                     'RI': {"name":      'Rigid Inclusion',
                            "Model":     'LE',
                            "Eref":      (10e6, 30e6),
                            "nu":        (0.15, 0.20),
                            "gamma":     (23,   25),
                            "Rinter":    (0.6,  0.8),
                            },
                     'RI_Nguyen': {"name":   'Rigid Inclusion',
                                   "Model":  'LE',
                                   "Eref":   (10e6, 30e6),
                                   "nu":     (0.15, 0.20),
                                   "gamma":  (23, 25),
                                   "Rinter": (0.6, 0.8),
                                   },
                     'RP': {"name":      'Replacement Piles',
                            "Model":     'LE',
                            "Eref":      (10e6, 30e6),
                            "nu":        (0.15, 0.20),
                            "gamma":     (23,   25),
                            "Rinter":    (0.5,  0.7),
                            },
                     'DP': {"name":      'Displacement Piles',
                            "Model":     'LE',
                            "Eref":      (10e6, 30e6),
                            "nu":        (0.15, 0.20),
                            "gamma":     (23,   25),
                            "Rinter":    (0.5,  0.7),
                            }
                     }

    if   typ == 'limits':  return METHOD_LIMITS
    elif typ == 'costs':  return METHOD_COSTS
    elif typ == 'params': return METHOD_PARAMS
    else: raise ValueError(f"'{typ}' not defined within get_buildingData")

###############################################################################
###############################################################################
# COSTS
###############################################################################
###############################################################################

# =============================================================================
# Read database
# =============================================================================

def read_model_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    sections = {}
    patterns = {"slab_params":   r"SLAB_PARAMS.*?\n-+\n(.*?)\n\n",
                "slab_material": r"SLAB_MATERIAL.*?\n-+\n(.*?)\n\n",
                "si_params":     r"SI_PARAMS:\n(.*?)\n\n",
                "si_material":   r"SI_MATERIAL:\n(.*?)\n\n",
                "ltp_params":    r"LTP_PARAMS:\n(.*?)\n\n",
                "ltp_material":  r"LTP_MATERIAL:\n(.*?)\n\n",
                "soil_material": r"SOIL_MATERIAL.*?\n-+\n(.*?)\n\n"
                }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                sections[key] = ast.literal_eval(match.group(1).strip())
            except Exception:
                sections[key] = None
        else:
            sections[key] = None
    return sections

# =============================================================================
# Volume calculations
# =============================================================================

def calculate_volume_foundation(plate):
    if plate is None or plate["height"] <= 0:
        return 0.0
    area = (plate["maxx"] - plate["minx"]) * (plate["maxy"] - plate["miny"])
    volume = area * plate["height"]
    return volume

def calculate_volume_excavation(slab):
    if slab is None:
        return 0.0
    area = (slab["maxx"] - slab["minx"]) * (slab["maxy"] - slab["miny"])
    excavation_depth = abs(min(slab["UK"], 0))
    volume = area * excavation_depth
    return volume

def calculate_volume_improvement(si):
    if si is None:
        return 0.0
    single_column_volume = (math.pi * si["radius"]**2 * si["height"])
    n_columns = si["numx"] * si["numy"]
    return single_column_volume * n_columns

# =============================================================================
# Cost calculation
# =============================================================================

def calculate_project_costs(model_data, METHOD_COSTS):

    vol_slab = calculate_volume_foundation(  model_data["slab_params"] )
    vol_ltp  = calculate_volume_foundation(  model_data["ltp_params"]  )
    vol_exc  = calculate_volume_excavation(  model_data["slab_params"] )
    vol_si   = calculate_volume_improvement( model_data["si_params"]   )
    method   = model_data["si_material"].get("Method", "VR")

    cost_slab = (vol_slab * METHOD_COSTS["concrete"]["cost_eur_m3"])
    cost_ltp  = (vol_ltp  * METHOD_COSTS["ltp"]["cost_eur_m3"])
    cost_exc  = (vol_exc  * METHOD_COSTS["excavation"]["cost_eur_m3"])
    cost_si   = (vol_si   * METHOD_COSTS["improvement"][method]["cost_eur_m3"])

    co2_slab  = (vol_slab * METHOD_COSTS["concrete"]["co2_kg_m3"])
    co2_ltp   = (vol_ltp  * METHOD_COSTS["ltp"]["co2_kg_m3"])
    co2_exc   = (vol_exc  * METHOD_COSTS["excavation"]["co2_kg_m3"])
    co2_si    = (vol_si   * METHOD_COSTS["improvement"][method]["co2_kg_m3"])
    total_co2 = (co2_slab + co2_ltp + co2_exc + co2_si)

    cost_co2   = (total_co2 / 1000.0 * METHOD_COSTS["co2"]["cost_eur_ton"])
    total_cost = (cost_slab + cost_ltp + cost_exc + cost_si + cost_co2)

    return {"slab_volume_m3":       vol_slab,
            "ltp_volume_m3":        vol_ltp,
            "excavation_volume_m3": vol_exc,
            "si_volume_m3":         vol_si,
            "slab_cost_eur":        cost_slab,
            "ltp_cost_eur":         cost_ltp,
            "excavation_cost_eur":  cost_exc,
            "si_cost_eur":          cost_si,
            "co2_kg":               total_co2,
            "co2_cost_eur":         cost_co2,
            "total_cost_eur":       total_cost,
            }

# =============================================================================
# Example
# =============================================================================

def fct_calculateCost(filepath, METHOD_COSTS):
    filename = "input_parameters.txt"
    model_data = read_model_file(os.path.join(filepath, filename))
    results = calculate_project_costs(model_data, METHOD_COSTS)
    return results["total_cost_eur"], results

###############################################################################
###############################################################################
# PLOTS
###############################################################################
###############################################################################

def plot_parameterVariation(path):
    _, colors = fct_setPlotstyle(flat=True, r=1)
    _, cost_cmap = fct_setPlotstyle(flat=True, r=1, colorMap=True)
    # cost_color = colors[0]   # purple
    # g_color    = colors[1]   # turquoise
    # best_color = colors[4]   # red
    # path_color = colors[5]   # grey
    # Creates 3D scatter plots showing the parameter variation.
    history = pd.read_csv(os.path.join(path,
                                       "optimization_history.csv"),
                          sep=';')
    history = history.loc[history["status"]
                          .astype("string").str.strip()
                          .eq("OK").fillna(False)].reset_index(drop=True)
    if history.empty:
        print("No optimization results with status OK found.")
        return
    g_cols = [col for col in history.columns if col.startswith("g")]
    # history = history[(history["cost"] < 1e11)
    #                   & (~history["status"].str.contains("ERROR",
    #                                                      na=False))]
    if len(history) == 0:
        print("No valid optimization results found.")
        return
    history["g_sum"] = (history[g_cols]
                        .apply(pd.to_numeric, errors="coerce")
                        .sum(axis=1,
                             min_count=len(g_cols))
                        )
    fig = plt.figure(figsize=(16, 7))
    # Plot 1 - Cost
    ax1 = fig.add_subplot(121, projection="3d")
    sc1 = ax1.scatter(history["si_R"],
                      history["si_L"],
                      history["cost"],
                      c=history["cost"],
                      cmap=cost_cmap,
                      s=45,
                      )
    ax1.plot(history["si_R"],
             history["si_L"],
             history["cost"],
             color="grey",
             alpha=0.35,
             linewidth=1,
             )
    ax1.set_xlabel("SI Radius $r$ [m]")
    ax1.set_ylabel("SI Length $L$ [m]")
    ax1.set_zlabel("Cost")
    ax1.set_title("Optimization Landscape (Cost)")
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.08)
    cbar1.set_label("Cost")
    # Plot 2 - Constraint values
    ax2 = fig.add_subplot(122,
                          projection="3d"
                          )
    sc2 = ax2.scatter(history["si_R"],
                      history["si_L"],
                      history["g_sum"],
                      c=history["g_sum"],
                      cmap=cost_cmap,
                      s=45,
                      )
    ax2.plot(history["si_R"],
             history["si_L"],
             history["g_sum"],
             color="grey",
             alpha=0.35,
             linewidth=1,
             )
    ax2.set_xlabel("SI Radius $r$ [m]")
    ax2.set_ylabel("SI Length $L$ [m]")
    ax2.set_zlabel(r"$\sum g_i$")
    ax2.set_title("Optimization Landscape (Constraint Sum)")
    cbar2 = fig.colorbar(sc2, ax=ax2, pad=0.08)
    cbar2.set_label(r"$\sum g_i$")
    # SAVE
    plt.tight_layout()
    save_path = fct_formatSave(True, fig, path,
                               'optimization_parameter_variation')
    plt.close()
    print(save_path)
    return save_path

def plot_optiDistribution(path, plot = [0,1,2]): # take 1
    _, colors = fct_setPlotstyle(flat=True, r=1)
    _, cost_cmap = fct_setPlotstyle(flat=True, r=1, colorMap=True)
    cost_color = colors[0]   # purple
    g_color    = colors[1]   # turquoise
    best_color = colors[4]   # red
    # path_color = colors[5]   # grey
    filenames = []
    history = pd.read_csv(os.path.join(path,
                                       "optimization_history.csv"),
                          sep=';')
    g_cols = [col for col in history.columns if col.startswith("g")]
    if 0 in plot:
        df = pd.DataFrame(history)
        eval_id = np.arange(len(df))
        fig, axes = plt.subplots(1, len(df.columns),
                                 figsize=(len(df.columns)*1, 10),
                                 sharey=True)
        for ax, col in zip(axes, df.columns):
            sc = ax.scatter(df[col],
                            eval_id, # E_i vs. i
                            c=df["cost"],
                            cmap=cost_cmap,
                            s=8, alpha=0.7)
            ax.set_xlabel(col)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Evaluation")
        cbar = fig.colorbar(sc, ax=axes)
        cbar.set_label("Cost [EUR]")
        fig.suptitle("Parameter Evolution during Optimization")
        save_path = fct_formatSave(True, fig, path,
                                   'optimization_parameter_evolution')
        print(f"Plot saved to: {save_path}")
    if 1.2 in plot:
        # cost_history = [h["cost"] for h in history]
        cost_history = history["cost"].tolist()
        cost_history_clean = []
        for c in cost_history:
            if c is None:
                continue
            if np.isfinite(c) and c < 1e11:   # filtert Dummy-Strafen raus
                cost_history_clean.append(c)
        fig = plt.figure(figsize=(10, 5))
        plt.plot(cost_history_clean, linewidth=1)
        plt.xlabel("Evaluation / Iteration")
        plt.ylabel("Total Cost [EUR]")
        plt.title("Cost Minimization during Optimization")
        plt.grid(True)
        best_idx = np.argmin(cost_history_clean)
        plt.scatter(best_idx, cost_history_clean[best_idx],
                    color=best_color,
                    label="Best solution")
        save_path = fct_formatSave(True, fig, path,
                                   'optimization_cost_minimization')
        print(f"Plot saved to: {save_path}")
    if 1 in plot:
        df_cost = history[["cost"] + g_cols].copy()
        for col in df_cost.columns:
            df_cost[col] = pd.to_numeric(df_cost[col], errors="coerce")
        eval_id = np.arange(1, len(df_cost) + 1)
        # Dummy-Strafkosten nicht als reguläre Kosten darstellen
        valid_cost = np.isfinite(df_cost["cost"]) & (df_cost["cost"] < 1e11)
        g_sum = df_cost[g_cols].sum(axis=1, min_count=len(g_cols))
        valid_g = np.isfinite(g_sum)
        fig, ax_cost = plt.subplots(figsize=(10, 5))
        ax_g = ax_cost.twinx()
        # Zweite y-Achse ebenfalls links platzieren
        ax_g.spines["right"].set_visible(False)
        ax_g.spines["left"].set_position(("outward", 65))
        ax_g.spines["left"].set_visible(True)
        ax_g.yaxis.set_label_position("left")
        ax_g.yaxis.tick_left()
        line_cost, = ax_cost.plot(eval_id[valid_cost],
                                  df_cost.loc[valid_cost, "cost"],
                                  color     = cost_color,
                                  linewidth = 1.5,
                                  label     = "Total cost",
                                  )
        line_g, = ax_g.plot(eval_id[valid_g],
                            g_sum[valid_g],
                            color     = g_color,
                            linewidth = 1.5,
                            label     = "Objectives",
                            )
        if valid_cost.any():
            best_idx = df_cost.loc[valid_cost, "cost"].idxmin()
            ax_cost.scatter(best_idx + 1,
                            df_cost.loc[best_idx, "cost"],
                            color=best_color,
                            zorder=5,
                            label="Best solution")
        ax_cost.set_xlabel("Evaluation")
        ax_cost.set_ylabel("Costs [EUR]",   color=cost_color)
        ax_g.set_ylabel(r"Penalty",         color=g_color)
        ax_cost.tick_params(axis="y",       colors=cost_color)
        ax_g.tick_params(axis="y",          colors=g_color)
        ax_cost.set_title("Cost and constraint development during optimization")
        ax_cost.grid(True, alpha=0.3)
        handles, labels = ax_cost.get_legend_handles_labels()
        handles.append(line_g)
        labels.append(line_g.get_label())
        ax_cost.legend(handles, labels,
                       loc="upper left",
                       bbox_to_anchor=(1.02, 1.0),
                       frameon=False,
                       )
        fig.subplots_adjust(left=0.23, right=0.75)
        save_path = fct_formatSave(True, fig, path,
                                   'optimization_cost_minimization')
        print(f"Plot saved to: {save_path}")
    if 2 in plot:
        from pandas.plotting import parallel_coordinates
        # values for the plot
        cols = ["slab_t", "ltp_t", "si_num",
                "si_L", "si_R", "si_E", "cost",
                ] + g_cols
        df_plot = history[cols].copy()
        for col in cols:
            df_plot[col] = pd.to_numeric(df_plot[col],
                                         errors="coerce")
        df_plot = df_plot.dropna()
        if df_plot.empty:
            print("No complete numeric rows for parallel coordinates.")
        else:
            for col in cols:
                xmin, xmax = df_plot[col].min(), df_plot[col].max()
                df_plot[col] = ((df_plot[col] - xmin) / (xmax - xmin)
                                if xmax > xmin else 0.0)
            # parallel_coordinates benötigt eine Gruppenspalte.
            df_plot["group"] = "Evaluations"
            fig = plt.figure(figsize=(16, 8))
            parallel_coordinates(df_plot,
                                 class_column="group",
                                 color=["steelblue"],
                                 alpha=0.15,
                                 linewidth=0.8,)
            plt.title("Parallel Coordinates Plot (Optimization)")
            plt.ylabel("Normalized value")
            plt.grid(alpha=0.3)
            save_path = fct_formatSave(True, fig, path,
                                       'optimization_parallel_coordinates')
            print(f"Plot saved to: {save_path}")
    return filenames

unit_map = {"diameter":    "m",
            "spacing":     "m",
            "ltp":         "m",
            "gamma":       "kN/m³",
            "nu":          "-",
            "Rinter":      "-",
            "stiffness":   "kPa",
            "Eref":        "kPa",
            "cost_eur_m3": "€/m³",
            "co2_kg_m3":   "kg/m³",}

def format_valueUnit(prop, v,
                     unit_map=unit_map): # not used
    """auto scaling for stiffness etc."""
    if prop == "stiffness":
        if   v >= 1e6: return v / 1e6, "MPa"
        elif v >= 1e3: return v / 1e3, "kPa"
        else:          return v,       "Pa"
    return v, unit_map.get(prop, "")

# Display labels only: input keys and calculation data stay unchanged.
METHOD_ABBREVIATIONS = {"SSI":  "SSI",
                        "PVD":  "PVD",
                        "DC":   "DC",
                        "RIC":  "RIC",
                        "WIP":  "WIP",
                        "BD":   "BD",
                        "CMC":  "CMC",
                        "MRM":  "MRM",
                        "VC":   "CVC",
                        "CVC":  "CVC",
                        "RI":   "RI",
                        "DSM":  "DSM",
                        "RAP":  "RAP",
                        "JG":   "JG",
                        "CEM":  "CEM",
                        "CG":   "CG",
                        "LMG":  "LMG",
                        "AGF":  "AGF",
                        "VR":   "VR",
                        "VSC":  "VSC",
                        "CFA":  "CFA",
                        "RP":       "ReP",
                        "REP":      "ReP",
                        "DP":       "DiP",
                        "DIP":      "DiP",
                        "KPP":      "KPP",
                        "ASIRI":    "ASIRI",
                        }
PROPERTY_LABELS = {"diameter":      "Diameter",
                   "spacing":       "Spacing",
                   "ltp":           "LTP thickness",
                   "stiffness":     r"$E_{\mathrm{SI}}$",
                   "Eref":          r"$E_{\mathrm{SI}}$",
                   "nu":            r"$\nu$",
                   "gamma":         r"$\gamma$",
                   "Rinter":        r"$R_{\mathrm{inter}}$",
                   "cost_eur_m3":   "costs",
                   "co2_cost_eur":  "costs",
                   "co2_kg_m3":     r"$\mathrm{CO}_{2}$",
                   }

# co2_cost_eur is a total monetary amount, not a cost per volume.
unit_map["co2_cost_eur"] = "€"

def format_methodLabel(method):
    """Display only the thesis abbreviation, including for study variants."""
    base = str(method).split("_", 1)[0]
    return METHOD_ABBREVIATIONS.get(base.upper(), base)


def plot_methodRanges(data,
                      properties,
                      unit_map=unit_map,
                      savePath=None):
    # Exclude study/material variants, e.g. VR_Nguyen or VR_quarry.
    methods = [m for m in data if "_" not in str(m)]
    if not methods:
        return None
    _, cmap = fct_setPlotstyle(flat=True, r=1, colorMap=True)
    fig, axes = plt.subplots(1, len(properties),
                             figsize=(15 / 2.54,
                                      4 / 2.54),
                             sharey=True) #  same y axis
    if len(properties) == 1: axes = [axes]
    method_names = {m: format_methodLabel(m) for m in methods}
    norms = {}
    for ax, prop in zip(axes, properties):
        unit = unit_map.get(prop, "")
        label = PROPERTY_LABELS.get(prop, prop.replace("_", " "))
        ax.set_xlabel(f"{label} [{unit}]" if unit else label)
        use_log = (prop in ["stiffness","Eref"])
        values = []
        for y, method in enumerate(methods):
            if prop not in data[method]:
                continue
            val = data[method][prop]
            if isinstance(val, tuple):
                values.extend([v for v in val
                               if isinstance(v, (int, float))])
            elif isinstance(val, (int, float)):
                values.append(val)
        if not values:
            continue
        vmin, vmax = min(values), max(values)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        norms[prop] = norm
        for y, m in enumerate(methods):
            if prop not in data[m]:
                continue
            val = data[m][prop]
            # missing values
            if val is None:
                # ax.scatter(0, y, color="lightgrey", s=20)
                continue
            # extract range
            if isinstance(val, tuple):
                if not all(isinstance(v, (int, float)) for v in val):
                    ax.scatter(0, y, color="lightgrey", s=20)
                    continue
                x1, x2 = val
            elif isinstance(val, (int, float)):
                x1 = x2 = val
            else:
                # ax.scatter(0, y, color="lightgrey", s=20)
                continue
            # colored lines
            # ax.plot([x1, x2], [y, y], lw=3,
            #         color=cmap(norm(0.5 * (x1 + x2))))
            # ax.plot([x1, x1], [y - 0.2, y + 0.2],
            #         lw=2, color=cmap(norm(x1)))
            # ax.plot([x2, x2], [y - 0.2, y + 0.2],
            #         lw=2, color=cmap(norm(x2)))
            # colored lines
            ax.plot([x1, x2], [y, y],                color='k')
            ax.plot([x1, x1], [y - 0.15, y + 0.15], color='k')
            ax.plot([x2, x2], [y - 0.15, y + 0.15], color='k')
        if use_log and all(v > 0 for v in values): ax.set_xscale("log")
        ax.grid(alpha=0.3)
    # y labels with thesis abbreviations
    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels([method_names[m] for m in methods])
    # plot
    plt.tight_layout()
    if savePath is None:
        plt.show()
        filename = None
    else:
        label = ""
        for p in properties:
            part = p[:4]
            part = part.capitalize()
            label += part
        filename = fct_formatSave(True, fig, savePath,
                                  f'methodRanges_{label}')
        plt.close(fig)
    return filename

if __name__ == "__main__":
    device = 27
    _, _, path = open_plaxisData(device, 'opt', 'opti_methodRanges')
    plot_methodRanges(get_buildingData('limits'),
                      ['diameter','ltp','stiffness'],
                      savePath = path)
    plot_methodRanges(get_buildingData('costs')['improvement'],
                      ['cost_eur_m3','co2_kg_m3'],
                      savePath = path)
    plot_methodRanges(get_buildingData('params'),
                      ['Eref','nu','gamma','Rinter'],
                      savePath = path)