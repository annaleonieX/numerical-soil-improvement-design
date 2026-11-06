# -*- coding: utf-8 -*-
'''
Created on Wed Apr 29 16:31:35 2026
@author: annaleoniesalzmann
'''

###############################################################################
#---------------------------------- IMPORTS -----------------------------------
###############################################################################

import os
import re
import numpy as np
import pandas as pd

def open_plaxisData(device, dimension, pathFolder, create=True): # USERDEFINED!
    # Plaxis Input -> Expert -> Python -> Editor -> Start server
    if device == 0: # privat
        pathGeneral = r'C:\Users\annal\Downloads'
        port_pw_2D  = r'/QW^>cfc%sf$GWP2'
        port_pw_3D  = r'cMH<DA2L6<ACnGua'
    elif device == 27: # university OFFICE
        pathGeneral = r'D:\SalzmannAnnaLeonie_2026'
        # pathGeneral = r'C:\Users\dreambigworkhard\Desktop\MA_UnitCell'
        port_pw_2D  = r'6iL$NP6~eDZ1@Xy!'
        port_pw_3D  = r'XwRm%WH/Au7e9VzG'
    elif device == 89: # university HANNES' PC
        pathGeneral = r'C:\Users\dreambigworkhard\Desktop\MA_SoilImprov_PC3'
        port_pw_2D  = r'Y5zTtBfZffkC@DrX'
        port_pw_3D  = r'iCg8^4zrF=!t79!n'
    else: raise ValueError(f'Device {device} is not defined.')
    if dimension == 'opt':
        port_ip = None
        port_pw = None
        path = os.path.join(pathGeneral,
                            'Optimization - 2026-09-28 - Optimization Study',
                            pathFolder)
        if create: os.makedirs(path, exist_ok=True)
    else:
        path = os.path.join(pathGeneral, pathFolder)
        if   dimension.endswith('2'):
            port_ip = 10000
            port_pw = port_pw_2D
        elif dimension.endswith('3'):
            port_ip = 10002
            port_pw = port_pw_3D
        elif dimension == 'spacer':
            port_ip = None
            port_pw = None
        else: raise ValueError(f'Dimension {dimension} not defined!')
    return port_ip, port_pw, path
# port_ip, port_pw, path = open_plaxisData(device, dimension, pathFolder)
# port_ip, port_pw, path = open_plaxisData(1, 'unitcell2', 'Optimization v2')

def close_plaxisOutput():
    import psutil
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] in ['Plaxis2DXOutput.exe',
                                 'Plaxis3DOutput.exe']:
            print('Shutdown of Plaxis-Output-Window.')
            proc.kill()

def close_plaxis10001():#old
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 10001))
    print("Port 10001 open:", result == 0)
    sock.close()

def close_plaxis(number):#new
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', number))
    print(f"Port {number} open:", result == 0)
    sock.close()

###############################################################################
#------------------------------------ Main ------------------------------------
###############################################################################

def fct_mathLabel(text):
    if "_" in text:
        parts = text.split("_")
        base = parts[0]
        subs = ",".join(parts[1:])
        return rf"${base}_{{{subs}}}$"
    return text

def fct_adjustFontForPrint(fig,
                           width_cm    = 15,
                           min_font_pt = 7):
    from matplotlib.text import Text
    pad_inches = 0.1
    target_width_in = width_cm / 2.54
    for _ in range(8):
        fig.canvas.draw()
        bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        pdf_width_in = bbox.width + 2 * pad_inches
        scale = target_width_in / pdf_width_in
        texts = [obj
                 for obj in fig.findobj(match=Text)
                 if obj.get_visible() and obj.get_text()
                 ]
        if not texts: return
        smallest_print_font = min(obj.get_fontsize() * scale
                                  for obj in texts)
        if smallest_print_font >= min_font_pt - 0.05:
            return
        factor = min_font_pt / smallest_print_font
        for obj in texts:
            obj.set_fontsize(obj.get_fontsize() * factor)

def fct_formatSave(save, fig, path, name,
                   pad=None):
    import matplotlib.pyplot as plt
    if not save:
        plt.show()
        return 'shown'
    else:
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f'Plot_{name}.pdf')
        if pad is None:
            ### for thesis:
            fig.savefig(filename)
            ### for bigger writting:
            # fct_adjustFontForPrint(fig)
            # filename = os.path.join(path, f'Plot_{name}_SIZE7.pdf')
            # fig.savefig(filename, bbox_inches='tight')
            ### for powerpoint:
            # filename = os.path.join(path, f'Plot_{name}.jpg')
            # fig.savefig(filename,
            #             bbox_inches='tight',
            #             bbox_extra_artists=fig.legends) #eliminate cutten
        else:
            fig.savefig(filename,
                        bbox_inches="tight",
                        pad_inches=pad)
        plt.close(fig)
        return filename

def fct_setPlotstyle(case=0, flat=False, r=4, colorMap=False):
    import matplotlib as mpl
    import matplotlib.colors as mcolors
    markers = [('.', 4), ('o', 3), ('s', 3),
               ('^', 3.5), ('v', 3.5), ('D', 3),
               ('*', 4.5), ('X', 3.5), ('P', 3.5),
               ('H', 3.5), ('+', 4)]
    if case in [0]: # same colors as used in the svg files
        # purple | turquoise | green | yellow | red | gray
        base_colors = ['#c85aa6', '#88ccc6', '#c1ce00',
                       '#fcc94e', '#ea4c5f', '#b7b8b7']
        alphas = np.linspace(1, 0.2, r)
        colors = []
        for c in base_colors:
            colors.append([mcolors.to_rgba(c, a) for a in alphas])
        if flat is True:
            colors = [dist_color
                      for list_color in colors
                      for dist_color in list_color]
    else:
        # n = total possible colors = basecolor*r
        # r = different shades per color (hardcoded)
        if   case == 1:  colorMap = mpl.colormaps['Pastel1']; n=9;  r=1
        elif case == 2:  colorMap = mpl.colormaps['Pastel2']; n=8;  r=2
        elif case == 3:  colorMap = mpl.colormaps['Paired'];  n=12; r=2
        elif case == 4:  colorMap = mpl.colormaps['Accent'];  n=8;  r=1
        elif case == 5:  colorMap = mpl.colormaps['Dark2'];   n=8;  r=1
        elif case == 6:  colorMap = mpl.colormaps['Set1'];    n=9;  r=1
        elif case == 7:  colorMap = mpl.colormaps['Set2'];    n=8;  r=1
        elif case == 8:  colorMap = mpl.colormaps['Set3'];    n=12; r=1
        elif case == 9:  colorMap = mpl.colormaps['tab10'];   n=10; r=1
        elif case == 10: colorMap = mpl.colormaps['tab20'];   n=20; r=2
        elif case == 11: colorMap = mpl.colormaps['tab20b'];  n=20; r=4
        elif case == 12: colorMap = mpl.colormaps['viridis']; n=12; r=1
        elif case == 13: colorMap = mpl.colormaps['plasma'];  n=12; r=1
        else:            colorMap = mpl.colormaps['tab20c'];  n=20; r=4
        colors = colorMap(np.linspace(0, 1, n))
        if flat is False:
            colors = [colors[i:i+4] for i in range(0, n, r)]
    if colorMap:
        from matplotlib.colors import LinearSegmentedColormap
        colors = np.array(colors)
        if colors.ndim != 2 or colors.shape[1] not in [3, 4]:
            raise ValueError(f"Invalid color shape: {colors.shape}")
        colors = colors.tolist()  # list of RGBA tuples
        colors = LinearSegmentedColormap.from_list("my_cmap", colors, N=256)
    return (markers, colors)

def fct_sortKey(name):
    import re
    s = name.lower()
    if   "coarse" in s and "medium" not in s:                      quality = 0
    elif "mediumcoarse" in s:                                      quality = 1
    elif "medium" in s and "coarse" not in s and "fine" not in s:  quality = 2
    elif "finemedium" in s:                                        quality = 3
    elif "fine" in s and "medium" not in s:                        quality = 4
    else:                                                          quality = 5
    if s.endswith("full3"):                                        geom = 0
    elif s.endswith("quarter3"):                                   geom = 1
    elif s.endswith("unitcell3"):                                  geom = 2
    elif s.endswith("unitcell2"):                                  geom = 3
    else:                                                          geom = 99
    # natural = [int(t)
    #            if t.isdigit() else t
    #            for t in re.split(r"(\d+)", s)]
    numbers = re.findall(r"\d+(?:\.\d+)?", s)
    if numbers:
        natural = tuple(float(n)
                        for n in numbers)
    else:
        natural = (s,)
    return (quality, geom, natural)

if __name__ == "__main__":
    lst = ["model_coarse_unitcell2",
           "model_medium_quarter3",
           "modell_fine_quarter3",
           "model_mediumcoarse_full3",
           "model_finemedium_unitcell2",
           "radius_6_full3",
           "radius_1.5_full3",
           "radius_2_full3",
           "modell_fine_full3"]
    print(sorted(lst, key=fct_sortKey))

def fct_getPhases(phase_ref, n_phases, accuracy, diff, csv_path=None):
    phase_max = phase_ref + diff + n_phases
    if accuracy == 'full':
        # The entire file is recalculated and overwritten with all phases
        phases = np.arange(phase_ref, phase_max, 1)
    elif accuracy == 'join':
        # Continue from already analyzed phases (resume functionality)
        if csv_path is None:
            raise ValueError('Path needs to be defined')
        all_phases = np.arange(phase_ref, phase_max, 1)
        print(f'All phases are {all_phases}.')
        if os.path.exists(csv_path):
            Fsu_phases_df = pd.read_csv(csv_path, sep=';')
            if (not Fsu_phases_df.empty and
                'Phase' in Fsu_phases_df.columns):
                existing_phases = Fsu_phases_df['Phase'].values
                phases = np.array([phaseNum for phaseNum in all_phases 
                                   if phaseNum not in existing_phases])
                # print(existing_phases)
                # print(phases)
            else:
                phases = all_phases
        else:
            Fsu_phases_df = pd.DataFrame()
            phases = all_phases
    elif accuracy == 'bare':
        # Recompute only the first and last load phase + phi c reduction
        phases = [phase_ref, phase_max-2, phase_max-1]
    else:
        raise ValueError(f'Accuracy {accuracy} not defined.')
    print(f'Necessary phases are {phases}.')
    return phases

def fct_writeRuntime(path, runtime, section='0', accuracy='full'):
    # Write or update a runtime section in input_parameters.txt.
    # If the section (e.g. RUNTIME_MODEL) already exists, it will be replaced.
    # Otherwise, it will be appended to the file.
    file_path = os.path.join(path, 'input_parameters.txt')
    section_name = f'RUNTIME_{section.upper()}'
    separator = '-' * 58
    # --- Read existing file (if it exists) ---
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
    else:   lines = []
    new_lines = []
    i = 0
    existing_runtime = None
    # --- Loop through file and remove existing section ---
    while i < len(lines):
        line = lines[i].strip()
        # If the current line matches the section header → remove block
        if line == section_name:
            i += 1
            # Skip separator line (-----)
            if i < len(lines) and set(lines[i].strip()) == {'-'}: i += 1
            # parse block content
            while i < len(lines) and lines[i].strip() != '':
                if "runtime_seconds" in lines[i]:
                    try: existing_runtime = float(
                        re.findall(r"[-+]?\d*\.\d+|\d+",
                                   lines[i]
                                   )[0]
                        )
                    except Exception: pass
                i += 1
            # Skip all lines until an empty line is reached
            while i < len(lines) and lines[i].strip() != '': i += 1
            continue
        # Keep all other lines
        new_lines.append(lines[i])
        i += 1
    if accuracy == "join" and existing_runtime is not None:
        runtime = existing_runtime + runtime
    # --- Create new runtime block ---
    new_block = [f'\n{section_name}\n',
                 f'{separator}\n',
                 f'runtime_seconds = {runtime:.2f}\n',
                 f'runtime_minutes = {runtime/60:.2f}\n']
    # Append new block at the end
    new_lines.extend(new_block)
    # --- Optional: clean multiple empty lines ---
    cleaned_lines = []
    for line in new_lines:
        if not (cleaned_lines and
                cleaned_lines[-1].strip() == '' and
                line.strip() == ''):
            cleaned_lines.append(line)
    # --- Write updated content back to file ---
    with open(file_path, 'w') as f:
        f.writelines(cleaned_lines)
    return file_path

###############################################################################
#----------------------------------- HELPER -----------------------------------
###############################################################################

# datasets = ['plaxis_01model_01main', 'plaxis_01model_02helpers',
#             'plaxis_02eval_01main',  'plaxis_00calling_02helpers',
#             'plaxis_03plot_01main',  'plaxis_03plot_02helpers']
# keywords = [('region',"contact_force","g_oEnd"), # from to
#             ('first',"contact_force"), # at line
#             ('all',"Soil improvement interface")]

def fct_getLines(datasets, keywords):
    for data in keywords:
        typ = data[0]
        keyword = data[1]
        output_file = f"result_keyword_{keyword}_{typ}.txt"
        with open(output_file, "w", encoding="utf-8") as out:
            for dataset in datasets:
                lines = []
                with open(f"{dataset}.py", "r", encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        if keyword in line:
                            lines.append(i)
                            if typ == "first": break
                            if typ == "region" and "g_oEnd" in line: break
                if lines:
                    if typ == "first":
                        sentence = f"Line {lines[0]} from Code {dataset}"
                    elif typ == "region":
                        sentence = f"Line {lines[0]} to Line {lines[-1]} from Code {dataset}"
                    elif typ == "all":
                        formatted = ", ".join(map(str, lines[:-1]))
                        if len(lines) > 1: formatted += f" and {lines[-1]}"
                        else:              formatted = str(lines[0])
                        sentence = f"in Line {formatted} from Code {dataset}"
                    else: sentence = ""
                    out.write(sentence + "\n")