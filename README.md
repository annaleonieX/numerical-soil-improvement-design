# Numerical Modelling and Optimisation of Soil Improvement

Python framework for automated numerical modelling, evaluation and design optimisation of soil improvement using PLAXIS.

This repository accompanies a master's thesis on numerical modelling and optimisation of soil improvement. It is intended to provide the supporting Python code and selected figures, including reference-model comparisons, parameter studies and optimisation results.

The workflow couples PLAXIS calculations with Python-based model generation, result evaluation and optimisation. Design variants are assessed in terms of settlement and other specified constraints, with material costs used as an optimisation objective. The optimisation studies include Genetic Algorithm (GA), Differential Evolution (DE) and Particle Swarm Optimisation (PSO).

## Structure

Figures are organised in folders corresponding to the respective thesis chapters and studies.

```text
├── README.md           - Repository overview and documentation
├── LICENSE             - MIT License
├── requirements.txt    - Python dependencies
├── code/               - Python scripts for modelling, evaluation, optimisation and plotting
├── chap4_pro/          - Figures for Chapter 4
├── chap5_sen_base/     - Figures for the base-model sensitivity studies
├── chap5_sen_simpl/    - Figures for the simplified-model sensitivity studies
├── chap6_opt_studies0/ - Figures for optimisation study group 0
├── chap6_opt_studies1/ - Figures for optimisation study group 1
└── chap6_opt_studies2/ - Figures for optimisation study group 2
```

## Code organisation

The framework is organised into modules for workflow control, model generation, result evaluation, plotting and optimisation. The internal functions and their interactions are described in greater detail in the accompanying master's thesis.

```text
code/
└── plaxis_00calling_01main.py          - Main workflow configuration and execution
    ├── plaxis_00calling_02helpers.py  - Shared utilities and workflow support
    ├── plaxis_01model_01main.py       - PLAXIS model generation and calculation
    │   ├── plaxis_01model_02helpers.py   - Supporting model-generation functions
    │   └── plaxis_01model_03refstudy.py  - Reference-study configurations
    ├── plaxis_02eval_01main.py        - Result extraction and evaluation
    │   ├── plaxis_02eval_02helpers.py    - Supporting evaluation functions
    │   └── plaxis_02eval_03plotter.py    - Plots of numerical results
    ├── plaxis_03plot_01main.py        - Comparative plots and parameter studies
    │   └── plaxis_03plot_02helpers.py    - Plotting, formatting and labelling utilities
    └── plaxis_04optimization_00calling.py - Optimisation-study configuration
        ├── plaxis_04optimization_01main.py    - Optimisation and candidate evaluation
        ├── plaxis_04optimization_02helpers.py - Method parameters and material costs
        └── plaxis_04optimization_03plots.py   - Optimisation results and convergence plots
```

The hierarchy illustrates the functional organisation of the modules. All Python files are stored directly in `code/`.

## Setup

Create a virtual environment using a Python version compatible with your PLAXIS scripting installation.

```bash
python -m venv .venv
```

Activate it on Windows (Command Prompt):

```bat
.venv\Scripts\activate
```

For a standalone post-processing environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the identified Python dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Running numerical calculations additionally requires a licensed PLAXIS installation and its compatible Python scripting interface. These are not included in this repository or installed by `requirements.txt`.

Before running the released scripts, configure the local PLAXIS connection, project paths, input parameters and output directories. Keep connection passwords outside the public repository. Detailed entry points and the tested software versions will be documented with the final code release.

## Dependencies

`numpy`, `pandas`, `matplotlib`, `scipy`, `pymoo` and `plxscripting` — see `requirements.txt`.

This provisional list is based on the available optimisation and plotting modules, including imports inside functions. The optimisation source specifies `pymoo >= 0.6`. Other package versions have not been verified against a complete working environment and are therefore not pinned.

The model-generation, evaluation and shared calling modules must also be provided in `code/`. Their additional dependencies and the PLAXIS scripting setup still need to be checked against the final source files. Python standard-library modules and project-local modules are not pip requirements.

## Citation

The framework and results presented in this repository are based on the following master's thesis:
*Salzmann, A. (2026). Numerical Optimisation of Geotechnical Structures. Master's thesis, Graz University of Technology (TU Graz)*.
Available at: [TU Graz Repository](https://online.tugraz.at/tug_online/wbAbs.showThesis?pThesisNr=100281&pOrgNr=953).

Please cite the thesis when using the framework, code or results presented in this repository.

## License

The original code in this repository is released under the MIT License; see [LICENSE](LICENSE). PLAXIS and third-party dependencies retain their respective licences. Third-party figures or data, if added, require their own attribution and licence information.
