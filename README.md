# Numerical Modelling and Optimisation of Soil Improvement

Python framework for automated numerical modelling, evaluation and design optimisation of soil improvement using PLAXIS.

This repository accompanies a master's thesis on numerical modelling and optimisation of soil improvement. It is intended to provide the supporting Python code and selected figures, including reference-model comparisons, parameter studies and optimisation results.

The workflow couples PLAXIS calculations with Python-based model generation, result evaluation and optimisation. Design variants are assessed in terms of settlement and other specified constraints, with material costs used as an optimisation objective. The optimisation studies include Genetic Algorithm (GA), Differential Evolution (DE) and Particle Swarm Optimisation (PSO).

## Structure

| Path | Contents |
| --- | --- |
| `code/` | Python scripts for modelling, evaluation, optimisation and plotting |
| `graphics/` | Selected figures and output plots |
| `requirements.txt` | Identified third-party Python dependencies |
| `LICENSE` | MIT License |

**Status:** This initial repository contains the documentation and folder structure. The final code and figures will be added separately.

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

`numpy`, `pandas`, `matplotlib`, `scipy` and `pymoo` — see `requirements.txt`.

This provisional list is based on the available optimisation and plotting modules, including imports inside functions. The optimisation source specifies `pymoo >= 0.6`. Other package versions have not been verified against a complete working environment and are therefore not pinned.

The model-generation, evaluation and shared calling modules must also be provided in `code/`. Their additional dependencies and the PLAXIS scripting setup still need to be checked against the final source files. Python standard-library modules and project-local modules are not pip requirements.

## Citation

Thesis title, final author details, year and permanent publication link will be added when the thesis is available. Please cite the thesis when using the framework or its results.

## License

The original code in this repository is released under the MIT License; see [LICENSE](LICENSE). PLAXIS and third-party dependencies retain their respective licences. Third-party figures or data, if added, require their own attribution and licence information.
