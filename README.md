# CelerisAi

<p align="center">
  <img src="https://github.com/wrenteria/CelerisAi/blob/main/docs/source/images/CelerisAILogo.png" alt="CelerisAI Logo" width="400">
</p>

<p align="center">
    <img src="https://img.shields.io/github/license/wrenteria/CelerisAi" alt="Github License">
    <img src="https://img.shields.io/github/stars/wrenteria/CelerisAi" alt="Github stars">
    <img src="https://img.shields.io/github/forks/wrenteria/CelerisAi" alt="Github forks">
</p>

CelerisAi is a Python-[Taichi](https://github.com/taichi-dev/taichi)-based software designed for nearshore wave modeling. This solver offers high-performance simulations on various hardware platforms and seamlessly integrates with machine learning and artificial intelligence environments. The solver leverages the flexibility of Python for customization and interoperability, while Taichi's high-performance parallel programming capabilities ensure efficient computations.

## Key Features
* High Performance: CelerisAi delivers efficient simulations on a wide range of hardware, from personal laptops to high-performance computing clusters.
* Machine Learning Integration: The solver's seamless integration with AI frameworks empowers users to develop hybrid models and leverage data-driven approaches.
* Flexibility and Customization: CelerisAi's Python-based implementation provides a high degree of customization and adaptability to meet specific modeling needs.
* Interoperability: The solver's compatibility with various Python libraries and tools enhances its versatility and integration capabilities.

## Applications
CelerisAi can be applied to a variety of coastal engineering problems, including:
- Wave propagation modeling
- Coastal hydrodynamics
- Tsunami simulation
- Inverse problems (e.g., depth inversion)
- Assimilation problems

## Quick Start
Use the editable install so the `CelerisAI` package is available from any working directory.

```bash
git clone https://github.com/wrenteria/CelerisAi.git
cd CelerisAi
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Wet/dry scheme (BGC fork)

`Solver(..., wetdry_scheme="legacy" | "conserving")`, default `"legacy"`, which
reproduces upstream CelerisAi bit for bit. `"conserving"` changes only the
treatment of wet/dry faces:

- a dry cell carries no face depth, so the flat `eta = bed` reconstruction no
  longer puts phantom water on the face that looks down onto a lower neighbour;
- a wet surface standing above a dry neighbour's bed drives a flux into it
  (gravity-driven wetting); a wet/dry face without wetting is a wall for mass
  and momentum (legacy drains momentum into the dry cell, a numerical wall drag);
- fully dry cells keep `eta = bed` instead of `eta = 0`, so terrain below the
  datum no longer fills with water;
- sub-`delta` water is kept rather than reset to the bed, and isolated wet cells
  are not deleted, so mass is conserved;
- the explicit Manning friction rate is capped at `0.5 / dt` in thin films.

Validation: `examples/validation/breach` (TELEMAC-2D hydrograph routing and dyke
overtopping) and `examples/validation/malpasset` (1959 dam break) both take
`--wetdry conserving`; `tests/test_hydrograph.py` pins the behaviours above.

Outlets: `celeris.spillway.SpillwaySink` draws a level-dependent discharge
(`Q(h_up, h_down)` from a Poleni weir law or a stage-discharge table) off a wet
patch, optionally into an outlet patch, through the same continuity source as
the hydrograph inflows. `tests/test_spillway.py` checks it against the exact
tank-drawdown solution; `examples/validation/spillway` is the TELEMAC-2D
`weirs2` twin (six ponds, five weirs, levels within 1.4 cm over 12 h).
Use `"conserving"` for anything where a slowly rising level must wet dry ground
(reservoir filling, overtopping of a crest, lake inflow); the legacy scheme only
wets a cell when momentum already points at it.

## Documentation
The full user guide, API references, and examples live in the Sphinx docs under `docs/`. Hosted documentation is available at [CelerisAi’s documentation](https://wrenteria.github.io/CelerisAi/).

## Examples
After installing, run one of the sample simulations:

### 1D simulation with CelerisAI
Minimal 1D setup script for running CelerisAI.

[1D profile](./setrun_1D.py)

![1Dprofile](docs/source/images/Celeris1D.gif)
```bash
python setrun_1D.py
```

### Using CelerisWEBGPU configuration

Run a CelerisAI example using the SAME configuration layout as [CelerisWEBGPU](https://plynett.github.io/).

[Balboa beach](./setrun_web.py)

![Balboa](docs/source/images/BalboaBeach.gif)
```bash
python setrun_web.py
```

### Control image/log output interval

`Evolve` prints progress and saves frames (when `saveimg=True`) every 100 steps by default.
You can change this with `plot_interval`:

```python
run = Evolve(solver=solver, maxsteps=10000, saveimg=True, plot_interval=25)
```

## Tutorial Notebooks

Jupyter tutorials that mirror the main run scripts are available in `notebooks/`:

- [`notebooks/setrun_1D_tutorial.ipynb`](notebooks/setrun_1D_tutorial.ipynb): step-by-step 1D setup from `setrun_1D.py`.
- [`notebooks/setrun_web_tutorial.ipynb`](notebooks/setrun_web_tutorial.ipynb): CelerisWebGPU-style 2D case setup from `setrun_web.py`.

## Dependencies
### Core requirements
Installed automatically by `pip install -e .`:

- Python 3.x
- Taichi
- NumPy
- SciPy
- Matplotlib
- ImageIO

### Optional requirements
PyTorch is used for AI integration workflows. Install it separately if you plan to run the differentiability or learning examples.


## License
CelerisAi is released under the MIT License. See [LICENSE](https://github.com/wrenteria/CelerisAi/blob/main/LICENSE) for details.

## Contributions
CelerisAi is an open-source project. Contributions from the community are welcome. Please refer to the project's guidelines for contributing.

## Citation
If you find this version of CelerisAI useful for your research, consider citing:

```
@article{renteria_differentiable_2026,
	title = {A {Differentiable} {Solver} for {Phase}-resolved {Nearshore} {Wave} {Modelling}},
	issn = {00104655},
	url = {https://linkinghub.elsevier.com/retrieve/pii/S0010465526002134},
	doi = {10.1016/j.cpc.2026.110231},
	language = {en},
	urldate = {2026-05-19},
	journal = {Computer Physics Communications},
	author = {Renteria, Willington and McCann, Maile and Bonus, Justin and Ebrahimi, Behzad and Lynett, Patrick},
	month = may,
	year = {2026},
	pages = {110231},
}
```
