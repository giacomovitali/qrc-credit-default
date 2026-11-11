# Classical (mean-field) reservoir

The classical spin reservoir — the quantum reservoir with every spin promoted to a classical unit
vector — is adapted from the notebook `Classical Spin Reservoir Demo.ipynb` of QuEra's
[QRC-tutorials](https://github.com/QuEraComputing/QRC-tutorials) repository, which accompanies
M. Kornjača et al., *Large-scale quantum reservoir learning with an analog quantum computer*,
[arXiv:2407.02553](https://arxiv.org/abs/2407.02553). `utils.jl` keeps its layer (`ClassicalDetuningLayer`,
`apply_classical_layer`, `generate_Vmat`); `CRC.jl` applies it to the 12 aggregated features of this
work. Usage: [docs/pipeline.md](../docs/pipeline.md#classical-reservoir).
