# validation_RFDF3

A post-design validation pipeline for **RFdiffusion3 (RFD3)**-generated nucleic-acid-binding proteins.

This repository provides scripts and notebooks to: (1) generate protein backbones with RFD3, (2) design amino-acid sequences with LigandMPNN, and (3) validate the designs with AlphaFold2-monomer structure prediction, OpenMM relaxation, and structural metrics (RMSD / clashes).

---

## Overview

```
RFD3 backbone (CIF)  →  LigandMPNN sequence design (TXT)  →  AF2-monomer prediction + OpenMM relax  →  RMSD / clash validation
```

| Step | Script / Tool | Purpose |
|------|---------------|---------|
| 1 | `rfd3 design` | Generate *de novo* protein backbones around a nucleic-acid target |
| 2 | `step_2_MPNN.py` | Design sequences for the generated backbones using LigandMPNN |
| 3 | `step_3_af2monomer.py` | Predict structures with AlphaFold2, relax with OpenMM, and validate against the RFD3 reference |

---

## Repository Structure

```
validation_RFDF3/
├── all.ipynb                 # End-to-end RFD3 → MPNN → RF3 design notebook
├── test.ipynb                # Additional testing / development notebook
├── step_2_MPNN.py            # Sequence design with LigandMPNN
├── step_3_af2monomer.py      # AF2 prediction, OpenMM relaxation, and validation
├── run.sh                    # Example shell script for the full pipeline
├── environment.yaml          # Conda environment definition
├── example/                  # Example input / output data
│   ├── input_pdb/
│   ├── output_cif/
│   ├── rfd3_outputs/
│   ├── test/                 # MPNN output sequences
│   └── test_output/          # Passing validated structures
└── ColabDesign-main/         # ColabDesign (AlphaFold2 inference) submodule
    └── params/               # AF2 model parameters
```

---

## Installation

### 1. Create the conda environment

```bash
conda env create -f environment.yaml
conda activate RFDF3
```

### 2. Install / locate RFD3 and LigandMPNN

The pipeline assumes you have a working RFD3 installation. For details see the [RFD3 nucleic-acid binder design tutorial](https://rosettacommons.github.io/foundry/models/rfd3/examples/na_binder_design.html).

You will also need the AF2 model parameters inside `ColabDesign-main/params/`.

> **Note for WSL/WSL2 users:** If you encounter GCC / `libstdc++.so.6` issues, try:
> ```bash
> export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
> ```

---

## Usage

### Full pipeline (see `run.sh`)

```bash
#!/bin/bash
conda activate RFDF3

# 1. RFD3 backbone generation
rfd3 design \
    out_dir=/your/output/dir \
    inputs=/your/json/document \
    ckpt_path=/path/to/your/checkpoint/files/rfd3_latest.ckpt

# Decompress generated CIF files
gunzip /your/output/dir/*.gz

# 2. Sequence design with LigandMPNN
python3 step_2_MPNN.py \
    -i /path/to/rfd3/output_cif \
    -o /path/to/mpnn/output

# 3. AF2 validation + OpenMM relaxation
python3 step_3_af2monomer.py \
    -ir /path/to/mpnn/output \
    -ic /path/to/rfd3/output_cif \
    -o /path/to/validated/output \
    -n 14 \
    -af /path/to/ColabDesign-main/params
```

### Step 2: LigandMPNN sequence design

```bash
python3 step_2_MPNN.py \
    -i /path/to/rfd3/output_cif \
    -o /path/to/mpnn/output \
    -n 2
```

Arguments:
- `-i`, `--input`: Directory containing RFD3-generated `.cif` files.
- `-o`, `--output`: Directory where designed sequences (`.txt`) will be saved.
- `-n`, `--number`: Number of protein chains to design (default: 2).

The script automatically detects DNA/RNA chains and protein chains, identifies interface residues within 4 Å, and runs LigandMPNN to generate sequences.

### Step 3: AlphaFold2 validation

```bash
python3 step_3_af2monomer.py \
    -ir /path/to/mpnn/output \
    -ic /path/to/rfd3/output_cif \
    -o /path/to/validated/output \
    -n 14 \
    -af /path/to/ColabDesign-main/params
```

Arguments:
- `-ir`, `--inputresidue`: Directory containing MPNN-designed sequences (`.txt`).
- `-ic`, `--inputcif`: Directory containing the original RFD3 `.cif` files.
- `-o`, `--output`: Directory where passing relaxed PDBs will be moved.
- `-n`, `--nc`: Half-length of the nucleotide strand (default: 28; total nucleic-acid length = `n * 2`).
- `-af`, `--af2model`: Path to the AF2 parameter directory (default: `./ColabDesign-main/params`).

A structure passes validation when:
- `pLDDT > 0.85` and `pTM > 0.65` from AlphaFold2
- Relaxed PDB has no internal clashes
- Interface clash score < 10
- Backbone RMSD to the RFD3 reference < 2.5 Å

---

## Example Data

The `example/` directory contains small input/output sets that can be used to sanity-check the pipeline:

- `example/output_cif/` — sample RFD3 outputs
- `example/test/` — sample MPNN-designed sequences
- `example/test_output/` — structures that passed validation

---

## Notebooks

- **`all.ipynb`** — end-to-end demonstration of the RFD3 + LigandMPNN + RF3 design workflow.
- **`test.ipynb`** — development and testing notebook for validation routines.

---

## Dependencies

Key dependencies are listed in `environment.yaml`. Notable packages include:

- Python 3.12
- `rfd3` / `rc-foundry`
- `ligandmpnn`
- `colabdesign`
- `biotite`, `biopython`
- `openmm`, `pdbfixer`
- `scipy`, `numpy`, `natsort`
- `jupyter`

---

## Contributing

Contributions and bug reports are welcome. Please open an issue or pull request on GitHub.

---

## References

If you use this pipeline in your research, please cite the following papers:

- Butcher, J., Krishna, R., Mitra, R., et al. (2025).  
  *De novo Design of All-atom Biomolecular Interactions with RFdiffusion3.*  
  bioRxiv. https://doi.org/10.1101/2025.09.18.676967

- Dauparas, J., Anishchenko, I., Bennett, N., et al. (2022).  
  *Robust deep learning–based protein sequence design using ProteinMPNN.*  
  Science, 378(6615), 49–56. https://doi.org/10.1126/science.add2187

- Jumper, J., Evans, R., Pritzel, A., et al. (2021).  
  *Highly accurate protein structure prediction with AlphaFold.*  
  Nature, 596(7873), 583–589. https://doi.org/10.1038/s41586-021-03819-2

---

## License

[Specify your license here, e.g., MIT / Apache-2.0]

---

## Acknowledgments

- RFdiffusion3, LigandMPNN, and RF3 are developed by the [Institute for Protein Design](https://www.ipd.uw.edu/) and distributed via [Rosetta Commons Foundry](https://rosettacommons.github.io/foundry/).
- AlphaFold2 structure prediction is powered by [ColabDesign](https://github.com/sokrypton/ColabDesign).
