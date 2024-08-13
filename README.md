# RINGER

This repository is the official implementation of [RINGER: Rapid Conformer Generation for Macrocycles with Sequence-Conditioned Internal Coordinate Diffusion](https://arxiv.org/abs/2305.19800).

![cover](assets/overview.png)

## Requirements

To install requirements:

```setup
conda env create -f environment.yaml
conda activate ringer
pip install -e .
```

## Data

Download and extract the CREMP pickle.tar.gz from [here](https://doi.org/10.5281/zenodo.7931444). Use [train.csv](data/cremp/train.csv) and [test.csv](data/cremp/test.csv) to partition it into training and test data and put the corresponding files into [train](data/cremp/train) and [test](data/cremp/test).

## Training

To train the full conditional model, run this command:

```train
train conditional.json
```

The config file can be specified by an absolute path or by a path relative to the [configs](configs) folder. Similarly, within the config file, `data_dir` can be an absolute path or a path relative to the [data](data) folder.

To log a training run with Weights & Biases, set up your configuration in [configs/wandb/wandb.json](configs/wandb/wandb.json) and set up logging using:

```train
train conditional.json --wandb-run <run_name>
```

## Sampling

The [pre-trained model](assets/models/conditional) is included in this repository.

To generate samples for the CREMP test set, run:

```eval
evaluate \
    --model-dir assets/models/conditional \
    --data-dir cremp/test \
    --split-sizes 0.0 0.0 1.0 \
    --sample-only
```

This creates a `sample` directory containing samples for all molecules in `sample/samples.pickle`.

Run `evaluate --help` to see all options available for sampling and evaluation.

## Reconstruction

The `evaluate` command can also be used to reconstruct backbones (not including side chains) and to compute evaluation metrics. However, it is not recommended to do so because `evaluate` does not parallelize well across molecules.

Instead, reconstruction (including side chains) is done most effectively for each molecule individually using [scripts/reconstruct_single.py](scripts/reconstruct_single.py). Parallelization can then be efficiently achieved by submitting a batch job array using an HPC job scheduler (e.g., Slurm) and passing the job array index as the first argument to the script. To reconstruct molecule 0, run:

```shell
python scripts/reconstruct_single.py 0 \
    cremp/test \
    sample/samples.pickle \
    sample/reconstructed_mols \
    assets/models/conditional/training_mean_distances.json
```

The script will run the optimization to reconstruct the ring coordinates, followed by a linear (NeRF) reconstruction of the side chains using the [conformer samples previously generated](#sampling), and save the resulting molecule in `sample/reconstructed_mols`. Note that even though we point the script to `cremp/test`, it only uses the atom identities and connectivity information from the test molecules; their geometries are entirely set during the reconstruction procedure.

Run `python scripts/reconstruct_single.py --help` for an overview of other parameters available for reconstruction.

## Evaluation

As with reconstruction, computing metrics is best done separately for each molecule using [scripts/compute_metrics_single.py](scripts/compute_metrics_single.py) followed by aggregation across molecules using [scripts/aggregate_metrics.py](scripts/aggregate_metrics.py). For example, to compute metrics for the `H.A.S.V` macrocycle, run

```shell
python scripts/compute_metrics_single.py \
    cremp/test/H.A.S.V.pickle \
    sample/reconstructed_mols/H.A.S.V.pickle
```

Run `python scripts/compute_metrics_single.py --help` and `python scripts/aggregate_metrics.py --help` for an overview of other parameters available for computing metrics.

## Contributing

Install pre-commit hooks to use automated code formatting before committing changes. Make sure you're in the top-level directory and run:

```bash
pre-commit install
```

After that, your code will be automatically reformatted on every new commit.

To manually reformat all files in the project, use:

```bash
pre-commit run -a
```

To update the hooks defined in [.pre-commit-config.yaml](.pre-commit-config.yaml), use:

```bash
pre-commit autoupdate
```

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for additional details.

## Citations

For the code and/or model, please cite:

```
@misc{grambow2023ringer,
    title={{RINGER}: Rapid Conformer Generation for Macrocycles with Sequence-Conditioned Internal Coordinate Diffusion}, 
    author={Colin A. Grambow and Hayley Weir and Nathaniel L. Diamant and Alex M. Tseng and Tommaso Biancalani and Gabriele Scalia and Kangway V. Chuang},
    year={2023},
    eprint={2305.19800},
    archivePrefix={arXiv},
    primaryClass={q-bio.BM}
}
```

To cite the CREMP dataset, please use:

```
@article{grambow2024cremp,
    title = {{CREMP: Conformer-rotamer ensembles of macrocyclic peptides for machine learning}},
    author = {Grambow, Colin A. and Weir, Hayley and Cunningham, Christian N. and Biancalani, Tommaso and Chuang, Kangway V.},
    year = {2024},
    journal = {Scientific Data},
    doi = {10.1038/s41597-024-03698-y},
    pages = {859},
    number = {1},
    volume = {11}
}
```

You can also cite the CREMP Zenodo repository directly:

```
@dataset{grambow_colin_a_2023_7931444,
  author       = {Grambow, Colin A. and
                  Weir, Hayley and
                  Cunningham, Christian N. and
                  Biancalani, Tommaso and
                  Chuang, Kangway V.},
  title        = {{CREMP: Conformer-Rotamer Ensembles of Macrocyclic 
                   Peptides for Machine Learning}},
  month        = may,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {1.0.1},
  doi          = {10.5281/zenodo.7931444},
  url          = {https://doi.org/10.5281/zenodo.7931444}
}
```
