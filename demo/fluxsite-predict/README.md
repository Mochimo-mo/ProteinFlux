# FluxSite phosphorylation-site demo

Predict Ser/Thr phosphorylation sites in mouse KCNMA1 (UniProt Q08460). The
example includes the FASTA sequence, PDB structure, candidate positions, and
precomputed sequence and structure features.

From the repository root, install `requirements.txt`, download the
[`pho_st_model.pt` checkpoint](https://huggingface.co/clab-qqt/ProteinFlux/blob/main/fluxsite/pho_st_model.pt)
to `fluxsite/pho_st_model.pt`, then run:

```bash
bash demo/fluxsite-predict/predict_phos.sh
```

The prediction table is written to `demo/fluxsite-predict/out/prediction_results.csv`.
Its `probability` column is the model's phosphorylation score for each Ser/Thr
candidate, and `prediction` uses a threshold of 0.5. The script also writes a
run summary and, when applicable, attention plots. Pass `--gpu_id N` to select
a GPU, or set `MODEL_PATH` to use a checkpoint stored elsewhere.

The supplied HDF5 file covers Q08460. To predict another protein, provide a
matching feature HDF5 file and position CSV through the script's command-line
arguments.
