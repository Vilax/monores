# MonoRes - Local resolution

MonoRes is an algorithm to estimate the local resolution in cryoEM maps. It estimates the local resolution by establishing an statistical comparison at different resolutions between the local signal (obtained from the monogenic amplitude) and the noise. The limit at which the local signal cannot be distisguished from noise will be the local resolution of the region. 

**Reference**: This method can be detailed read in:
> J.L. Vilas et al., *MonoRes: Automatic and Accurate Estimation of Local
> Resolution for Electron Microscopy Maps*, **Structure** 26, 337-344 (2018).

The understading of local resolution can be misleading. Local resolution should be understood as a relative measurement between different regions. Therefore it is not an absolute measurement. For a discussion about the good practices in local resolution, the developers of local resolution wrote an article to inform about their use, see

> J.L. Vilas et al., *Local resolution estimates of cryoEM reconstructions*, **Current opinion in structural biology** 64, 74-78 (2020).


## Install

Using conda/mamba (recommended, `environment.yml` included in this folder):

```bash
conda env create -f environment.yml
conda activate monores
```

By default `environment.yml` installs the CPU-only PyTorch build. To use a
GPU, edit the file: comment out the `pytorch` / `cpuonly` pair under the
CPU section and uncomment the matching `pytorch` / `pytorch-cuda=...` block
for your installed NVIDIA driver, then re-run `conda env create`.

Or with plain pip:

```bash
pip install --user torch mrcfile numpy
# CPU-only torch (much smaller):
# pip install --user --index-url https://download.pytorch.org/whl/cpu torch
```

## Usage

Before each execution the conda environment must be activated

```bash
# To activate conda enviroinment
conda activate monores
```

Then, MonoRes can be executed

```bash
# two half maps (recommended)
python3 monores.py \
    --vol half1.mrc --vol2 half2.mrc --mask mask.mrc \
    --sampling_rate 1.34 --maxRes 25 --step 0.25 \
    -o monores_out

# single map
python3 monores.py --vol map.mrc --mask mask.mrc \
    --sampling_rate 1.34 --maxRes 25 -o monores_out
```

Other options can be checked with

```bash
# To check all monores options
python3 monores.py --help
```

### Watch out for `--minRes` / `--maxRes`

This is upstream behaviour, faithfully reproduced:

* `--minRes` (default 30) is **overwritten with `2 * sampling_rate`** on the
  first line of `run()`, so the CLI value has no effect at all.
* The sweep starts at `resolution = maxRes` and **decreases** by `--step` until
  it drops below Nyquist. `--maxRes` is therefore the *low-frequency* start of
  the range, despite its name.

## Outputs

Written into the `-o` folder as MRC mode 2 (float32), matching Xmipp
(`ImageBase::writeMRC` maps both `Image<double>` and `Image<int>` to mode 2):

| File | Content |
| --- | --- |
| `meanMap.mrc` | `0.5*(V1+V2)` -- half-map mode only |
| `refinedMask.mrc` | binary mask of voxels that got a resolution value |
| `monoresResolutionChimera.mrc` | smoothed resolution map over the whole box (for visualisation) |
| `monoresResolutionMap.mrc` | smoothed resolution map, zero outside the mask |
