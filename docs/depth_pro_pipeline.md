# Depth Pro Pipeline

Repository: `/home/a1/XZB/ml-depth-pro`
Environment: `conda activate depth-pro`
Primary entry used: `depth-pro-run`
Source references:

- `src/depth_pro/cli/run.py`
- `src/depth_pro/depth_pro.py`
- `src/depth_pro/utils.py`

## 1. Input Data Structure

Accepted by the CLI:

- single image path
- directory path

For the DyNeRF runs here, input is a directory like:

- `/media/a1/16THDD/XZB/DyNeRF/coffee_martini/images/cam00`

Per frame raw structure before loading:

- file type: `.jpg`
- example size: `1352 x 1014`
- color mode: `RGB`
- EXIF on checked sample: empty

## 2. Step-By-Step Data Structure Changes

### Step A. File discovery

In `cli/run.py`:

- if input is a directory, `glob("**/*")` is used
- all paths are iterated
- non-image files are not pre-filtered, but load failures are caught and skipped

Boundary condition:

- if a directory contains unrelated files, they trigger an exception in `load_rgb`, are logged, and the loop continues

### Step B. Image load and metadata extraction

In `utils.py::load_rgb`:

- image is opened with Pillow
- HEIC is specially supported through `pillow_heif`
- EXIF is parsed through `extract_exif`
- optional EXIF orientation correction is applied
- grayscale images are expanded to 3 channels
- alpha channel is dropped if present

Structure after load:

- type: `numpy.ndarray`
- layout: `H x W x 3`
- dtype: usually `uint8`
- side outputs: `icc_profile`, `f_px`

Boundary conditions:

- no EXIF focal-length tag -> `f_px = None`
- grayscale input -> duplicated into 3 channels
- alpha input -> truncated to first 3 channels
- unusual EXIF orientations other than `1/3/6/8` are ignored with a warning

### Step C. Tensor transform

In `create_model_and_transforms`:

- `ToTensor()` converts image to tensor in `C x H x W`
- tensor is moved to target device
- image is normalized with mean/std `[0.5, 0.5, 0.5]`
- dtype is converted to requested precision, usually `torch.float16` in the CLI

Structure after transform:

- type: `torch.Tensor`
- layout: `3 x H x W`
- device: CPU / CUDA / MPS depending on availability
- dtype: default CLI uses `float16`

### Step D. Model inference

In `DepthPro.infer`:

- if the input is 3D, a batch dimension is added -> `1 x 3 x H x W`
- if input spatial size does not equal internal model size `self.img_size`, it is resized to a square internal resolution
- model predicts canonical inverse depth and optional FOV
- if `f_px` from EXIF is missing, focal length is estimated from predicted FOV
- inverse depth is rescaled by image width and focal length
- output is resized back to original `H x W` if internal resizing occurred
- final depth is computed as `1 / clamp(inverse_depth, min=1e-4, max=1e4)`

Structure after inference:

- `prediction["depth"]`: `H x W`, `torch.Tensor`
- `prediction["focallength_px"]`: scalar tensor or scalar-like value

Boundary conditions:

- internal size mismatch is handled automatically by interpolate-down / interpolate-up
- inverse depth is clamped before reciprocal to avoid divide-by-zero and extreme values
- output is always per-frame independent; there is no temporal smoothing

### Step E. Visualization conversion

In `cli/run.py`:

- `inverse_depth = 1 / depth`
- visualization range is clipped to approximately `[0.1 m, 250 m]` in inverse-depth space
- normalized inverse depth is mapped through matplotlib `turbo`

Structure:

- color visualization array: `H x W x 3`, `uint8`

Boundary conditions:

- the visualization is not the raw metric depth; it is a clipped inverse-depth colormap for display only

### Step F. Save-to-disk

When `--output-path` is provided:

- raw depth saved as compressed `npz`, key = `depth`
- visualization saved as `jpg`

Per-frame output naming:

- input `0000.jpg` -> `0000.npz` and `0000.jpg`

Output structure used in this project:

- directory: `/media/a1/16THDD/XZB/DyNeRF/coffee_martini/raw_depth_pro_depth/camXX`
- one frame = one `npz` + one `jpg`

## 3. Input / Output Flow

```text
image file(s)
-> load_rgb()
-> HWC numpy RGB image
-> transform()
-> CHW normalized tensor
-> DepthPro.infer()
-> depth tensor at original resolution
-> inverse-depth visualization
-> save *.npz (raw depth) + *.jpg (vis)
```

## 4. Constraints And Boundary Conditions Summary

- Best suited for per-image inference, not temporal-consistency tasks
- Directory traversal may encounter non-image files; those are skipped through exception handling
- If EXIF focal length is absent, model-estimated FOV is used
- Output resolution follows original input resolution, even though the network runs internally at its own square size
- CLI device selection is `cuda:0` if CUDA is available; in our runs, `CUDA_VISIBLE_DEVICES` was used so each remote worker still saw its assigned GPU as local `cuda:0`
- Saving only happens if `--output-path` is provided

## 5. Observed Example On This Machine

Checked sample:

- input: `/media/a1/16THDD/XZB/DyNeRF/coffee_martini/images/cam00/0000.jpg`
- input size: `1352 x 1014`
- EXIF orientation: none
- EXIF 35mm focal length: none

Observed output example:

- output: `/media/a1/16THDD/XZB/DyNeRF/coffee_martini/raw_depth_pro_depth/cam01/0000.npz`
- array shape: `(1014, 1352)`
- dtype: `float32`
