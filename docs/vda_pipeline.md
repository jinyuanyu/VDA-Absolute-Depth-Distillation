# Video Depth Anything Pipeline

Repository: `/home/a1/XZB/Video-Depth-Anything-main`
Environment: `conda activate vda`
Core source references:

- `video_depth_anything/video_depth.py`
- `utils/dc_utils.py`
- `utils/util.py`
- `run.py`

## 0. Machine-Specific Status

The local `run.py` is not currently a clean universal entry for DyNeRF image directories:

- it is modified to read a hardcoded THumanMV path
- it still contains the generic model logic, but not a generic image-directory CLI flow anymore

For DyNeRF `camXX` directories, the actual usable procedure on this machine was:

- call `read_image_sequence(...)`
- instantiate `VideoDepthAnything(...)`
- call `infer_video_depth(...)`
- save outputs manually as `npy/jpg/npz/mp4`

Also note:

- available checkpoint on this machine: `checkpoints/video_depth_anything_vitl.pth`
- metric VDA checkpoint is not present here
- therefore currently generated `raw_vda_depth/*` results are relative depth results

## 1. Input Data Structure

Used input form in this project:

- ordered image sequence from a directory such as `/media/a1/16THDD/XZB/DyNeRF/coffee_martini/images/cam00`

Raw frame structure before load:

- file type: `.jpg`
- example original size: `1352 x 1014`
- frame count per camera used here: `300`

## 2. Step-By-Step Data Structure Changes

### Step A. Read image sequence

In `utils/dc_utils.py::read_image_sequence`:

- all file paths are read with OpenCV
- BGR is converted to RGB
- frames are stacked into one numpy array
- if `max_res > 0` and long side exceeds `max_res`, all frames are uniformly resized
- if `target_fps <= 0`, fps defaults to `30`

Structure after this step:

- type: `numpy.ndarray`
- layout: `T x H x W x 3`
- dtype: `uint8`

In our runs:

- original `1352 x 1014` was downscaled to `1280 x 960` because `max_res = 1280`
- example stacked structure: `(300, 960, 1280, 3)`

Boundary conditions:

- if no readable frames are found, `ValueError("未读取到任何图片")` is raised
- unreadable individual frames are skipped with a warning
- all remaining frames must still stack to a single array shape

### Step B. Aspect-ratio-based input size adjustment

In `infer_video_depth`:

- original frame ratio is computed as `max(H, W) / min(H, W)`
- if ratio > `1.78`, internal `input_size` is reduced
- reduced `input_size` is rounded to a multiple of `14`

Reason:

- the code explicitly warns that very wide videos should be reduced for memory reasons

Boundary condition:

- aspect ratio above roughly 16:9 triggers smaller internal processing size

### Step C. Per-frame transform for the network

Each frame is transformed with:

- `Resize(... keep_aspect_ratio=True, ensure_multiple_of=14, resize_method='lower_bound')`
- `NormalizeImage(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`
- `PrepareForNet()`

Per transformed frame structure before batching:

- type: `torch.Tensor`
- layout: `1 x 1 x 3 x h x w`
- dtype: floating-point

Important constraint:

- spatial size must align with ViT patching, so multiples of `14` are enforced internally

### Step D. Temporal window construction

Fixed inference constants in `video_depth.py`:

- `INFER_LEN = 32`
- `OVERLAP = 10`
- `KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]`
- `INTERP_LEN = 8`

Sequence handling:

- window step = `INFER_LEN - OVERLAP = 22`
- the sequence is padded by repeating the last frame so that all windows are valid
- for windows after the first one, the first `OVERLAP` positions are replaced by cached keyframe inputs from the previous window

Structure of one model input window:

- `1 x 32 x 3 x h x w`

Boundary conditions:

- any sequence length is accepted because the tail is padded with copies of the last frame
- the code assumes ordered input; wrong filename ordering changes temporal consistency

### Step E. Model forward pass

In `VideoDepthAnything.forward`:

- input shape: `B x T x C x H x W`
- temporal-depth head predicts dense depth for all frames in the window
- result is interpolated back to the transformed frame size and passed through `ReLU`

Immediate output structure:

- shape: `B x T x H x W`
- negative values are prevented by `ReLU`

### Step F. Upsampling back to original frame resolution

Still inside `infer_video_depth`:

- each window output is interpolated back to the original pre-transform frame size from Step A
- per-frame depth arrays are moved to CPU and collected in a python list

Structure after this step:

- python list of `H_original x W_original` numpy arrays
- on this machine that means `960 x 1280` for the resized DyNeRF sequence used by VDA

### Step G. Cross-window alignment and interpolation

After all windows are inferred:

- if `metric=False`, overlap regions from the current window are aligned to reference frames from the previous window using `compute_scale_and_shift`
- overlap transition is smoothed by `get_interpolate_frames`
- negative values after alignment are clipped to `0`
- if `metric=True`, scale/shift is fixed to `(1.0, 0.0)` in this code path

This is the key stage that turns independent window predictions into one consistent sequence.

Boundary conditions:

- alignment depends on overlap regions existing, so sequence stitching logic assumes the fixed `INFER_LEN / OVERLAP` design
- if determinant in `compute_scale_and_shift_full` is zero, fallback defaults remain `(scale=1, shift=0)`
- post-alignment negative values are hard-clipped to zero

### Step H. Final stack and save

Final returned structure:

- `np.stack(depth_list[:org_video_len], axis=0)`
- shape: `T x H x W`
- plus returned fps

For the DyNeRF wrapper used here, we saved:

- one per-frame `*.npy` depth array
- one per-frame `*.jpg` visualization
- one sequence file `camXX_depths.npz`
- one RGB preview video `camXX_src.mp4`
- one depth visualization video `camXX_vis.mp4`

Per-frame output naming:

- input `0000.jpg` -> `0000.npy` and `0000.jpg`

## 3. Input / Output Flow

```text
ordered image sequence
-> read_image_sequence()
-> T x H x W x 3 numpy frames
-> per-frame transform + resize + normalize
-> sliding windows of 32 frames with 10-frame overlap
-> VideoDepthAnything.forward()
-> per-window depth maps
-> resize back to original frame resolution
-> scale/shift temporal alignment + interpolation across overlaps
-> final T x H x W depth stack
-> save framewise *.npy + *.jpg + sequence *.npz + preview *.mp4
```

## 4. Constraints And Boundary Conditions Summary

- Designed for temporally ordered data, not independent shuffled images
- Internal patching requires sizes compatible with multiples of `14`
- Very wide aspect ratios reduce effective input size automatically
- Empty or unreadable sequence raises an error before inference
- Tail padding repeats the last frame to make window scheduling valid
- Relative-model mode uses overlap alignment; metric-model mode skips scale/shift correction in this implementation
- Current machine only has relative VDA weights available
- Current local `run.py` is not a general DyNeRF batch tool because of a hardcoded dataset path

## 5. Observed Example On This Machine

Observed output example:

- output: `/media/a1/16THDD/XZB/DyNeRF/coffee_martini/raw_vda_depth/cam01/0000.npy`
- array shape: `(960, 1280)`
- dtype: `float32`

Relative to Depth Pro:

- VDA currently works on the resized sequence (`960 x 1280` in our runs)
- Depth Pro outputs match the original frame resolution (`1014 x 1352` in the checked sample)
