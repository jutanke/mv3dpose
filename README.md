# mv3dpose
Off-the-shelf Multiple Person Multiple View 3D Pose Estimation. 

![out](https://user-images.githubusercontent.com/831215/58240723-db7e0880-7d4b-11e9-955d-24ac7e0f44c4.gif)

## Abstract
In this work we propose an approach for estimating 3D human poses of multiple people
from a set of calibrated cameras. Estimating 3D human poses from 
multiple views has several compelling properties: human poses are estimated within a 
global coordinate space and 
multiple cameras provide an extended field of view which helps in resolving
ambiguities, occlusions and motion blurs.
Our approach builds upon a real-time 2D multi-person pose estimation system and
greedily solves the association problem between multiple views.
We utilize
bipartite matching to track multiple people over multiple frames.
This proofs to be especially efficient as problems associated with greedy matching
such as occlusion can be easily resolved in 3D.
Our approach achieves state-of-the-art results on popular benchmarks and may
serve as a baseline for future work.

## Install

This project requires [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and drivers that support cuda 10.

Clone this repository with its submodules as follows:
```bash
git clone --recursive https://github.com/jutanke/mv3dpose.git
```

## Usage

Your dataset must reside in a pre-defined folder structure:

* dataset
  * dataset.json
  * cameras
    * camera00
      * frame00xxxxxxm.json
    * camera01
      * frame00xxxxxxm.json
    * ...
    * camera_n
      * frame00xxxxxxm.json
  * videos
    * camera00
      * frame00xxxxxxm.png
    * camera01
      * frame00xxxxxxm.png
    * ...
    * camera_n
      * frame00xxxxxxm.png

The file names per frame utilize the following schema: 
```python
"frame%09d.{png/json}"
```

The camera json files follow two types of structures: A simple camera with only the projection matrix and width and height:
```javascript
{
  "P" : [ 3 x 4 ],
  "w" : int(width),
  "h" : int(height)
}
```
or a more complex camera setup with distortion coefficients. This camera is based on OpenCV.
```javascript
{
  "K" : [ 3 x 3 ], /* intrinsic paramters */
  "rvec": [ 1 x 3 ], /* rotation vector */
  "tvec": [ 1 x 3 ], /* translation vector */
  "discCoef": [ 1 x 5 ], /* distortion coefficient */
  "w" : int(width),
  "h" : int(height)
}
```

The system expects a camera for each view at each point in time. If your dataset uses fixed cameras you will need to simply repeat them for all frames.

The _dataset.json_ file contains general information for the model:
```javascript
{
  "n_cameras": int(#cameras), /* number of cameras */
  "scale_to_mm": 1, /* scales the calibration to mm */
}
```

The variable __scale_to_mm__ is needed as we operate in [mm] but calibrations might be in other metrics. For example, when the calibration is done in meters, _scale_to_mm_ must be set to _1000_.

### optional Parameters
* __valid_frames__: if frames do not start at 0 and/or are not continious you can set a list of frames here
* __epi_threshold__: epipolar line distance threshold in PIXEL
* __max_distance_between_tracks__: maximal distance in [mm] between tracks so that they can be associated
* __min_track_length__: drop any track which is shorter than _min_track_length_ frames
* __last_seen_delay__: allow to skip _last_seen_delay_ frames for connecting a lost track
* __smoothing_sigma__: sigma value for Gaussian smoothing of tracks
* __smoothing_interpolation_range__: define how far fill-ins should be reaching
* __do_smoothing__: should smoothing be done at all? (Default is True)

### Run the system

```bash
./mvpose.sh /path/to/your/dataset
```

The resulting tracks will be in your dataset folder under __tracks3d__, each track represents a single person. 
The files are organised as follows:
```javascript
{
  "J": int(joint number), /* number of joints */
  "frames": [int, int], /* ordered list of the frames where this track is residing */
  "poses": [ n_frames x J x 3 ] /* 3D poses, 3d location OR None, if joint is missing */
}
```
