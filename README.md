# Multi person and multi camera 3D pose estimator from unlabelled data using a transformer.

Based on the paper - [Multi-person 3D pose estimation from unlabelled data](https://link.springer.com/article/10.1007/s00138-024-01530-6)

In this implementation, a transformer has been developed as a 3D pose estimator in the last step.

[Trained models](https://drive.google.com/drive/folders/1MoBUoc4LqjBnrLz5F38Z4WttkSdf02Mo?usp=sharing)


## Installation

To start using the system you just need to clone this repository on your local machine:

``` shell
git clone https://github.com/gnns4hri/3D_multi_pose_estimator.git
```
Install the dependencies:

- pip3 install -r *requirements.txt*
- sudo apt install python3-opencv
- Install DGL with cuda following the instructions in [https://www.dgl.ai/pages/start.html](https://www.dgl.ai/pages/start.html)

## Cameras' calibration

To use our system with your own camera setup, you must first calibrate the cameras to share a common frame of reference. Create a script that generates a pickle file containing the extrinsic matrices for each camera.

We utilized an AprilTag to establish the coordinate origin during calibration. In this repository, you will find two sample calibration files: tm_arp.pickle and tm_panoptic.pickle, representing the transformation matrices for ARP lab cameras and CMU Panoptic cameras, respectively. Each file contains a _TransformManager_ of [pytransform3d](https://dfki-ric.github.io/pytransform3d/) comprising all the transformations between the cameras and the global frame of reference.

Make sure to specify the camera names in the `parameters.py` file creating a new configuration for your environment.

## Datasets
To train or test a new or existing model, you can use either of the two available datasets or create a new dataset with your own set of cameras. The pre-trained models can be found in the _models_ directory [here](https://www.dropbox.com/sh/6cn6ajddrfkb332/AACg_UpK22BlytWrP19w_VaNa?dl=0). Please note that a pre-trained model must be tested with a dataset that shares the same configuration used during the model's training process.

### CMU Panoptic and ARP Lab datasets

To download the available datasets for testing our model, visit the _datasets_ directory [here](https://www.dropbox.com/sh/6cn6ajddrfkb332/AACg_UpK22BlytWrP19w_VaNa?dl=0). This includes the ARP Lab and CMU Panoptic datasets. To test our proposal on one of these two datasets, set `CONFIGURATION` in `paramters.py` to the dataset name ('PANOPTIC' or 'ARPLAB').

Note that the Panoptic dataset only contains HD cameras with IDs 3, 6, 12, 13, and 23. To train the models using a different set of cameras from the [CMU Panoptic dataset](http://domedb.perception.cs.cmu.edu/), download the sequences by following the instructions in the [PanopticStudio Toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox). Use the Panoptic toolbox to uncompress the HD images of the sequences for the desired camera IDs. 

After uncompressing the images, generate a JSON file for each sequence using the scripts in *panoptic_conversor*. Download the backbone model for the CMU Panoptic dataset from the [VoxelPose project](https://github.com/microsoft/voxelpose-pytorch) and place it in the *panoptic_conversor* directory. To generate each training JSON file, run the following commands:

```shell
cd panoptic_conversor
python3 get_joints_from_panoptic_model.py PANOPTIC_SEQUENCE_DIRECTORY
```

To generate a JSON file for testing, run *get_joints_from_panoptic_model_multi.py* instead of the previous script. Both scripts generate a JSON file and a Python pickle file containing the transformation matrices of the cameras.

Remember to modify the Panoptic configuration in `parameters.py` with the new camera information when using this dataset for training.
 

### Custom dataset

To record your own dataset, follow these steps:

1. **Calibrate your cameras**: Refer to the [cameras calibration section](#cameras-calibration) for instructions on obtaining the transformation pickle file. Specify the path to this file in the appropriate configuration of `parameters.py`, along with the camera numbers and names, and the intrinsic parameters of the cameras.

2. **Record data & 2D detector**: Capture footage of a single person walking around the environment, covering a wide range of natural walking movements. Yoy can use any third-party 2D detector for this task, such as [TRT-pose](https://github.com/NVIDIA-AI-IOT/trt_pose). Ensure that data from all cameras are as synchronized as possible.

3. **JSON file format**: Save the recorded data as a JSON file. Examine the JSON files from the ARP Lab dataset for the required format. These files contain a list of dictionaries, where each dictionary corresponds to a frame with 2D detections from each camera.


## Training

Once the dataset has been generated,  the two networks (matching network and 3D estimator) can be trained separately:

#### Commands for training the skeleton matching network
``` shell
cd skeleton_matching
python3 train_skeleton_matching.py --trainset training_jsons --devset dev_jsons --testset test_jsons
```

The lists of JSON files specified for each option should contain more than $1$ file. The number of files in the training set determine the number of people the model will learn to distinguish.


#### Commands for training the pose estimator network

##### MLP
``` shell
cd pose_estimator
python3 train_mlp_pose_estimator.py --trainset training_jsons --devset dev_jsons 
```

##### TRANSFORMER
``` shell
cd pose_estimator
python3 train_transformer_pose_estimator.py --trainset training_jsons --devset dev_jsons 
```

For each set, $1$ or more JSON files can be specified. For simplicity, a single file can be created from several JSON files using the script `merge_jsons.py` in *utils*.

## Testing

To test our models, first download the test files from either dataset in the [CMU Panoptic and ARP Lab datasets section](#cmu-panoptic-and-arp-lab-datasets) and the corresponding pre-trained models from the _models_ directory [here](https://www.dropbox.com/sh/6cn6ajddrfkb332/AACg_UpK22BlytWrP19w_VaNa?dl=0). In addition, you must set `CONFIGURATION` in `paramters.py` to the dataset name ('PANOPTIC' or 'ARPLAB'). 

Alternatively, if you want to test the models you've trained with your own data, simply record additional data for this purpose. Note that any number of people can be present in these new recordings. Once everything is set up, open a terminal and navigate to the test directory:


``` shell
cd test_transformer
```
### Metrics

To evaluate the models, performance and accuracy metrics can be obtained using the folowing scripts:

``` shell
python3 metrics_from_model.py --testfiles test_files --tmdir tm_files_directory --modelsdir models_directory
```
The second script checks the results using triangulation instead of the pose estimation model.
These scripts only can be run using the CMU Panoptic dataset, since they require a ground truth for comparison purposes.

Accuracy metrics for the models trained with the ARP Lab dataset can be obtained using the following script:

``` shell
python3 reprojection_error.py --testfiles test_files --modelsdir models_directory
```

### Visualization

Visual results of the estimations can be displayed with these two scripts:

``` shell
python3 show_results_from_model.py --testfile test_file --modelsdir models_directory
```
If the files to test include a ground truth, that ground truth can be displayed along with the estimations by adding the arguments `--showgt --tmfile path_to_the_tm_file`.
