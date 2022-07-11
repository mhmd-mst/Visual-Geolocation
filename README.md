

<details open>
  <summary><strong>pix2pix</strong></summary>
  
Clone repo and install required libraries:
```bash
$ git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
$ cd pytorch-CycleGAN-and-pix2pix
$ pip install -r requirements.txt
```
## <div align="center">Dataset</div>

Download night2day dataset from the repo:
```bash
$ bash ./datasets/download_pix2pix_dataset.sh night2day
```
If you want to create your own dataset, create folder /path/to/data with subdirectories A and B. A and B should each have their own subdirectories train, val, test, etc. In /path/to/data/A/train, put training images in style A. In /path/to/data/B/train, put the corresponding images in style B. Repeat same for other data splits (val, test, etc).

Corresponding images in a pair {A,B} must be the same size and have the same filename, e.g., /path/to/data/A/train/1.jpg is considered to correspond to /path/to/data/B/train/1.jpg.

Once the data is formatted this way, call:
```bash
$ python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data
```
This will combine each pair of images (A,B) into a single image file, ready for training.
## <div align="center">Train</div>
To train run: 

```bash
$ python train.py --dataroot ./datasets/night2day --name day2night --model pix2pix --direction BtoA --batch_size 32 --n_epochs 75 --n_epochs_decay 75
```
For further arguments' configurations and information refer to `options/base_options.py` and `options/train_options.py`.

To resume training add `--continue_train` and set `--checkpoints_dir` to the saved models' path and change `epoch_count` to the epoch you want to resume from.

## <div align="center">Testing</div>

To test images, make sure you have paired images (refer to **Dataset** Section), so if you are using your own dataset
```bash
$ python test.py --dataroot ./datasets/ --name day2night --num_test <nb of samples to augment> --model pix2pix --direction BtoA --checkpoints_dir <path>
```

## <div align="center">Augment SF-train</div>
	To apply the augmentation use `pix2pix` [TO DO]
	

</details>

<br>

<details open>
  <summary><strong>BicycleGAN</strong></summary>
 Install required libraries:
```bash
$ pip install torch
$ pip install torchvision
$ pip install visdom
$ pip install dominate
```
Clone repo:
```bash
$ git clone -b master --single-branch https://github.com/junyanz/BicycleGAN.git
$ cd BicycleGAN
$ pip install -r requirements.txt
```
## <div align="center">Dataset</div>
To download night2day call:
```bash
$ bash ./datasets/download_dataset.sh night2day
```
If you want to create your own dataset, copy `datasets/combine_A_and_B.py` from pix2pix repository, then create folder /path/to/data with subdirectories A and B. A and B should each have their own subdirectories train, val, test, etc. In /path/to/data/A/train, put training images in style A. In /path/to/data/B/train, put the corresponding images in style B. Repeat same for other data splits (val, test, etc).

Corresponding images in a pair {A,B} must be the same size and have the same filename, e.g., /path/to/data/A/train/1.jpg is considered to correspond to /path/to/data/B/train/1.jpg.

Once the data is formatted this way, call:
```bash
$ python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data
```
This will combine each pair of images (A,B) into a single image file, ready for training.
	
## <div align="center">Train</div>
To train run: 

```bash
$ python train.py --dataroot ./datasets/night2day --name day2night --direction BtoA --batch_size 32 --niter 75 --niter_decay 75
```
For further arguments' configurations and information refer to `options/base_options.py` and `options/train_options.py`.

To resume training add `--continue_train` and set `--checkpoints_dir` to the saved models' path and change `epoch_count` to the epoch you want to resume from.

## <div align="center">Testing</div>

To test images, make sure you have paired images (refer to **Dataset** Section), so if you are using your own dataset
```bash
$ python test.py --phase test --dataroot ./datasets/ --n_samples 1 --num_test <nb of samples to augment> --name day2night --direction BtoA --checkpoints_dir <path>
```

## <div align="center">Augment SF-train</div>
	To apply the augmentation use `Bicycle` [TO DO]
	

</details>
	
<br>

<details open>
  <summary><strong>ToDayGAN</strong></summary>
Install required libraries:
```bash
$ pip install torch
$ pip install torchvision
$ pip install visdom
$ pip install dominate	
```
Clone repo:
```bash
$ git clone https://github.com/AAnoosheh/ToDayGAN.git
$ cd ToDayGAN
```
## <div align="center">Dataset</div>	
Prepare test set `test` with subfolders `test0` for day images and `test1` empty (It is important to create `test1` even though it is empty so that the code doesn't crash)
## <div align="center">Download Pre-trained model</div>
Download pretrained model for Oxford RobotCars dataset [HERE](https://www.dropbox.com/s/mwqfbs19cptrej6/2DayGAN_Checkpoint150.zip?dl=0).


## <div align="center">Testing</div>


```bash
$ python test.py --phase test --serial_test --name day2night --dataroot ./datasets/ --n_domains 2 --which_epoch 150 --loadSize 512 --checkpoints_dir <path>
```

## <div align="center">Augment SF-train</div>
	To apply the augmentation use `Bicycle` [TO DO]
	

</details>



















# Rethinking Visual Geo-localization for Large-Scale Applications

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-pittsburgh-250k)](https://paperswithcode.com/sota/visual-place-recognition-on-pittsburgh-250k?p=rethinking-visual-geo-localization-for-large)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-pittsburgh-30k)](https://paperswithcode.com/sota/visual-place-recognition-on-pittsburgh-30k?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-tokyo247)](https://paperswithcode.com/sota/visual-place-recognition-on-tokyo247?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-mapillary-val)](https://paperswithcode.com/sota/visual-place-recognition-on-mapillary-val?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-st-lucia)](https://paperswithcode.com/sota/visual-place-recognition-on-st-lucia?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-sf-xl-test-v1)](https://paperswithcode.com/sota/visual-place-recognition-on-sf-xl-test-v1?p=rethinking-visual-geo-localization-for-large)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rethinking-visual-geo-localization-for-large/visual-place-recognition-on-sf-xl-test-v2)](https://paperswithcode.com/sota/visual-place-recognition-on-sf-xl-test-v2?p=rethinking-visual-geo-localization-for-large)

This is the official repository for the CVPR 2022 paper [Rethinking Visual Geo-localization for Large-Scale Applications](https://arxiv.org/abs/2204.02287).
The paper presents a new dataset called San Francisco eXtra Large (SF-XL, go [_here_](https://forms.gle/wpyDzhDyoWLQygAT9) to download it), and a highly scalable training method (called CosPlace), which allows to reach SOTA results with compact descriptors.

The images below represent respectively:
1) the map of San Francisco eXtra Large
2) a visualization of how CosPlace Groups (read datasets) are formed
3) results with CosPlace vs other methods on Pitts250k (CosPlace trained on SF-XL, others on Pitts30k)
<p float="left">
  <img src="https://github.com/gmberton/gmberton.github.io/blob/main/images/SF-XL%20map.jpg" height="200" />
  <img src="https://github.com/gmberton/gmberton.github.io/blob/main/images/map_groups.png" height="200" /> 
  <img src="https://github.com/gmberton/gmberton.github.io/blob/main/images/backbones_pitts250k_main.png" height="200" />
</p>



## Train
After downloading the SF-XL dataset, simply run 

`$ python3 train.py --dataset_folder path/to/sf-xl/processed`

the script automatically splits SF-XL in CosPlace Groups, and saves the resulting object in the folder `cache`.
By default training is performed with a ResNet-18 with descriptors dimensionality 512 is used, which fits in less than 4GB of VRAM.

To change the backbone or the output descriptors dimensionality simply run 

`$ python3 train.py --dataset_folder path/to/sf-xl/processed --backbone resnet50 --fc_output_dim 128`

You can also speed up your training with Automatic Mixed Precision (note that all results/statistics from the paper did not use AMP)

`$ python3 train.py --dataset_folder path/to/sf-xl/processed --use_amp16`

Run `$ python3 train.py -h` to have a look at all the hyperparameters that you can change. You will find all hyperparameters mentioned in the paper.

#### Reproducibility
Results from the paper are fully reproducible, and we followed deep learning's best practices (average over multiple runs for the main results, validation and hyperparameter search on the val set).

## Test
You can test a trained model as such

`$ python3 eval.py --dataset_folder path/to/sf-xl/processed --backbone resnet50 --fc_output_dim 128 --resume_model path/to/best_model.pth`

You can download plenty of trained models below.

## Model Zoo

<details>
     <summary><b>Models with different backbones and dimensionality of descriptors, trained on SF-XL</b></summary></br>
    Pretained networks employing different backbones.</br></br>
	<table>
		<tr>
			<th rowspan=2>Model</th>
			<th colspan=7>Dimension of Descriptors</th>
	 	</tr>
	 	<tr>
	  		<td>32</td>
	   		<td>64</td>
	   		<td>128</td>
	   		<td>256</td>
	   		<td>512</td>
	   		<td>1024</td>
	   		<td>2048</td>
	 	</tr>
		<tr>
			<td>ResNet-18</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>-</td>
			<td>-</td>
	 	</tr>
		<tr>
			<td>ResNet-50</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
	 	</tr>
		<tr>
			<td>ResNet-101</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
	 	</tr>
		<tr>
			<td>ResNet-152</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
	 	</tr>
		<tr>
			<td>VGG-16</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>Coming Soon</td>
			<td>-</td>
			<td>-</td>
	 	</tr>
	</table>
</details>

## Cite
Here is the bibtex to cite our paper
```
@inProceedings{Berton_CVPR_2022_cosPlace,
  author = {Berton, Gabriele and Masone, Carlo and Caputo, Barbara},
  title = {Rethinking Visual Geo-localization for Large-Scale Applications}, 
  booktitle = {CVPR},
  month = {June}, 
  year = {2022}, }
```

## Issues
If you find some problems in our code, or have any advice or questions, feel free to open an issue or send an email to berton.gabri@gmail.com

