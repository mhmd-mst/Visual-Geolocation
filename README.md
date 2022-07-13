


## Train
After downloading the SF-xs dataset, simply run 

`$ python3 train.py --dataset_folder path/to/sf-xs/processed`

the script automatically splits SF-XS in CosPlace Groups, and saves the resulting object in the folder `cache`.
By default training is performed with a ResNet-18 with descriptors dimensionality 512 is used, which fits in less than 4GB of VRAM.

If you want to apply augmentation to the datasets with translation models refer `image-to-image translation/README.md` and after augmenting run
`$ python3 train.py --is_aug --dataset_folder path/to/sf-xs/processed`

To change the backbone or the output descriptors dimensionality simply run 

`$ python3 train.py --dataset_folder path/to/sf-xs/processed --backbone resnet50 --fc_output_dim 128`

You can also speed up your training with Automatic Mixed Precision (note that all results/statistics from the paper did not use AMP)

`$ python3 train.py --dataset_folder path/to/sf-xs/processed --use_amp16`

Run `$ python3 train.py -h` to have a look at all the hyperparameters that you can change. You will find all hyperparameters mentioned in the paper.

#### Reproducibility
Results from the paper are fully reproducible, and we followed deep learning's best practices (average over multiple runs for the main results, validation and hyperparameter search on the val set).

## Test
You can test a trained model as such

`$ python3 eval.py --dataset_folder path/to/sf-xs/processed --backbone resnet50 --fc_output_dim 128 --resume_model path/to/best_model.pth`

You can download plenty of trained models below.

## SF-N Dataset
You can download the SF-N dataset using this [link](https://drive.google.com/file/d/1GUA4VVxh389i_FTJ-caVHHx8alr1hmg3/view?usp=sharing). 

