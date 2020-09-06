<img src='imgs/teaser_720.gif' align="right" width=360>

<br><br><br><br>

# pix2pixHD_ACM
## Image-to-image translation at 2k/1k resolution
- Our label-to-streetview results
<p align='center'>  
  <img src='imgs/teaser_label.png' width='400'/>
  <img src='imgs/teaser_ours.jpg' width='400'/>
</p>
- Interactive editing results
<p align='center'>  
  <img src='imgs/teaser_style.gif' width='400'/>
  <img src='imgs/teaser_label.gif' width='400'/>
</p>
- Additional streetview results
<p align='center'>
  <img src='imgs/cityscapes_1.jpg' width='400'/>
  <img src='imgs/cityscapes_2.jpg' width='400'/>
</p>
<p align='center'>
  <img src='imgs/cityscapes_3.jpg' width='400'/>
  <img src='imgs/cityscapes_4.jpg' width='400'/>
</p>

- Label-to-face and interactive editing results
<p align='center'>
  <img src='imgs/face1_1.jpg' width='250'/>
  <img src='imgs/face1_2.jpg' width='250'/>
  <img src='imgs/face1_3.jpg' width='250'/>
</p>
<p align='center'>
  <img src='imgs/face2_1.jpg' width='250'/>
  <img src='imgs/face2_2.jpg' width='250'/>
  <img src='imgs/face2_3.jpg' width='250'/>
</p>

- Our editing interface
<p align='center'>
  <img src='imgs/city_short.gif' width='330'/>
  <img src='imgs/face_short.gif' width='450'/>
</p>

## Prerequisites
- Linux or macOS
- Python 3
- NVIDIA GPU (11G memory or larger) + CUDA cuDNN

## Getting Started
### Installation
- Install PyTorch 1.7+ and dependencies from http://pytorch.org
- Install python libraries [dominate](https://github.com/Knio/dominate).
```bash
pip install dominate
```
- Clone this repo:
```bash
git clone https://github.com/deepglugs/pix2pixHD_ACM
cd pix2pixHD_ACM
```

### Conditional pix2pixHD

This repository has been modified to support "conditional" labels to further 
control the output of the generated images. There are a few extra arguments
added to training and generation to support this feature:

--cond: Use conditional features

--vocab: file with comma-separated list of labels trained with the model. For
example:

```
1girl, dress, black_hair, blonde_hair, shorts
```

--vocab_size: size of the vocabulary. If larger than the list from the vocab file
the vocabulary input will be padded to match this size.

Instance and label map modes are not supported in pix2pixHD_ACM

### Additional modifications over pix2pixHD:

We have added support for Adaptive Instance Normalization, Spectral Normalization
and added optional attention layers to both generator and discriminators.

### Generating

"generate.py" has been created to make generating single, or directories of images easier.

To generate a single image with label:

```bash
python3 generate.py --name name_of_trained_model --netG local --ngf 32 \
                    --image foobar.png --label "1girl, dress, black_hair" \
                    --vocab vocabulary.txt --vocab_size=512
```

"--label" can either be a quoted list of labels separated by comma, or a file.
If "--label" is not specified, generate.py will look for a "foobar.txt" in the
same directory as "--image".

### Testing (deprecated)
- A few example Cityscapes test images are included in the `datasets` folder.
- Please download the pre-trained Cityscapes model from [here](https://drive.google.com/file/d/1h9SykUnuZul7J3Nbms2QGH1wa85nbN2-/view?usp=sharing) (google drive link), and put it under `./checkpoints/label2city_1024p/`
- Test the model (`bash ./scripts/test_1024p.sh`):
```bash
#!./scripts/test_1024p.sh
python test.py --name label2city_1024p --netG local --ngf 32 --resize_or_crop none
```
The test results will be saved to a html file here: `./results/label2city_1024p/test_latest/index.html`.

More example scripts can be found in the `scripts` directory.

### Dataset
- We use the Cityscapes dataset. To train a model on the full dataset, please download it from the [official website](https://www.cityscapes-dataset.com/) (registration required).
After downloading, please put it under the `datasets` folder in the same way the example images are provided.


### Training
In general, pix2pixHD requires a "train_A" and a "train_B" folder within the 
directory specified by "--dataset".  For conditional training (with --cond)
an additional "tags" directory will be expected containing text files with
comma-seprated labels (see "vocab" argument example in "Conditional" section 
above).

The label file in the "tags" subdirectory should have the same filename as the
image in train_A that corresponds with it.

### Conditional training

```bash
python3 train.py --name acm_gen5_cond_local \
                 --label_nc 0 \
                 --no_instance \
                 --serial_batches \
                 --dataroot path/to/dataset \
                 --resize_or_crop scale_width \
                 --ngf 32 --ndf 64 \
                 --batchSize 6 \
                 --nThreads=8 \
                 --netG local \
                 --loadSize 512 \
                 --cond --vocab my_vocab.vocab
```

### global + local network training.

This was described in the original paper, but documentation was lacking in the
original pix2pixHD project. So here's a good example of conditional global
training and local fine tuning:

Train the global network at 256px ngf 64:

```bash
python3 train.py --name acm_gen5_cond_global \
                 --label_nc 0 \
                 --no_instance \
                 --serial_batches \
                 --dataroot path/to/dataset \
                 --resize_or_crop scale_width \
                 --ngf 64 --ndf 64 \
                 --batchSize 20 \
                 --nThreads=8 \
                 --netG global \
                 --loadSize 256 \
                 --cond --vocab my_vocab.vocab
```

Now train the local network loading the global network and finetuning:

```bash
python3 train.py --name acm_gen5_cond_local \
                 --label_nc 0 \
                 --no_instance \
                 --serial_batches \
                 --dataroot path/to/dataset \
                 --resize_or_crop scale_width \
                 --ngf 32 --ndf 64 \
                 --batchSize 6 \
                 --nThreads=8 \
                 --netG local \
                 --loadSize 512 \
                 --cond --vocab my_vocab.vocab \
                 --load_pretrain checkpoints/acm_gen5_cond_global \
                 --niter_fix_global=100
```

This will finetune the local network on top of the global network and
boost the resolution to 512px for 100 epochs.  Note the change in ngf setting.

- Train a model at 1024 x 512 resolution (`bash ./scripts/train_512p.sh`):
```bash
#!./scripts/train_512p.sh
python train.py --name label2city_512p
```
- To view training results, please checkout intermediate results in `./checkpoints/label2city_512p/web/index.html`.
If you have tensorflow installed, you can see tensorboard logs in `./checkpoints/label2city_512p/logs` by adding `--tf_log` to the training scripts.

### Multi-GPU training
- Train a model using multiple GPUs (`bash ./scripts/train_512p_multigpu.sh`):
```bash
#!./scripts/train_512p_multigpu.sh
python train.py --name label2city_512p --batchSize 8 --gpu_ids 0,1,2,3,4,5,6,7
```
Note: this is not tested and we trained our model using single GPU only. Please use at your own discretion.

### Training with Automatic Mixed Precision (AMP) for faster speed
(This has not been proven to work...)
- You can then train the model by adding `--fp16`. For example,
```bash
#!./scripts/train_512p_fp16.sh
python -m torch.distributed.launch train.py --name label2city_512p --fp16
```
In our test case, it trains about 80% faster with AMP on a Volta machine.

### Training at full resolution
- To train the images at full resolution (2048 x 1024) requires a GPU with 24G memory (`bash ./scripts/train_1024p_24G.sh`), or 16G memory if using mixed precision (AMP).
- If only GPUs with 12G memory are available, please use the 12G script (`bash ./scripts/train_1024p_12G.sh`), which will crop the images during training. Performance is not guaranteed using this script.

### Training with your own dataset
- If you want to train with your own dataset, please generate label maps which are one-channel whose pixel values correspond to the object labels (i.e. 0,1,...,N-1, where N is the number of labels). This is because we need to generate one-hot vectors from the label maps. Please also specity `--label_nc N` during both training and testing.
- If your input is not a label map, please just specify `--label_nc 0` which will directly use the RGB colors as input. The folders should then be named `train_A`, `train_B` instead of `train_label`, `train_img`, where the goal is to translate images from A to B.
- If you don't have instance maps or don't want to use them, please specify `--no_instance`.
- The default setting for preprocessing is `scale_width`, which will scale the width of all training images to `opt.loadSize` (1024) while keeping the aspect ratio. If you want a different setting, please change it by using the `--resize_or_crop` option. For example, `scale_width_and_crop` first resizes the image to have width `opt.loadSize` and then does random cropping of size `(opt.fineSize, opt.fineSize)`. `crop` skips the resizing step and only performs random cropping. If you don't want any preprocessing, please specify `none`, which will do nothing other than making sure the image is divisible by 32.

## More Training/Test Details
- Flags: see `options/train_options.py` and `options/base_options.py` for all the training flags; see `options/test_options.py` and `options/base_options.py` for all the test flags.
- Instance map: we take in both label maps and instance maps as input. If you don't want to use instance maps, please specify the flag `--no_instance`.

## Acknowledgments
This code borrows heavily from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD).
