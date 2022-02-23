# ProFormer: Learning Data-efficient Representations of Body Movement with Prototype-based Feature Augmentation and Visual Transformers

![ProFormer Overview](images/teaser.pdf)

This repository contains the code for ProFormer based on the code from SL-DML



## Requirements

* `pip install -r requirements.txt`
*  install pytorch metric learning library from pip
## Precalculated Representations

The precalculated representations can be downloaded from the following links:

* [NTU RGB+D 120 One-Shot](https://agas.uni-koblenz.de/datasets/sl-dml/ntu_120_one_shot.zip)
* [UTD-MHAD](https://agas.uni-koblenz.de/datasets/sl-dml/utdmhad_one_shot.zip)
* [Simitate](https://agas.uni-koblenz.de/datasets/sl-dml/simitate_one_shot.zip) 

## Quick Start


```
pip install -r requirements.txt
export DATASET_FOLDER="$(pwd)/data"
mkdir -p data/ntu/
wget https://agas.uni-koblenz.de/datasets/sl-dml/ntu_120_one_shot.zip
unzip ntu_120_one_shot.zip -d $DATASET_FOLDER/ntu/ntu_swap_axes_testswapaxes
python train.py dataset=ntu_swap_axis
```
when returning you have to set the dataset folder again:

```
export DATASET_FOLDER="$(pwd)/data"
python train.py dataset=ntu_swap_axis
```

## Training

Note, the following commands require an environment variable `$DATASET_FOLDER` to be existing.

### NTU 120 One-Shot

Training for the NTU 120 one-shot action recognition experiments can be executed like:

`python train.py dataset=ntu_swap_axis`

During development, we suggest using the classes `A002, A008, A014, A020, A026, A032, A038, A044, A050, A056, A062, A068, A074, A080, A086, A092, A098, A104, A110,  A116` as validation classes.

