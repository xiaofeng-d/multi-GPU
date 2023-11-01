
### Pytorch FSDP modifications

We're testing FSDP fully sharded model to distribute the parameters and data into multiple GPUs, in order to enlarge the maximum possible training data dimension.

### Pytorch Lightning modifications

We're using the Pytorch Lightning framework for convenience of training with multiple GPUs and possibly tackling larger box and many more particles.


# ML-Recon

## Objective:

ML project to predict Nbody simulation output from initial condition.
Both input and output are particle displacement fields.

## File descriptions:

* `reconLPT2Nbody_uNet.py` : main excute files (with modifications based on FSDP, lightning, etc)
* `periodic_padding.py` : code to fulfill periodic boundary padding
* `data_utils.py` : how to load data + test/analysis
* `model/BestModel.pt` : Best trained model
* `configs/config_unet.json` : most of the hyperparameters
* `Unet/uNet.py` : architecture

## To run the code:

srun --ntasks-per-node=1 --gpus-per-task=1 ./reconLPT2Nbody_uNet_lightning_FSDP.py -c ./configs/config_unet.json 0,1,2,3


## Instruction:

1. Input raw data should be in the format of '0_train.npy','1_train.npy'. The shape of the data
in each file should be `(sample_size,3,dim,dim,dim)`, where the first coloumn is sample size, the
3rd to 5th coloumn is (\phi_x, \phi_y,\phi_z) for ZA. 0 and 1 represents initial snapshot and final snapshot.

2. The output of the model is in the shape of `(6,dim,dim,dim)` where
`(0:3,dim,dim,dim)` stores the predicted fastPM simulations from uNet model and
`(3:6,dim,dim,dim)` stores the corresponding real simulations. Here our PM simulation dimension is dim = 128.

3. The best trained model is stored in `model/BestModel.pt`. All the tests
(pancake, cosmology, etc) should be tested on this model.
You should only change the following parameters in `configs/config_unet.json`
to do different tests:
    * `base_data_path`: tell where the input (LPT/ZA) is stored.
    * `output_path`:  where do you want to store the output

4. The ZA/PM128 data are stored in the directory
on LCRC Swing.
