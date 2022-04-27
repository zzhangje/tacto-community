# fcrn
A tactile-height map estimation network.

## Write a data loading file
- `data_preprocess/gen_data_fcrn.py` is used to generate data loading files for training/validation/testing.

## Test with well-trained model
- `train.py` Trains with the heightmaps and contact masks 
- `test_dataset.py` is testing on cpu. Remember to change the data loading path and result saving path.
