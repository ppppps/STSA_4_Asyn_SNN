# STSA_4_Asyn_SNN
Pytorch Implementation of *Spatial-Temporal Self-Attention for Asynchronous Spiking Neural Networks, IJCAI 2023* 

It is an implementation of the STS-Transformer on the DVS128 Gesture dataset.

You can download the DVS128 Gesture from [here](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794), and we recommend downloading the DVS128 dataset to 'your_path/dvs128/download'. Then you can change the *root_dir* in the *main_train_dvs128.py* file to 'your_path/dvs128'.

We use a 24G NVIDIA RTX6000 GPU, and set the batch size to 32. This code can also run in parallel on multiple GPUs, we do not recommend setting the batch size too small.

## Paper
[Paper Link](https://www.ijcai.org/proceedings/2023/0344.pdf)

## Citation
```
@inproceedings{ijcai2023p344,
  title     = {Spatial-Temporal Self-Attention for Asynchronous Spiking Neural Networks},
  author    = {Wang, Yuchen and Shi, Kexin and Lu, Chengzhuo and Liu, Yuguo and Zhang, Malu and Qu, Hong},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {3085--3093},
  year      = {2023},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2023/344},
  url       = {https://doi.org/10.24963/ijcai.2023/344},
}
```
