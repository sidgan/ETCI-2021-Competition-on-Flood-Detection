# Flood Detection Challenge

This repository contains code for our submission to the **ETCI 2021 Competition on Flood Detection** ([Leaderboard](https://competitions.codalab.org/competitions/30440), [Homepage](https://nasa-impact.github.io/etci2021/)) (Winning
Solution #2). 

Accompanying paper: [Flood Segmentation on Sentinel-1 SAR Imagery with Semi-Supervised Learning](http://arxiv.org/abs/2107.08369).

by Sayak Paul\*, Siddha Ganju\*.

(\*) equal contribution.

<div align="center">
	<img src="https://i.ibb.co/X7chPyT/pipeline.png" width=600/>
</div><br>

**Update (December 11, 2021)**: We are releasing a [presentation deck](https://docs.google.com/presentation/d/1CMNK5FuNht_m6owi8lSHd0ksa50JVPeYU88XF39AzA4/edit?usp=sharing), a [presentation video](https://recorder-v3.slideslive.com/?share=52682&s=0258146f-46b5-4696-83a8-60ab25be06c7), and [a poster](https://docs.google.com/presentation/d/1zza2g-9YTsifrq3v10YWz5tbNcsAV0ILANTfuz-IgSI/edit?usp=sharing) of our work. 

**Update (October 26, 2021)**: Our work made it to the following NeurIPS 2021 workshops: [_AI for Science: Mind the Gaps_](https://ai4sciencecommunity.github.io/), [_Tackling Climate Change with Machine Learning_](https://www.climatechange.ai/events/neurips2021), 
[_Women in ML_](https://wimlworkshop.org/sh_events/wiml-neurips2021/), and [_Machine Learning and the Physical Sciences_](https://ml4physicalsciences.github.io/).

## Team 

* [Siddha Ganju](http://sidgan.github.io/siddhaganju)
* [Sayak Paul](https://sayak.dev)


## Executing the code

We executed the scripts and notebooks on a [Vertex AI](https://cloud.google.com/vertex-ai) Notebook instance. The instance has four
Tesla V100 GPUs and its base configuration is [`n1-standard-16`](https://cloud.google.com/compute/docs/machine-types).

We use Python 3.8 and PyTorch 1.9. Apart from the requirements specified in `requirements.txt` you'd need to install the following
as well to run the scripts and notebooks:

```shell
$ pip install git+https://github.com/qubvel/segmentation_models.pytorch
$ pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

Our scripts and notebooks make use of mixed-precision training (via [`torch.cuda.amp`](https://pytorch.org/docs/stable/notes/amp_examples.html)) and distributed training (via [`torch.nn.parallelDistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)). With this combination we are able to achieve significant boosts in the overall model training time.

**Execution instructions for the scripts (`src`) and notebooks (`notebooks`) are provided in their respective directories.** Download the data from the [website](https://competitions.codalab.org/competitions/30440) after agreeing to their license agreements.

## Pre-trained weights

For complete reproducibility, we provide the pre-trained weights [here](https://github.com/sidgan/ETCI-2021-Competition-on-Flood-Detection/releases/download/v1.0.0/pretrained_weights.tar.gz). With these weights and the workflow depicted in the notebooks and scripts one should be able to get an IoU of ~0.76 (as per the [competition leaderboard](https://competitions.codalab.org/competitions/30440#results)) on the test set provided at the competition.

## Results

You can verify the reported results [here](https://competitions.codalab.org/competitions/30440#results). Just switch to "Test (Phase 2)" after
clicking the link.

<div align="center">
	<img src="https://i.ibb.co/q7RWwZB/image.png"/>
</div>

## FAQ

- [Create a new Notebooks instance](https://cloud.google.com/notebooks/docs/create-new)

## Acknowledgements

* We are grateful to the [ML-GDE program](https://developers.google.com/programs/experts/) for providing GCP credits to support our experiments. 
* Thanks to [Charmi Chokshi](https://in.linkedin.com/in/charmichokshi), and domain experts Shubhankar Gahlot, May Casterline, Ron Hagensieker, Lucas Kruitwagen, Aranildo Rodrigues, Bertrand Le Saux, Sam Budd, Nick Leach, and, Veda Sunkara for insightful discussions.

## Citation

```
@inproceedings{paul2021flood,
    title   = {Flood Segmentation on Sentinel-1 SAR Imagery with Semi-Supervised Learning},
    author  = {Sayak Paul and Siddha Ganju},
    year    = {2021},
    URL = {https://arxiv.org/abs/2107.08369},
    booktitle = {NeurIPS Tackling Climate Change with Machine Learning Workshop}
}
```
