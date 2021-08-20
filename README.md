# Flood Detection Challenge

This repository contains code for our submission to the [ETCI 2021 Competition on Flood Detection](https://competitions.codalab.org/competitions/30440) (Winning
Solution #2). 


Accompanying paper: [Flood Segmentation on Sentinel-1 SAR Imagery with Semi-Supervised Learning](http://arxiv.org/abs/2107.08369).

by Sayak Paul\*, Siddha Ganju\*.

(\*) equal contribution.

<div align="center">
	<img src="https://i.ibb.co/X7chPyT/pipeline.png" width=600/>
</div>


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

**Execution instructions for the scripts (`src`) and notebooks (`notebooks`) are provided in their respective directories.** 

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
* Thanks to [Charmi Chokshi](https://in.linkedin.com/in/charmichokshi), and domain experts Nick Leach and Veda Sunkara for insightful discussions.

## Citation

```
@misc{paul2021flood,
      title={Flood Segmentation on Sentinel-1 SAR Imagery with Semi-Supervised Learning}, 
      author={Sayak Paul and Siddha Ganju},
      year={2021},
      eprint={2107.08369},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
