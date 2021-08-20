# Code

## Data Exploration

Data exploration is available in `notebooks/Data_Viz.ipynb`. 

## Training and Submission

1. `train.py`: First, we train two models on the training set: UNet [1] and UNet++ [2] with a MobileNetV2 [3] backend.
2. `notebooks/Generate_Pseudo.ipynb`:
    * We then use their averaged ensemble to generate pseudo-labels on the unlabeled test set.
    * We then create a new training set with these generated pseudo-labels and the existing training set.
4. `train_pseudo_label.py`: We fine-tune the **UNet model weights** obtained from step 1 on this newly created training set. Note that to run this script one _must_ first run the `notebooks/Generate_Pseudo.ipynb` notebook, generate the pseudo-labels and set the dataframe path (`pseudo_df` variable) accordingly inside this script. 
5. Repeat for n rounds.


**Note**: After the first round of pseudo-labeling, we refine our pseudo-labels. This process is again
governed by an averaged ensemble but instead of using just two models we also add the model (trained with pseudo-labels
in the last iteration) to the ensemble. 

> This idea of pseudo-label generation is taken from this talk: [How to cook pseudo-labels](https://www.youtube.com/watch?v=SsnWM1xWDu4).

After completing training for the final round, we do the following:

* Using the `notebooks/Ensemble_Inference.ipynb` notebook, we first generate the initial predictions. We use stacking for ensembling. 
* We then apply Conditional Random Fields [4] on the predictions to enhance segmentation boundaries. This is demonstrated in the `notebooks/Apply_CRF.ipynb` notebook.

### Additional notes on our inference pipeline

* For creating the model ensemble, we use the initially trained UNet and UNet++ models along with the _last_
  fine-tuned UNet model from the pseudo-labeling step.
* To further account for uncertainty and improve our predictions, we apply test-time augmentation using the [`ttach`](https://github.com/qubvel/ttach)
library. 
  
In case you have any difficulties understanding the overall workflow feel free to open an issue on GitHub and we will get back to you. 

## Semi-supervision with approaches from Noisy Student Training and AdaMatch

We explore another avenue of semi-supervision in this work motivated by Noisy Student Training [5] and AdaMatch [6]. You can find the training
approach here in this script: [`nst_pseudo_label.py`]. It does not require cyclical pseudo-labeling but still yields good results. Even though it
does not get us the best results we think this is a promising avenue to explore further. For more details, please refer to the code and our paper. If you
have doubts feel free to open an issue and we will try to clarify further.

## References

[1] Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for Biomedical Image Segmentation. In: Navab N., Hornegger J., Wells W., Frangi A. (eds) Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. MICCAI 2015. Lecture Notes in Computer Science, vol 9351. Springer, Cham. https://doi.org/10.1007/978-3-319-24574-4_28

[2] Zhou, Z., Siddiquee, M., Tajbakhsh, N., & Liang, J. (2018). UNet++: A Nested U-Net Architecture for Medical Image Segmentation. Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support : 4th International Workshop, DLMIA 2018, and 8th International Workshop, ML-CDS 2018, held in conjunction with MICCAI 2018, Granada, Spain, S..., 11045, 3–11. https://doi.org/10.1007/978-3-030-00889-5_1

[3] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov and L. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 4510-4520, doi: 10.1109/CVPR.2018.00474.

[4] Philipp Krähenbühl and Vladlen Koltun. 2011. Efficient inference in fully connected CRFs with Gaussian edge potentials. In <i>Proceedings of the 24th International Conference on Neural Information Processing Systems</i> (<i>NIPS'11</i>). Curran Associates Inc., Red Hook, NY, USA, 109–117.

[5] Q. Xie, M. -T. Luong, E. Hovy and Q. V. Le, "Self-Training With Noisy Student Improves ImageNet Classification," 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 10684-10695, doi: 10.1109/CVPR42600.2020.01070.

[6] Berthelot, David, et al. “AdaMatch: A Unified Approach to Semi-Supervised Learning and Domain Adaptation.” ArXiv:2106.04732 [Cs], June 2021. arXiv.org, http://arxiv.org/abs/2106.04732.
