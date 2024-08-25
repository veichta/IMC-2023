# 7th place Solution using a novel matcher LightGlue (2nd for price eligible)
# Intro

This is our solution for the [Image Matching Challenge 2023](https://www.kaggle.com/competitions/image-matching-challenge-2023/overview) where we took 7th place (2nd for price eligible solutions). We used a novel matcher called [LightGlue](https://github.com/cvg/LightGlue), which was developed here at ETH Zurich. LightGlue is a fast and accurate matcher that can be used for image retrieval and image matching. It is based on a transformer architecture and is trained on the MegaDepth dataset. We used LightGlue in combination with other feature extractors such as SIFT, DISK, and ALIKED to create an ensemble of matchers. We also used PixSfM to refine the reconstructions and the hloc toolbox to localize unregistered images. Using this pipeline, we were able to achieve a score of 0.529 (7th) on the private leaderboard while matching the score of 2nd place when using SuperPoint.

| Features                 | Matchers       | Train     | Public    | Private   |
| ------------------------ | -------------- | --------- | --------- | --------- |
| ALIKED*                  | LG             | 0.763     | 0.361     | 0.407     |
| ALIKED+SIFT*             | LG+NN          | 0.594     | 0.434     | 0.480     |
| DISK*                    | LG             | 0.761     | 0.386     | 0.437     |
| DISK+SIFT*               | LG+NN          | **0.843** | 0.438     | 0.479     |
| ALIKED2K+DISK*           | LG+LG          | 0.837     | 0.444     | 0.488     |
| ALIKED2K+DISK+SIFT**     | LG+LG+NN       | 0.837     | **0.475** | 0.523     |
| ALIKED2K+DISK+SIFT       | LG(h)+LG(h)+NN | 0.824     | 0.450     | **0.529** |
| ------------------------ | ------------   | --------- | --------- | --------- |
| DISK+SP                  | LG+LG          | 0.876     | 0.484     | **0.562** |
| DISK+SP*                 | LG+SG          | 0.880     | 0.498     | 0.517     |
| DISK+SIFT+SP*            | LG+NN+LG       | **0.890** | **0.511** | 0.559     |
| DISK+SIFT+SP*            | LG+NN+SG       | 0.867     | T/o       | T/o       |

** was our final submission, * have been submitted after the deadline.

# LightGlue vs SuperGlue
LightGlue is an advanced matching framework developed here at ETH Zurich, which exhibits remarkable efficiency and precision. Its architecture features self- and cross-attention mechanisms, empowering it to make robust match predictions. By employing early pruning and confidence classifications, LightGlue efficiently filters out unmatchable points and terminates computations early, thus avoiding unnecessary processing. LightGlue, along with its training code, is made available under a permissive APACHE license, facilitating broader usage.

In comparison to SuperGlue when combined with SuperPoint, LightGlue demonstrates superior performance in both accuracy and speed. It notably enhances the scores on the train, public, and private datasets while accomplishing these results in nearly half the time required by alternative methods.

| Config | Train | Public | Private |
| ------ | ----- | ------ | ------- |
| SP+SG  | 0.643 | 0.361  | 0.438   |
| SP+LG  | 0.650 | 0.384  | 0.461   |

# Method
We developed a modular pipeline that can be called with various arguments, enabling us to try out different configurations and combine methods very easily. In our pipeline, we made heavy use of hloc, which we used as a starting point.

## Image Retrieval
To avoid matching all image pairs of a scene in an exhaustive manner, we used NetVLAD to retrieve the top k images to construct our image pairs. We also tried out CosPlace but did not observe any notable improvements over NetVLAD. Depending on the configuration of each run, we either used *k=20*, *30* or *50* due to run time constraints. For our final submission, we used *k=30*.

## Feature Extraction
For keypoint extraction, we combined and tried multiple alternatives. For all feature extractions, we experimented with different image sizes but finally settled on resizing the larger edge to 1600 as it provided the most robust scores:

- ALIKED: We played around with a few settings and finally chose to add it to our ensemble as it showed promising results on a few train scenes. We had to limit the number of keypoints to 2048 due to run-time limitations.
- DISK: DISK was the most promising replacement for SP. We tried a few different configurations and finally settled with the default using a max of 5000 keypoints.
- SIFT: Due to its rotation invariance and fast matching, adding sift to our ensemble turned out to boost performance, especially for heritage/dioscuri and heritage/cyprus.
- SP: SuperPoint was the best-performing features extractor in all our experiments, however, we did not choose it for our final submission because of its restrictive license.

## Feature Matching
We used NN-ratio to match SIFT features. For the deep features such as DISK, ALIKED and SP, we trained LightGlue on the MegaDepth dataset.

## Ensembles
The ensembles gave us the biggest boost in the score. It allowed us to run extraction and matching for different configurations and combine the matches of all configurations. This basically gives us the benefits of all used methods. The only drawback is the increased run-time and we thus had to decrease the number of retrievals. Adding SIFT was always a good option because it did not increase the run-time by much while helping to deal with rotations.

## Structure-from-Motion
For the reconstruction, we used PixSfM and forced COLMAP to use shared camera parameters for some scenes.

### Pixel-Perfect-SfM
We added PixSfM (after compiling a wheel for manylinux, following the build pipeline of pycolmap) as an additional refinement step to the reconstruction process. During our experiments, we noted that using PixSfM decreased the score on scenes with rotated images as the S2DNet features are not rotation invariant. We thus only used it if no rotations are found in the scene. Due to the large number of keypoints in our ensemble, we had to use the low memory configuration in all scenes, even on the very small ones.

### Shared Camera Parameters
We noticed that most scenes have been taken with the same camera and therefore decided to force COLMAP to use the same camera for all images in a scene if all images have the same shape. This turned out to be especially valuable on the haiper scenes where COLMAP assigned multiple cameras.

## Localizing Unregistered Images
Some images were not registered, even with a high number of matches to registered ones, possibly because the assumption of shared intrinsics was not always valid. We, therefore, introduced a post-processing step where we used the hloc toolbox to estimate the pose of unregistered images. Specifically, we checked if the camera of an unregistered image is already in the reconstruction database. If that was not the case, we would infer it from the exif data.


# Other things tried

- rotating images → We used an image orientation prediction model to correct for rotations. This worked well on the training set but reduced our score significantly upon submission.
- Inspired by last year's solutions, use cropping to focus matching on important regions between image pairs → Became infeasible as we would have a different set of keypoints for each pair of images used for matching.
- Other feature extractors and matcher such as a reimplementation of SP and dense matchers such as LoFTR, DKM → did not improve results or too slow, also unclear license for SP reimplementation.
- Estimated relative in-plane rotation pairwise from sift matches and then estimated the rotation for each image by propagating the rotation through the maximum spanning tree of pairwise matches. → Worked sometimes on Dioscuri but failed on other scenes.
- Resize for sfm did not help.

# Acknowledgments
We would like to thank Philipp Lindenberger for his awesome guidance, tips, and support. We also want to give a huge credit to his novel matcher LightGlue. We also want to thank the [Computer Vision and Geometry Group, ETH Zurich](https://cvg.ethz.ch) for the awesome project that started all this.

# Links
- [LightGlue Repo](https://github.com/cvg/LightGlue)
- [LightGlue Paper](https://arxiv.org/pdf/2306.13643.pdf)
- [Solution Repo](https://github.com/veichta/IMC-2023)
- [Kaggle Notebook](https://www.kaggle.com/code/alexanderveicht/imc2023-from-repo)

## Per Scene Train Scores

### Heritage

| Features             | Matchers       | Cyprus    | Dioscuri  | Wall      | Overall   |
| -------------------- | -------------- | --------- | --------- | --------- | --------- |
| ALIKED               | LG             | 0.850     | 0.684     | **0.967** | **0.833** |
| DISK                 | LG             | 0.314     | 0.592     | 0.843     | 0.583     |
| ALIKED+SIFT          | LG+NN          | 0.991     | 0.772     | 0.436     | 0.733     |
| DISK+SIFT            | LG+NN          | 0.993     | 0.624     | 0.756     | 0.791     |
| ALIKED2K+DISK        | LG+LG          | 0.792     | 0.712     | 0.930     | 0.811     |
| ALIKED2K+DISK+SIFT** | LG+LG+NN       | **0.993** | **0.802** | 0.595     | 0.796     |
| ALIKED2K+DISK+SIFT   | LG(h)+LG(h)+NN | **0.993** | **0.802** | 0.595     | 0.796     |

### Haiper

| Features             | Matchers       | bike      | chairs | fountain  | Overall   |
| -------------------- | -------------- | --------- | ------ | --------- | --------- |
| ALIKED               | LG             | 0.431     | 0.735  | **0.998** | 0.721     |
| DISK                 | LG             | **0.926** | 0.799  | **0.998** | 0.908     |
| ALIKED+SIFT          | LG+NN          | 0.579     | 0.931  | **0.998** | 0.836     |
| DISK+SIFT            | LG+NN          | 0.917     | 0.929  | **0.998** | 0.948     |
| ALIKED2K+DISK        | LG+LG          | 0.918     | 0.812  | **0.998** | 0.909     |
| ALIKED2K+DISK+SIFT** | LG+LG+NN       | 0.922     | 0.801  | **0.998** | 0.907     |
| ALIKED2K+DISK+SIFT   | LG(h)+LG(h)+NN | 0.920     | 0.934  | **0.998** | **0.951** |

### Urban

| Features             | Matchers       | kyiv-puppet-theater | Overall   |
| -------------------- | -------------- | ------------------- | --------- |
| ALIKED               | LG             | 0.735               | 0.735     |
| DISK                 | LG             | 0.793               | 0.793     |
| ALIKED+SIFT          | LG+NN          | 0.215               | 0.215     |
| DISK+SIFT            | LG+NN          | 0.789               | 0.789     |
| ALIKED2K+DISK        | LG+LG          | 0.742               | 0.742     |
| ALIKED2K+DISK+SIFT** | LG+LG+NN       | 0.806               | 0.806     |
| ALIKED2K+DISK+SIFT   | LG(h)+LG(h)+NN | **0.824**           | **0.824** |
