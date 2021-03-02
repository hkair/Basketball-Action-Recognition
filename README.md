# Basketball-Action-Recognition
Spatio-Temporal Classification of 🏀  Basketball Actions using 3D-CNN Models trained on the SpaceJam Dataset.

![Lebron Shoots](examples/lebron_shoots.gif)

## Motivation
Utilizing the SpaceJam Basketball Action Dataset [Repo](https://github.com/simonefrancia/SpaceJam), I aim to create a system that takes a video of a basketball game to classify a given action for each of the players tracked with a bounding box. There are two essential parts for this program: R(2+1)D Model (Can be any 3D CNN architecture) and the player tracker. The deep learning framework used was PyTorch and the machine used to train the model is the Nvidia RTX 3060ti GPU.

This is a demo video from the SpaceJam Repo.

[Demo Video](https://www.youtube.com/watch?v=PEziTgHx4cA)

## Action/Video Classification
A pretrained baseline R(2+1)D CNN (pretrained on kinetics-400 dataset) from [torchvision.models](https://pytorch.org/vision/0.8/models.html) is used and further fine-tuned on the SpaceJam dataset. Any 3D CNN architecture can be used, but for this project it was decided that the R(2+1)D was a perfect balance in terms of number of parameters and overall model performance. It was also shown in the [paper](https://arxiv.org/pdf/1711.11248.pdf) that factorizing 3D convolutional filters into separate spatial and temporal dimensions, alongside residuual learninng yields significant gains in accuracy. The training was done at [train.py](https://github.com/hkair/Basketball-Action-Recognition/blob/master/train.py).

### Dataset
As mentioned above, the SpaceJam Basketball Action Dataset was used to train the R(2+1)D CNN model for video/action classification of basketball actions. The [Repo](https://github.com/simonefrancia/SpaceJam) contains two datasets (clips->.mp4 files and joints -> .npy files) of basketball single-player actions. The size of the two final annotated datasets is about 32'560 examples. Custom dataloaders were used for the basketball dataset in the [dataset.py](https://github.com/hkair/Basketball-Action-Recognition/blob/master/dataset.py).

![alt text](https://raw.githubusercontent.com/simonefrancia/SpaceJam/master/.github/histogram.png)

#### Augmentations
After reading the thesis [Classificazione di Azioni Cestistiche mediante Tecniche di Deep Learning](https://www.researchgate.net/publication/330534530_Classificazione_di_Azioni_Cestistiche_mediante_Tecniche_di_Deep_Learning), (Written by Simone Francia) it was determined that the poorest classes with examples less than 2000 were augmented. Dribble, Ball in Hand, Pass, Block, Pick and Shoot were among the classes that was augmented. Augmentations are applied by running the script [augment_videos.py](https://github.com/hkair/Basketball-Action-Recognition/blob/master/augment_videos.py) and saved in a given output directory. Translation and Rotation were the only augmentations applied to the mentioned classes. After applying the augmentations the dataset has 49901 examples.

##### Rotate
![rotate](examples/0000000_flipped_rotate_330.gif)
##### Translate
![translate](examples/0000000_translate_32_0.gif)

### Training
- Hyperparameters, etc.

### Validation and Evaluation
The final model was a R(2+1)D CNN trained on the additional augmented examples. For validation on the test set, the model at epoch 19 was used as it was the best performing model in terms of validation f1-score and accuracy. The model performs significantly better than the reported 73% in the thesis [Classificazione di Azioni Cestistiche mediante Tecniche di Deep Learning](https://www.researchgate.net/publication/330534530_Classificazione_di_Azioni_Cestistiche_mediante_Tecniche_di_Deep_Learning), acheiving 85% for both validation accuracy and test accuracy. The confusion matrix was attained using the [inference.py](https://github.com/hkair/Basketball-Action-Recognition/blob/master/inference.py) code. Further analysis on predictions and errors are done on [error_analysis.ipynb](https://github.com/hkair/Basketball-Action-Recognition/blob/master/error_analysis.ipynb) notebook.

#### Confusion Matrix 
![confusion matrix](examples/confusion_matrix.png)

#### Inference Examples

##### Shooting
![true_shoot](examples/true_shoot.gif)
![false_shoot](examples/false_shoot.gif)

##### Dribble
![true_dribbble](examples/true_dribble.gif)
![false_dribbble](examples/false_dribble.gif)

##### Pass
![true_pass](examples/true_pass.gif)
![false_pass](examples/false_pass.gif)

##### Dribble
![true_defence](examples/true_defence.gif)
![false_defence](examples/false_defence.gif)

## Player Tracking 
All player tracking is done in [main.py](https://github.com/hkair/Basketball-Action-Recognition/blob/master/main.py). Players are tracked by manually selecting the ROI. In theory, an unlimited amount of people or players can be tracked, but this will significantly increase the compute time. In the example above only 2 players, LeBron James (Offence) & Deandre Jordan (Defence) were tracked.

## Future Additions
- Separate augmented examples from validation and only in training.
- Utilize better player tracking methods. 
- Restrict Box size to 176x128 (Or with similar Aspect Ratio), so resize of image is not applied.
- Fully automate player tracking. Potentially using YOLO or any other Object Detection Models.

## Credits
Major thanks to [Simone Francia](https://github.com/simonefrancia) for the basketball action dataset and paper on action classification with 3D-CNNs. 
