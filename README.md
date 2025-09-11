# Targeted Physical Evasion Attacks in the Near-Infrared Domain 

The idea of this framework is to take traffic sign images of a given dataset and to perturb the signs surface to cause a (un-)targeted misclassification against classification and detection models. The perturbations consist of square patches that are color transformed to simulate an infrared light source. To optimize towards the misclassification goal, we use black-box optimization strategies which are at the heart of this framework.

Please refer to the Artifact Appendix for additional details. In contrast to the Appendix, the relevant Mapillary data is now included within this repository and besides the segmentation model no additional data needs to be downloaded.

## Setup

### Requirements

Create a fresh conda environment:
```bash
conda create --name artifact python=3.9.19
```

Activate the environment with:
```bash
conda activate artifact
```

Install requirements:
```bash
pip install -r requirements.txt
```
### Dataset
Go to the root of the project and execute the following command to combine the part-files (and remove them after) of the dataset into the .pkl file used by the framework.

```
cd dataset/detection/Mapillary; cat mapillary_subset.a mapillary_subset.b > mapillary_subset.pkl; rm mapillary_subset.a mapillary_subset.b
```

### Models
Download the SegmentAnything model ViT-L (`sam_vit_l_0b3195.pth`) and place it in:
```
model/segmentation
```
### Other
Make sure that the environment is activated for all experiments and that each file is executed from the root of the project. Prepend `PYTHONPATH=$(pwd)` before the provided command lines, in case module import errors are encountered.
Device configuration (CUDA/MPS/CPU) is handled automatically by the `get_device()` function in `utils/utils.py`. 

**Only for CUDA acceleration**:

- Execution of experiments: always the first GPU listed in `nvidia-smi` is selected. This index can be changed in the variable `self.gpus` of `experiments/Experiment.py`. 

- Manual interactions with the framework: it is recommended to prepend the environment variables for selecting the GPU as follows:
```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python infrared_perturbation.py ...
```


## Experiments 

All experiments are in `experiments` and are numbered in line with the Artifact Appendix. They consist of a wrapper which schedules invocations of the underlying framework with different parametrizations, which are the basis for most of the reported results in the paper. 
Any log output of these runs is logged into the `logs` folder once a process terminates. The results and plots for each respective experiment are written into its experiment folder to keep everything nicely organized. 

Each of the experiments consists of one (`eX.py`) or two files (`eX_{classification, detection}.py`) to test classification/two-stage and detection/single-stage pipelines separately. 
The `X` resembles the number of the experiment defined in the Artifact Appendix. The execution for each experiment is structured in the same way. 

An experiment can be run with `python experiments/eX/eX.py --mode run`, evaluated/plotted with `python experiments/eX/eX.py --mode evaluate`. 
As the experiments can consume quite some time, the argument `--subset` can be used in combination with `--mode run` to only execute a representative subset of experiments, i.e., one parameter or reduced number of image samples for each dimension which does *not* change program behavior to facilitate functionality checks. 

Especially for a reduced number of samples the results do not match the results in the paper, as they are based on averages over all samples. The evaluation/plotting always considers only the data which is found (subset/full data/data of a running experiment). 
Experiments can also be terminated manually anytime to evaluate the data until then.

## Framework

In the following, we provide additional details on the framework, which are however *not* needed to conduct the experiments. 

## Structure

```text
.
├── README.md
├── requirements.txt
├── infrared_perturbation.py                // Main file for classification/two-stage attacks. 
├── infrared_perturbation_od.py             // Main file for detection/single-stage attacks.
├── experiments                             // Different experiments defined in the Artifact Appendix
│   ├── Experiment.py                       // Wrapper for running experiments
│   ├── Scheduler.py                        // Schedules grid searches in multiple threads (and on multiple GPUs)
│   ├── e1
│   ├── e2_classification
│   ├── e2_detection
│   ├── e3_classification
│   ├── e3_detection
│   ├── e4
│   ├── e5
│   ├── e6_classification
│   ├── e6_detection
│   ├── e7
│   └── e8
├── attacks                                 // Attack classes for attacks on classification/detection.
│   ├── Attack.py
│   ├── ClassificationAttack.py
│   └── DetectionAttack.py
├── dataset                                 // Datasets 
│   ├── classification                      // Datasets for classification, including adversarially augmented datasets.
│   │   ├── gtsrb.py
│   │   ├── gtsrb_test.pkl
│   │   ├── lisa.py
│   │   ├── lisa_test.py
│   │   ├── classes.json
│   │   └── traffic_sign_dataset.py
│   ├── detection                           // Datasets for detection, including adversarially augmented datasets.
│   │   ├── FasterRCNN.py
│   │   ├── GTSDB
│   │   ├── GTSDB.yaml
│   │   ├── Mapillary.yaml
│   │   └── gtsdb.py
│   ├── gtsrb-ir-100                        // GTSRB-IR-100 dataset proposed in the paper
├── model                                   // Model files for classification, detection, and segmentation
│   ├── classification
│   │   ├── GtsrbCNN.py
│   │   ├── LisaCNN.py
│   │   ├── model_gtsrb_ConvNeXt.pth
│   │   ├── model_gtsrb_GtsrbCNN.pth
│   │   ├── model_gtsrb_GtsrbCNN_adv_retrain.pth
│   │   ├── model_gtsrb_ResNet50.pth
│   │   ├── model_gtsrb_SwinTransformer.pth
│   │   ├── model_lisa_LisaCNN.pth
│   │   └── model_lisa_LisaCNN_adv_retrain.pth
│   ├── detection
│   │   ├── FasterRCNN_GTSDB.pth
│   │   └── yolo_mapillary.pt
│   └── segmentation
│       └── sam_vit_l_0b3195.pth
├── optimizers                              // Different optimization strategies
│   ├── EvolutionStrategy.py
│   ├── GeneticAlgorithm.py
│   ├── LocalRandomSearch.py
│   ├── Optimizer.py
│   ├── RandomBaseline.py
│   └── ParticleSwarmOptimization.py
├── perturbations                           // Implementation of our pixel perturbation
│   ├── AdversarialPerturbation.py
│   └── PixelPerturbation.py
└── utils                                   // Utilities
    ├── cnn_utils.py
    ├── ir_utils.py
    ├── od_utils.py
    ├── sign_masks.pkl
    └── utils.py
```

## Classification/Two-Stage Pipeline: infrared_perturbation.py
### Command-Line Arguments

```text
usage: infrared_perturbation.py [-h] [--dataset {GTSRB,LISA}] [--optimizer {pso,es,ga,lrs,rnd}] [--mesh_count MESH_COUNT] [--target_model {GtsrbCNN,ResNet50,ConvNeXt,SwinTransformer,LisaCNN}] [--adv_retrained]
                                [--patch_count PATCH_COUNT] [--max_queries MAX_QUERIES] [--target_mapping TARGET_MAPPING] [--lux LUX] [--save_dir SAVE_DIR] [--enforce_query_budget] [--instances_per_class INSTANCES_PER_CLASS]
                                [--seed SEED]

Adversarial attack by infrared perturbations

optional arguments:
  -h, --help            show this help message and exit
  --dataset {GTSRB,LISA}
                        the target dataset should be specified for a digital attack
  --optimizer {pso,es,ga,lrs,rnd}
                        The optimizer to use.
  --mesh_count MESH_COUNT
                        The number of columns / row in which to split the image to locate pixels (only applicable if pixel perturbation type)
  --target_model {GtsrbCNN,ResNet50,ConvNeXt,SwinTransformer,LisaCNN}
                        Select the classifier architecture to attack.
  --adv_retrained       Set this flag if you want to load the adversarially retrained version of the target model
  --patch_count PATCH_COUNT
                        number of perturbations
  --max_queries MAX_QUERIES
                        Maximum number of queries of the target model (per start of the optimizer)
  --target_mapping TARGET_MAPPING
                        Specify one or multiple desired mis-classification label mappings. See README.md for more details.
  --lux LUX             The intensity of the ambient light to assume for the IR attack [Lux]
  --save_dir SAVE_DIR   Set the path in which the results of the attack will be stored
  --enforce_query_budget
                        Set this flag if you want to enforce the given query budget to be fully used
  --instances_per_class INSTANCES_PER_CLASS
                        Set this value to set the number of instances per class to evaluate on. Only makes a difference for a digital attack on GTSRB (as the LISA dataset is way to small anyway).
  --seed SEED           Specify a seed for all PRNGs explicitly
```

#### Example 1: Digital Infrared Attack

The following command executes our infrared attack on the GTSRB dataset while assuming an ambient light intensity of 10 lux for the used infrared transformation.
The generated digital adversarial examples will be saved to `./output/digital/`.

```shell
python3 infrared_perturbation.py --lux 10 --save_dir ./output/digital/ 
```

```text
Image 0
Best loss -0.5797 was successful.
Attack successful with predicted label 8.
Image 5
Best solution: -0.1023 was successful.
Attack successful with predicted label 25.
...
Image 9764
Best solution: -0.1410 was successful.
Attack successful with predicted label 2.
Image 9739
Best solution: -0.6087 was successful.
Attack successful with predicted label 6.
Attack success rate: 0.982
Average number of queries: 35.541
```

#### Example 2: Change the number of patches (k)

```shell
python3 infrared_perturbation.py --patch_count 192
```

#### Example 3: Change the width of patches (l)

The perturbation pixel size is determined as $\text{image_width} / \text{mesh_count}$.
So, the perturbation pixels are aligned at a grid with `mesh_count` positions in x- and `mesh_count` positions in
y-direction.

Hence, the following command simulates perturbation pixels of size `1x1` for the GTSRB dataset (as the test images have
size `32x32`).

```shell
python3 infrared_perturbation.py --mesh_count 32
```

#### Example 4: Specify a `target_mapping`

The `target_mapping` parameter is a versatile parameter to control the behaviour of the attack.
It contains a **list of two-value tuples**:

```python
[(a, b), (b, c), ...]
```

Each of these tuples specifies a desired misclassification from one class to another.
The possible values for the values in the tuple depend on the position in the tuple.
We refer to the first tuple value as the *source class label* while the second is the *target class label*.

**Source labels:**

Possible values are:

- *a numerical class label*
    - identify the class for which this mapping applies by its label
    - Example: `0` for speed limit 20 in the GTSRB dataset
- *a wildcard class label* `*`:
    - Means that this mapping applies to all classes
    - Important: This can serve as a fallback. Explicit class labels always take precedence.

**Target labels:**

For the target label one of the following is possible:

- *a numerical class label*
    - execute a *targeted* attack and use that label as the target label
    - Example: `14` for stop sign in the GTSRB dataset
- *a wildcard class label* `'*'`:
    - execute an *untargeted* attack for the given source class
- *the automatic targeted attack* `'auto'`:
    - *First* execute an untargeted attack. *Afterward*, execute a targeted attack against the class that seems to be
      most promising.

**Examples:**

Execute an untargeted attack for all images in the dataset:

```shell
python3 infrared_perturbation.py --target_mapping "[('*','*')]"
```

Try to map all speed limits to stop signs:

```shell
python3 infrared_perturbation.py --dataset GTSRB --target_mapping "[(0,14),(1,14),(2,14),(3,14),(4,14),(5,14),(7,14),(8,14)]"
```

## Detection/Single-Stage Pipeline: infrared_perturbation_od.py
### Command-Line Arguments
```text
usage: infrared_perturbation_od.py [-h] [--dataset {GTSDB,Mapillary}] [--optimizer {pso,es,ga,lrs,rnd}] [--mesh_count MESH_COUNT] [--patch_count PATCH_COUNT] [--max_queries MAX_QUERIES] [--lux LUX] [--save_dir SAVE_DIR]
                                   [--store_predictions] [--target_mapping TARGET_MAPPING] [--enforce_query_budget] [--seed SEED] [--instances INSTANCES]
                                   {transferability} ...

Adversarial object detection attack by infrared perturbations

positional arguments:
  {transferability}

optional arguments:
  -h, --help            show this help message and exit
  --dataset {GTSDB,Mapillary}
                        the target dataset should be specified for a digital attack
  --optimizer {pso,es,ga,lrs,rnd}
                        The optimizer to use
  --mesh_count MESH_COUNT
                        The number of columns / row in which to split the image to locate pixels (only applicable if pixel perturbation type)
  --patch_count PATCH_COUNT
                        Number of perturbations (k)
  --max_queries MAX_QUERIES
                        Maximum number of queries of the target model (per start of the optimizer)
  --lux LUX             The intensity of the ambient light to assume for the infrared attack [Lux]
  --save_dir SAVE_DIR   Set the path in which the results of the attack will be stored
  --store_predictions   Set this flag to store the ground-truth prediction and the prediction on the adversarially perturbed images alongside the perturbed images
  --target_mapping TARGET_MAPPING
                        Specify one or multiple desired mis-classification label mappings. See README for more details.
  --enforce_query_budget
                        Set this flag if you want to enforce the given query budget to be fully used
  --seed SEED           Specify a seed for all PRNGs explicitly
  --instances INSTANCES
                        Reduces the number of files the experiment is executed on.
```