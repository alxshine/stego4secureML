# Requirements
We tested the scripts on a Ubuntu 16.04 machine using python3 and tensorflow 1.9.0 on a GPU, but they should run on similar setups.

# Setup
We added the [original mnist\_challenge](https://github.com/MadryLab/mnist_challenge) code as a submodule.
Before running our scripts you will have to initialize that repository first.
The command for this is: `git submodule update --init`

# Generating the data
We wrote a number of utility scripts to recreate our results very easily. To generate all data used in [our EUSIPCO paper](paper/SSPB2018_EUSIPCO.pdf), you have to run the following scripts in the order they are listed.
Scripts 02 and 05 can take quite a long time (05 takes 3 hours on our test machine).
- `01_download.py`: downloads the *natural* and *secret* model from [mnist\_challenge](https://github.com/MadryLab/mnist_challenge).
- `02_automated_test_generation.py`: creates adversarial examples for both networks, and saves the labels assigned to them by the respective target networks.
- `03_generate_estimates.py`: runs our estimator kernel on the adversarial images, saving the estimates so we do not have to recalculate them later on.
- `04_threshold_generation.py`: generates the *min*, *max*, and *mean* of the differences between the benign training images and our estimates of them.
	We use the *max* as threshold for detecting adversarial examples in order to make false positives extremely unlikely (see Section III.B in [the paper](paper/SSPB2018_EUSIPCO.pdf)).
- `05_diff_generation.py`: calculates the difference between the adversarial examples and our estimates generated from them, saving the differences for analysis.

# Analysis 
The following scripts generate the data used for the plots in [the paper](paper/SSPB2018_EUSIPCO.pdf), in "[TikZ](https://sourceforge.net/projects/pgf/) friendly" tab separated format. All data is in dependence on epsilon, except for the plot created by `07_adv_sample_comparison.py`.
 - `06_changed_pixels_plot_generation.py`: generates data for a plot of the percentage of modified pixels.
 - `07_adv_sample_comparison.py`: shows a comparison between a benign image, and the adversiarial images generated for the *natural* and *secret* model.
 - `08_accuracy_plot_generation.py`: generates data for a plot of the precentage of correctly classified adversarial images, evaluated on the targeted model.
 - `09_detection_evaluation.py`: generates data for a plot for the percentage of images either correctly classified by the target model, or flagged as adversarial by our method (see Figure 3 in [the paper](paper/SSPB2018_EUSIPCO.pdf)).
 - `10_diff_comparison.py`: generates data for a plot of the minimum difference between the adversarial sample and the benign image.
 - `11_tpr_extract.py`: generates data for a plot for the True Positive Rate (TPR) for our method (see Figure 2 in [the paper](paper/SSPB2018_EUSIPCO.pdf)).
