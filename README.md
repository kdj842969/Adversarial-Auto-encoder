## Synopsis

This is a term project for CSS 590B: Adversarial Machine Learning. The title of the paper is called Enhancing CNNs Robustness Against Adversarial Example Using Auto-encoder. We use auto-encoder as a pre-processing feature reduction method to enhance the robustness of CNNs.
In repository, Adversarial_training.py test original MNIST dataset and adversarial example. It is mainly copied from cleverhans tutorial on fgsm. CAE_training.py test original MNIST dataset with encoded adversarial example. fgsm.py includes the FGSM function API. img folder includes an adverarial example of FGSM. Details see in the paper.

## Installation

To run the code, you need have:
1. Python 2.7
2. Virtualenv
3. numpy
4. tensorflow
5. keras
6. Cleverhans adversarial machine learning library

## Code Example

To run the code:
1. Run Adversarial.py, get original mnist dataset and adversarial example CNN training results.
2. Run CAE_training.py, get encoded adversarial example training results.

## Contributors

Any question, contact yjj801@uw.edu
Same project repository available on https://JingjingY@bitbucket.org/JingjingY/css590b-jingjing.git
