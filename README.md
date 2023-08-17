# SaFeTS-master
This repository is the code involved in the paper Semantic Feature-Based Test Selection for Deep Neural Networks: A Frequency Domain Perspective.

## Prerequisite
The code should be run using python 3.8.5.
``` 
pip install requirements.txt
```

### File structure
* Empirical_Study: The experiments of Section 3 in the paper.
* SVHN: The experiments on SVHN dataset.
* Cifar10: The experiments on Cifar10 dataset.

## To run

```
    # turn to dataset files
    cd Cifar10/
    #generate adversarial examples
    python adv_gen.py
    #sample test cases from candidate set
    ./run_sample.sh
    #retrain model by selected dataset
    ./run_retrain.sh
```

## Large file download
Due to the repository's limitations on document size, we uploaded the dataset and model, as well as other large files, in a cloud drive at the following link：
https://pan.baidu.com/s/1WjzqAxTpwU-96ZXYOBeIGA?pwd=hlay 

password：hlay 
