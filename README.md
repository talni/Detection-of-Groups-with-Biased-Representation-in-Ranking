# Detection of Unfairly Treated Groups in Classification and Ranking

## Introduction
Machine learning (ML) tools are widely used in many real-life everyday applications. With their ubiquity use in recent years, we
also witness an increase in the reported cases where these tools discriminate unfairly. This, in turn, has given rise to increasing
interest in the study of algorithmic fairness. Fairness definitions usually refer to a given “protected group” in the data, which is
defined based on the values of some sensitive attributes. Given a protected group, confirming algorithmic fairness is a simple task.
However, the group definition may be unknown in advance.

We design an algorithm to detect any groups that are treated unfairly in classification and ranking algorithms,
eliminating the need to define sensitive attributes. For the problem in classification, we propose a heuristic
solution that employs pruning to significantly reduce the search space.
Then we leverage this method to efficiently tackle the problem in the context of ranking, presenting an additional optimization
utilizing a local search. 


## About this repo
This repo contains all the algorithms, experiments and datasets.
All algorithms are in directory Coding/Algorithms, experiments are in Coding/Experiments, and datasets in InputData.



## Requirements
python 3


## How to run
There are two examples reproducing results from ProPublica: Example/fp.py and Example/fn.py, which corresponds to false positive and false negative error rates, respectively.

Suppose you've already cloned this repository and start from the Example directory.

You just need:

    $ python3 fp.py
    $ python3 fn.py
    

Results are stored in file fp_greater_than.txt and fp_greater_than.txt







