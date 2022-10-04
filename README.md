# Detection of Groups with Biased Representation in Ranking

## Abstract
Real-life tools for decision-making in many critical domains are based on ranking results. With the increasing awareness of algorithmic fairness, recent works have presented measures for fairness in ranking. Many of those definitions consider the representation of different ``protected groups'', in the top-$k$ ranked items, for any reasonable $k$. 
Given the protected groups, confirming algorithmic fairness is a simple task. However, the groups' definitions may be unknown in advance.

In this paper, we study the problem of detecting groups with biased representation in the top-$k$ ranked items, eliminating the need to pre-define protected groups.
The number of such groups possible can be exponential, making the problem hard. We first formalize the problem, using two different fairness measures: global representation bounds, and proportional representation. Then we present a simple solution that traverses the different groups in the data and reports those with biased representation (by each fairness measure) in the top-$k$ items for each $k$ in a given range. We then leverage this method to efficiently tackle the problem for each of the fairness measures, presenting an additional optimization utilizing a local search. We conclude with an experimental study, showing the scalability of our approach and demonstrating the usefulness of the proposed optimizations.

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







