# Detection of Groups with Biased Representation in Ranking

## Abstract
Real-life tools for decision-making in many critical domains are based on ranking results. With the increasing awareness of algorithmic fairness, recent works have presented measures for fairness in ranking. Many of those definitions consider the representation of different ``protected groups'', in the top-$k$ ranked items, for any reasonable $k$.
Given the protected groups, confirming algorithmic fairness is a simple task. However, the groups' definitions may be unknown in advance.
In this paper, we study the problem of detecting groups with biased representation in the top-$k$ ranked items, eliminating the need to pre-define protected groups.
The number of such groups possible can be exponential, making the problem hard. 
We propose efficient search algorithms for two different fairness measures: global representation bounds, and proportional representation.
Then we propose a method to explain the bias in the representations of groups utilizing the notion of Shapley values. We conclude with an experimental study, showing the scalability of our approach and demonstrating the usefulness of the proposed algorithms.

## About this repo
This repo contains all the algorithms (Coding/Algorithms), experiments (Coding/Experiments), datasets (InputData), and results (OutputData).



## Requirements
python 3


## How to run
Run any scripts in Experiments folder.







