# Deep Reinforcement Learning for Energy Management in Residential Housing


This repository contains the code and resources for a seminar project on "Machine Learning for Sequential Decision Making". The project focuses on minimizing carbon emissions within a smart home environment using reinforcement learning methods.

## Abstract

This project provides a novel formulation of the carbon emission minimization problem within a smart home. The environment includes an energy storage system, a heat pump, and a flexible demand. The environment is formulated as a Markov Decision Process and tested using state of the art reinforcement learning methods. The effectiveness of the algorithms is compared to rule-based controllers. In addition, three variations of the environment are explored in an effort to improve the formulation in terms of learnability.

## Paper

The paper detailing the project can be found in the `doc/paper_final` directory.

## Setup

To set up the project, you need a Python environment. The required dependencies can be installed using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Training

To train a new agent, you need to adapt and run the `train.py` script:

```bash
python src/train.py
```

## Analysis

The `analysis.py` script provides functionalities to analyze the results of the training:

```bash
python src/analysis.py
```

Please note that you might need to adjust the paths and parameters in the scripts according to your setup.
