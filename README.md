# Reinforcement Learning Trading Notebook

This repository contains step by step process training RL for trading cryptocurrencies. Please note that most of this project is currently being migrated from private repo to here. There are three main notebooks:

1. **Data** – Generating data for testing via binance download or sythethic data.
2. **Train** – Related to model training with varying training method
3. **Test** – Validate model accuracy

## Overview
![RL Training Process](https://assets.algoboy-kevin.com/rl-trading-concept.png)

## Installation

To use these notebooks, you will need:
- Python 3.10 (or later)
- Jupyter Notebook
- Stablebaselines
- Tensorflow

Additionally, make sure to clone the Binance public data repository into this folder:

```bash
git clone https://github.com/binance/binance-public-data.git
```

## Objective
To find a trading setup using reinforcement learning

## Todo

Book 1 - Data
- [ ] Migrate sinewave script from private repo
- [ ] Generate complete data traing set: sinewave, sideways, real data

Book 2 - Train
- [ ] Complete training script for each training data
- [ ] Display tensorboard graphs on notebook (fitness level)

Book 3 - Test
- [ ] Create evaluate script 

