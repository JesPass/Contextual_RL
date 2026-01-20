# Contextual RL

Algorithm implementation, data and analysis code for Passlack J & MacAskill A (2025) Contextual inference through flexible integration of environmental features and behavioural outcomes. *bioRxiv* doi: https://doi.org/10.1101/2025.05.28.656607

## How to use

### Running models
In order to run the models using `algorithm.py`:
1) Select the type of models to be run by setting `data_type` in `main` and `chosen_ones` functions to generate data in main figures: `cue_SRstart`, `noise_SRstart`, `noise`, `cue`, `fig_1`, `struct_disc`, `DNMS`, `cue_CR_outcome`, `noise_CR_outcome` and supplemental figures: `other_noise`, `other_cue`, `cue_dist`, `noise_dist`, `cue_block_1`, `cue_block_5`, `cue_block_10`, `cue_block_20`, `cue_block_30`, `cue_block_40`, `cue_block_70`, `cue_block_100`, `noise_block_1`, `noise_block_5`, `noise_block_10`, `noise_block_20`, `noise_block_30`, `noise_block_40`, `noise_block_70`, `noise_block_100`, `cue_random`, `cue_prob4`, `cue_prob10`, `cue_prob20`, `cue_prob30`, `cue_prob40`, `noise_random`, `noise_prob4`, `noise_prob10`, `noise_prob20`, `noise_prob30`, `noise_prob40`.
2) `run_single` will run a single agent, and will save plots from figure 2b to figures. 
3) `run_opts` will run 40 agents for each model either on a cluster with a single node if `run_on = 'single_node'`, or on a cluster with multiple nodes otherwise.

### Analyzing data
In order to process data and generate figures using `analysis.py`:
1) Select the `data_type` to be processed and plotted from `cue_SRstart`, `noise_SRstart`, `noise`, `cue`, `fig_1`, `struct_disc`, `DNMS`, `CR_outcome` for main figures and from `other_noise`, `other_cue`, `noise_revisions`, `cue_revisions`, `cue_dist`, `noise_dist`, `noise_train`, `cue_train`, `noise_probs`, `cue_probs` for supplemental figures.
2) To analyze and generate figures using provided pre-processed data files saved as `saved_import` in `Data/data_type/` set `saved = True`.
3) To analyze data generated from `algorithm.py` move data generated into the corresponding folder in `Data` and set `saved = False`.

### Summary of data types analyzed for each figure

| Data type     | Description   | Figures    |
| ------------- | ------------- |------------- |
| cue_SRstart  | Feature and joint algorithms for cue-choice distance, saving the SRs during block and during random trials  | 5b,c, 6a-d|
| noise_SRstart  | Feature and joint algorithms for distractors around cue, saving the SRs during block and during random trials  | 2e, 5b,c, 6e |
| cue  | Feature, joint and outcome algorithms for cue-choice distance  |3, 4, 5a,d|
| noise  | Feature, joint and outcome algorithms for distractors around cue  |3, 4, 5a,d |
| fig_1  | TD, SR, SR1, feature and outcome algorithms on initial maze  | 2d |
| CR_outcome | Outcome algorithm for base task with the CRs saved during block trials | 3e
| DNMS  | Feature and joint algorithms for delayed non-match to sample task | 7b,c  |
| struct_disc  | Feature and joint algorithms for structured paired-associates task |7e,f  |
| other_cue  | Supplementary data for cue-choice distance for other methods of supporting feature algorithms | S2, S10  |
| other_noise  | Supplementary data for distractors around cue for other methods of supporting feature algorithms  | S2, S10  |
| cue_revisions | Feature, joint and outcome algorithms for cue-choice distance, with complete performance and posterior probability data recorded across blocks and random trials | S4, S6, S9, S15|
| noise_revisions | Feature, joint and outcome algorithms for distractors around cue, with complete performance and posterior probability data recorded across blocks and random trials | S3, S5, S8, S15|
| cue_dist | Different lengths of interaction of outcome and feature inference in the joint inference algorithm for cue-choice distance | S7 |
| noise_dist | Different lengths of interaction of outcome and feature inference in the joint inference algorithm for distractors around cue | S7 |
| cue_train | Different block lengths for cue-choice distance, consisting of data from `cue_block_x` | S11 |
| noise_train| Different block lengths for distractors around cue, consisting of data from `noise_block_x` | S12 |
| cue_probs | Different probabilistic blocks for cue-choice distance, consisting of data from `cue_prob_x` and  `cue_random` | S13 |
| noise_probs | Different probabilistic blocks for distractors around cue, consisting of data from `noise_prob_x` and  `noise_random` | S14 |
