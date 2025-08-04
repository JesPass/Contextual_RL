# Contextual RL

Algorithm implementation, data and analysis code for Passlack J & MacAskill A (2025) Contextual inference through flexible integration of environmental features and behavioural outcomes. *bioRxiv* doi: https://doi.org/10.1101/2025.05.28.656607

## How to use

### Running models
In order to run the models using `algorithm.py`:
1) Select the type of models to be run from `cue_SRstart`, `noise_SRstart`, `noise`, `cue`, `fig_1`, `other_noise`, `other_cue` `struct_disc`, `DNMS`, by setting `data_type` in `main` and `chosen_ones` functions.
2) `run_single` will run a single agent, and will save plots from figure 2b to figures. 
3) `run_opts` will run 40 agents for each model either on a cluster with a single node if `run_on = 'single_node'`, or on a cluster with multiple nodes otherwise.

### Analyzing data
In order to process data and generate figures using `analysis.py`:
1) Select the `data_type` to be processed and plotted from `cue_SRstart`, `noise_SRstart`, `noise`, `cue`, `fig_1`, `other_noise`, `other_cue` `struct_disc`, `DNMS`.
2) To analyze and generate figures using provided pre-processed data files saved as `saved_import` in `Data/data_type/` set `saved = True`.
3) To analyze data generated from `algorithm.py` move data generated into the corresponding folder in `Data` and set `saved = False`.

### Summary of data types analyzed for each figure

| Data type     | Description   | Figures    |
| ------------- | ------------- |------------- |
| cue_SRstart  | Feature and joint algorithms for cue-choice distance, saving the SRs during training and during random trials  | 5b,c, 6a-d|
| noise_SRstart  | Feature and joint algorithms for distractors around cue, saving the SRs during training and during random trials  | 2e, 5b,c, 6e |
| cue  | Feature, joint and outcome algorithms for cue-choice distance  |3, 4, 5a,d|
| noise  | Feature, joint and outcome algorithms for distractors around cue  |3, 4, 5a,d |
| fig_1  | TD, SR, SR1, feature and outcome algorithms on initial maze  | 2d |
| DNMS  | Feature and joint algorithms for delayed non-match to sample task | 7b,c  |
| struct_disc  | Feature and joint algorithms for structured paired-associates task |7e,f  |
| other_cue  | Supplementary data for cue-choice distance  | S1, S2  |
| other_noise  | Supplementary data for distractors around cue   | S1, S2  |

