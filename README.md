# Offline Policy Comparison under Limited Historical Agent-Environment Interactions
This is the official reopsitory for the paper ''Offline Policy Comparison under Limited HistoricalAgent-Environment Interactions'' that is available at *link_to_arxiv*.


### Requirements
Requirements are listed in `requirements.txt` and can be installed via the command
```
pip install -r requirements.txt
```


### Running Experiments
The presented numerical experiments can be reproduced by running the following commands from the root directory of the project:
```bash
# Example 1 from Section 5.1
python -m run_synthetic_data
# Example 2 from Section 5.2
python -m run_classification
# Example 3 from Section 5.3
python -m run_rl_environments
```
Each example takes several hours to complete on a personal laptop.


### Configuration
The optional `-d` flag specifies the reward function/dataset/environment for each example.
The available arguments are
```
run_synthetic_data:
	* 1 -- reward function defined by (10)
	* 2 -- reward function defined by (11)

run_classification:
	* abalone -- Abalone Dataset
	* algerian -- Algerian Forest Fires Dataset
	* ecoli -- Ecoli Dataset
	* glass -- Glass Identification Dataset
	* winequality -- Wine Quality Dataset

run_rl_environments:
	* main -- environments reported in Section 5.3
	* all -- environments reported in Section C.3
	* InvertedPendulumBulletEnv-v0
	* InvertedPendulumSwingupBulletEnv-v0
	* ReacherBulletEnv-v0
	* Walker2DBulletEnv-v0
	* HalfCheetahBulletEnv-v0
	* AntBulletEnv-v0
	* HopperBulletEnv-v0
	* HumanoidBulletEnv-v0
```

### Reproducibility
Our numerical results are completely reproducible and determined by the value of the random seed.
The optional flag `-s` specifies the random seed; the default value is `2021` for all examples.


### Save and Load
This code supports saving and loading functionality as follows:
* `-save` flag records the result of the experiment in a pickle file in the `./save/` directory
* `-load` flag loads the recording from the `./save/` directory and reports the result of the experiment


### Experiment Recordings
The recordings of the presented numerical examples, obtained via the `-save` flag, are available at <https://www.dropbox.com/s/xsx1bqx9c9hfcel/save.zip?dl=0> (329MB).
To load the data and recreate the presented pictures, download and extract `save.zip` to the root directory of the project and use the `-load` flag, i.e.
```bash
# load Example 1.1 from Sections 5.1.1 and C.1.1
python -m run_synthetic_data -d 1 -load
# load Example 1.2 from Sections 5.1.2 and C.1.2
python -m run_synthetic_data -d 2 -load
# load Example 2 from Sections 5.2 and C.2
python -m run_classification -load
# load Example 3 from Section 5.3
python -m run_rl_environments -load
# load additional environments for Example 3 from Section C.3
python -m run_rl_environments -d all -load
```
