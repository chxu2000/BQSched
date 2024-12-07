# BQSched
Here are the codes for four different experiments as follows on TPC-DS benchmark with PostgreSQL:

- train_from_scratch
- train_on_clusters
- pre_train
- fine_tune

## Quick Start

Please follow the steps below to run the script for each mode:

1. Set up the Python 3.8.16 environment with `pip install -r requirements.txt`.
2. Replace the files in the installed packages with the corresponding files in the `replace/` folder, where replacing `-` with `/` in the filenames gives you the paths to the target files to replace, starting from the `site-packages/` folder in your own environment.
3. Run the script `python -u train.py --mode {mode_name}`, where you can run different modes by specifying different `mode_name` values.
   - The choices for `mode_name` are `['train_from_scratch', 'train_on_clusters', 'pre_train', 'fine_tune']`.
   - The corresponding configurations for each mode are in files `modes/args/{mode_name}.py` and `modes/config/{mode_name}.ini`.
   - Note that you need to replace the configurations for the underlying DBMS connection with information about your target DBMS (including host, port, user, password, max_worker, etc.), and collect workload information in advance with reference to the contents in the folder `envs/scheduler/cache/your_host/`.
4. Check the runtime logs in the file `outs/{host_postfix}.runtime.out` and the results in the `logs/` folder.

## Reproducing Key Results

Coming soon.

## Citing

Coming soon.

