# hype
Command line hyperparameter tuner.

`hype` works by taking a json map of hyperparameters to tune which can be either categorical (from `[0,n)`) or numeric (from `(a,b)`), then exploring those parameters on a given command to run. The command should read in the hyperparameters as additional arguments, using short flags for single letter parameters (e.g. `-a`) or long flags for multiple letter parameters (e.g. `--learningrate`). It should then return the results to stdout in csv format, with a `score` column indicating the result of the objective function. `hype` will take the minimum of this column (or maximum if `-m` is set) and update the hyperparameter search accordingly to choose the next setting to try. `hype` prints the results of all trials to stdout in csv format. Usage:
```bash
hype -m "`cat example.json`" -- ./example.sh -c -5
```

See `hype -h` for more info. At the moment only one hyperparameter search algorithm is implemented (Tree of Parzen Estimators). This application can be downloaded from the releases page.
