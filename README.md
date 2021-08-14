# hype
Command line hyperparameter tuner.

Hype works by taking a json map of hyperparameters to tune which can be either categorical (from `[0,n)`) or numeric (from `(a,b)`), then exploring those parameters on a given command to run. Usage:
    hype -m "`cat example.json`" -- ./example.sh -c -5

See `hype -h` for more info. At the moment only one hyperparameter search algorithm is implemented (Tree of Parzen Estimators). This application can be downloaded from the releases page.
