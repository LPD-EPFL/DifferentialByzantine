# coding: utf-8
###
 # @file   reproduce.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Reproduce the (missing) experiments and plots.
###

import tools
tools.success("Module loading...")

import argparse
import pathlib
import signal
import sys
import traceback

import torch

import experiments

# ---------------------------------------------------------------------------- #
# Miscellaneous initializations
tools.success("Miscellaneous initializations...")

# "Exit requested" global variable accessors
exit_is_requested, exit_set_requested = tools.onetime("exit")

# Signal handlers
signal.signal(signal.SIGINT, exit_set_requested)
signal.signal(signal.SIGTERM, exit_set_requested)

# ---------------------------------------------------------------------------- #
# Command-line processing
tools.success("Command-line processing...")

def process_commandline():
  """ Parse the command-line and perform checks.
  Returns:
    Parsed configuration
  """
  # Description
  parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument("--data-directory",
    type=str,
    default="results-data",
    help="Path of the data directory, containing the data gathered from the experiments")
  parser.add_argument("--plot-directory",
    type=str,
    default="results-plot",
    help="Path of the plot directory, containing the graphs traced from the experiments")
  parser.add_argument("--devices",
    type=str,
    default="auto",
    help="Comma-separated list of devices on which to run the experiments, used in a round-robin fashion")
  parser.add_argument("--supercharge",
    type=int,
    default=1,
    help="How many experiments are run in parallel per device, must be positive")
  parser.add_argument("--only-plot",
    default=False,
    action="store_true",
    help="Only build the plots (useful to get some plots while experiments are still running)")
  # Parse command line
  return parser.parse_args(sys.argv[1:])

with tools.Context("cmdline", "info"):
  args = process_commandline()
  # Check the "supercharge" parameter
  if args.supercharge < 1:
    tools.fatal(f"Expected a positive supercharge value, got {args.supercharge}")
  # Make the result directories
  def check_make_dir(path):
    path = pathlib.Path(path)
    if path.exists():
      if not path.is_dir():
        tools.fatal(f"Given path {str(path)!r} must point to a directory")
    else:
      path.mkdir(mode=0o755, parents=True)
    return path
  args.data_directory = check_make_dir(args.data_directory)
  args.plot_directory = check_make_dir(args.plot_directory)
  # Preprocess/resolve the devices to use
  if args.devices == "auto":
    if torch.cuda.is_available():
      args.devices = list(f"cuda:{i}" for i in range(torch.cuda.device_count()))
    else:
      args.devices = ["cpu"]
  else:
    args.devices = list(name.strip() for name in args.devices.split(","))

# ---------------------------------------------------------------------------- #
# Serial preloading of the dataset
tools.success("Pre-downloading datasets...")

# Pre-load the datasets to prevent the first parallel runs from downloading them several times
with tools.Context("dataset", "info"):
  for name in ("svm-phishing",):
    with tools.Context(name, "info"):
      experiments.make_datasets(name, 1, 1)

# ---------------------------------------------------------------------------- #
# Run (missing) experiments
if not args.only_plot:
  tools.success("Running experiments...")

# Command maker helper
def make_command(params):
  cmd = ["python3", "-OO", "train.py"]
  cmd += tools.dict_to_cmdlist(params)
  return tools.Command(cmd)

# Jobs
jobs  = tools.Jobs(args.data_directory, devices=args.devices, devmult=args.supercharge)
seeds = jobs.get_seeds()

# Base parameters for the experiments
params_common = {
  "loss": "mse",
  "learning-rate": 2,
  "criterion": "sigmoid",
  "momentum": 0.99,
  "evaluation-delta": 50,
  "nb-steps": 1000,
  "nb-workers": 11,
  "nb-decl-byz": 5,
  "nb-real-byz": 5,
  "batch-size-test": 59,
  "test-repeat": 45,
  "gradient-clip": 0.01,
  "privacy-delta": 1e-6 }

# Submit all the experiments (if not disabled)
if not args.only_plot:
  for ds, dsa in (("svm-phishing", None),):
    for md, mda in (("simples-logit", "din:68"),):
      for gar, attacks in (("average", (("nan", None),)), ("brute", (("little", ("factor:1.5", "negative:True")), ("empire", "factor:1.1")))):
        for attack, attargs in attacks:
          for epsilon in (None, 0.1, 0.2, 0.5):
            for batch_size in (10, 25, 50, 100, 250, 500):
              name = f"{ds}-{md}-{gar}-{attack}-e_{'inf' if epsilon is None else epsilon}-b_{batch_size}"
              # Submit experiment
              params = params_common.copy()
              if gar == "average":
                # Disable attack for 'average' GAR
                params["nb-real-byz"] = 0
              params["dataset"] = ds
              params["dataset-args"] = dsa
              params["model"] = md
              params["model-args"] = mda
              params["gar"] = gar
              params["attack"] = attack
              params["attack-args"] = attargs
              params["privacy"] = epsilon is not None
              params["privacy-epsilon"] = epsilon
              params["batch-size"] = batch_size
              jobs.submit(name, make_command(params))

# Wait for the jobs to finish and close the pool
jobs.wait(exit_is_requested)
jobs.close()

# Check if exit requested before going to plotting the results
if exit_is_requested():
  exit(0)

# ---------------------------------------------------------------------------- #
# Produce graphs

# Import additional modules
try:
  import histogram
  import numpy
  import pandas
except ImportError as err:
  tools.fatal(f"Unable to plot results: {err}")

# Map gar name in code to key in graph
gar_to_legend = {"brute": "MDA"}

def compute_avg_err(name, *cols, avgs="", errs="-err"):
  """ Compute the average and standard deviation of the selected columns over the given experiment.
  Args:
    name Given experiment name
    ...  Selected column names (through 'histogram.select')
    avgs Suffix for average column names
    errs Suffix for standard deviation (or "error") column names
  Returns:
    Data frames, each for the computed columns
  """
  # Load all the runs for the given experiment name, and keep only a subset
  datas = tuple(histogram.select(histogram.Session(args.data_directory / f"{name}-{seed}"), *cols) for seed in seeds)
  # Make the aggregated data frames
  def make_df(col):
    nonlocal datas
    # For every selected columns
    subds = tuple(histogram.select(data, col).dropna() for data in datas)
    res   = pandas.DataFrame(index=subds[0].index)
    for col in subds[0]:
      # Generate compound column names
      avgn = col + avgs
      errn = col + errs
      # Compute compound columns
      numds = numpy.stack(tuple(subd[col].to_numpy() for subd in subds))
      res[avgn] = numds.mean(axis=0)
      res[errn] = numds.std(axis=0)
    # Return the built data frame
    return res
  # Return the built data frames
  return tuple(make_df(col) for col in cols)

with tools.Context("plot", "info"):
  # Plot all the experiments
  for ds, dsa in (("svm-phishing", None),):
    for md, mda in (("simples-logit", "din:68"),):
      for epsilon in (None, 0.1, 0.2, 0.5):
        for batch_size in (10, 25, 50, 100, 250, 500):
          legend = list()
          results = list()
          # Pre-process results for all available combinations of GAR and attack
          for gar, attacks in (("average", (("nan", None),)), ("brute", (("little", ("factor:1.5", "negative:True")), ("empire", "factor:1.1")))):
            for attack, _ in attacks:
              name = f"{ds}-{md}-{gar}-{attack}-e_{'inf' if epsilon is None else epsilon}-b_{batch_size}"
              key = f"{gar_to_legend.get(gar, gar.capitalize())} ({'no attack' if gar == 'average' else attack})"
              legend.append(key)
              results.append(compute_avg_err(name, "Accuracy", "Average loss"))
          # Plot top-1 cross-accuracy
          plot = histogram.LinePlot()
          for crossacc, _ in results:
            plot.include(crossacc, "Accuracy", errs="-err", lalp=0.8)
          plot.finalize(None, "Step number", "Cross-accuracy", xmin=0, xmax=1000, ymin=0, ymax=1, legend=legend)
          plot.save(args.plot_directory / f"{ds}-{md}-e_{'inf' if epsilon is None else epsilon}-b_{batch_size}.png", xsize=3, ysize=1.5)
          # Plot average loss
          plot = histogram.LinePlot()
          for _, avgloss in results:
            plot.include(avgloss, "Average loss", errs="-err", lalp=0.8)
          plot.finalize(None, "Step number", "Average loss", xmin=0, xmax=1000, ymin=0, ymax=.6, legend=legend)
          plot.save(args.plot_directory / f"{ds}-{md}-e_{'inf' if epsilon is None else epsilon}-b_{batch_size}-loss.png", xsize=3, ysize=1.5)
