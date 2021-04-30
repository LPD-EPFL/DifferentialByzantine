# coding: utf-8
###
 # @file   train.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Simulate a training session under attack.
###

import tools
tools.success("Module loading...")

import argparse
import collections
import json
import math
import os
import pathlib
import random
import signal
import sys
import torch
import torchvision
import traceback

import aggregators
import attacks
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
  parser.add_argument("--seed",
    type=int,
    default=-1,
    help="Fixed seed to use for reproducibility purpose, negative for random seed")
  parser.add_argument("--device",
    type=str,
    default="auto",
    help="Device on which to run the experiment, \"auto\" by default")
  parser.add_argument("--device-gar",
    type=str,
    default="same",
    help="Device on which to run the GAR, \"same\" for no change of device")
  parser.add_argument("--nb-steps",
    type=int,
    default=300,
    help="Number of training steps to do, non-positive for no limit")
  parser.add_argument("--nb-workers",
    type=int,
    default=11,
    help="Total number of worker machines")
  parser.add_argument("--nb-decl-byz",
    type=int,
    default=4,
    help="Number of Byzantine worker(s) to support")
  parser.add_argument("--nb-real-byz",
    type=int,
    default=0,
    help="Number of actual Byzantine worker(s)")
  parser.add_argument("--gar",
    type=str,
    default="average",
    help="(Byzantine-resilient) aggregation rule to use")
  parser.add_argument("--gar-args",
    nargs="*",
    help="Additional GAR-dependent arguments to pass to the aggregation rule")
  parser.add_argument("--privacy",
    action="store_true",
    default=False,
    help="Gaussian privacy noise ε constant")
  parser.add_argument("--privacy-epsilon",
    type=float,
    default=0.5,
    help="Gaussian privacy noise ε constant; ignore if '--privacy' is not specified")
  parser.add_argument("--privacy-delta",
    type=float,
    default=0.5,
    help="Gaussian privacy noise δ constant; ignore if '--privacy' is not specified")
  parser.add_argument("--gradient-clip",
    type=float,
    default=5,
    help="Maximum L2-norm, above which clipping occurs, for the estimated gradients")
  parser.add_argument("--attack",
    type=str,
    default="nan",
    help="Attack to use")
  parser.add_argument("--attack-args",
    nargs="*",
    help="Additional attack-dependent arguments to pass to the attack")
  parser.add_argument("--model",
    type=str,
    default="simples-conv",
    help="Model to train")
  parser.add_argument("--model-args",
    nargs="*",
    help="Additional model-dependent arguments to pass to the model")
  parser.add_argument("--loss",
    type=str,
    default="nll",
    help="Loss to use")
  parser.add_argument("--loss-args",
    nargs="*",
    help="Additional loss-dependent arguments to pass to the loss")
  parser.add_argument("--criterion",
    type=str,
    default="top-k",
    help="Criterion to use")
  parser.add_argument("--criterion-args",
    nargs="*",
    help="Additional criterion-dependent arguments to pass to the criterion")
  parser.add_argument("--dataset",
    type=str,
    default="mnist",
    help="Dataset to use")
  parser.add_argument("--dataset-args",
    nargs="*",
    help="Additional dataset-dependent arguments to pass to the dataset")
  parser.add_argument("--batch-size",
    type=int,
    default=25,
    help="Batch-size to use for training, 0 for maximum")
  parser.add_argument("--batch-size-test",
    type=int,
    default=100,
    help="Batch-size to use for testing, 0 for maximum")
  parser.add_argument("--test-repeat",
    type=int,
    default=100,
    help="How many evaluation(s) with the test batch-size to average for one evaluation")
  parser.add_argument("--no-transform",
    action="store_true",
    default=False,
    help="Whether to disable any dataset tranformation (e.g. random flips)")
  parser.add_argument("--learning-rate",
    type=float,
    default=0.01,
    help="Learning rate to use for training")
  parser.add_argument("--learning-rate-decay",
    type=int,
    default=5000,
    help="Learning rate hyperbolic half-decay time, non-positive for no decay")
  parser.add_argument("--learning-rate-decay-delta",
    type=int,
    default=5000,
    help="How many steps between two learning rate updates, must be a positive integer")
  parser.add_argument("--momentum",
    type=float,
    default=0.9,
    help="Momentum to use for training")
  parser.add_argument("--dampening",
    type=float,
    default=0.,
    help="Dampening to use for training")
  parser.add_argument("--weight-decay",
    type=float,
    default=0,
    help="Weight decay to use for training")
  parser.add_argument("--l1-regularize",
    type=float,
    default=None,
    help="Add L1 regularization of the given factor to the loss")
  parser.add_argument("--l2-regularize",
    type=float,
    default=None,
    help="Add L2 regularization of the given factor to the loss")
  parser.add_argument("--result-directory",
    type=str,
    default=None,
    help="Path of the directory in which to save the experiment results (loss, cross-accuracy, ...) and checkpoints, empty for no saving")
  parser.add_argument("--evaluation-delta",
    type=int,
    default=100,
    help="How many training steps between model evaluations, 0 for no evaluation")
  parser.add_argument("--user-input-delta",
    type=int,
    default=0,
    help="How many training steps between two prompts for user command inputs, 0 for no user input")
  # Parse command line
  return parser.parse_args(sys.argv[1:])

with tools.Context("cmdline", "info"):
  args = process_commandline()
  # Parse additional arguments
  for name in ("gar", "attack", "model", "dataset", "loss", "criterion"):
    name = f"{name}_args"
    keyval = getattr(args, name)
    setattr(args, name, dict() if keyval is None else tools.parse_keyval(keyval))
  # Count the number of real honest workers
  args.nb_honests = args.nb_workers - args.nb_real_byz
  if args.nb_honests < 0:
    tools.fatal(f"Invalid arguments: there are more real Byzantine workers ({args.nb_real_byz}) than total workers ({args.nb_workers})")
  # Check general training parameters
  if args.momentum < 0.:
    tools.fatal(f"Invalid arguments: negative momentum factor {args.momentum}")
  if args.dampening < 0.:
    tools.fatal(f"Invalid arguments: negative dampening factor {args.dampening}")
  if args.weight_decay < 0.:
    tools.fatal(f"Invalid arguments: negative weight decay factor {args.weight_decay}")
  # Check the learning rate and associated options
  if args.learning_rate <= 0:
    tools.fatal(f"Invalid arguments: non-positive learning rate {args.learning_rate}")
  if args.learning_rate_decay < 0:
    tools.fatal(f"Invalid arguments: negative learning rate decay {args.learning_rate_decay}")
  if args.learning_rate_decay_delta <= 0:
    tools.fatal(f"Invalid arguments: non-positive learning rate decay delta {args.learning_rate_decay_delta}")
  # Check the privacy-related metrics
  if args.gradient_clip <= 0.:
    tools.fatal(f"Invalid arguments: non-positive gradient clip constant {args.gradient_clip}")
  if args.privacy:
    if args.privacy_epsilon <= 0. or args.privacy_epsilon >= 1.:
      tools.fatal(f"Invalid arguments: off-bounds (]0, 1[) ε constant {args.privacy_epsilon}")
    if args.privacy_delta <= 0. or args.privacy_delta >= 1.:
      tools.fatal(f"Invalid arguments: off-bounds (]0, 1[) δ constant {args.privacy_delta}")
    args.privacy_sensitivity = 2 * args.gradient_clip / args.batch_size
  # Print configuration
  def cmd_make_tree(subtree, level=0):
    if isinstance(subtree, tuple) and len(subtree) > 0 and isinstance(subtree[0], tuple) and len(subtree[0]) == 2:
      label_len = max(len(label) for label, _ in subtree)
      iterator  = subtree
    elif isinstance(subtree, dict):
      if len(subtree) == 0:
        return " - <none>"
      label_len = max(len(label) for label in subtree.keys())
      iterator  = subtree.items()
    else:
      return f" - {subtree}"
    level_spc = "  " * level
    res = ""
    for label, node in iterator:
      res += f"{os.linesep}{level_spc}· {label}{' ' * (label_len - len(label))}{cmd_make_tree(node, level + 1)}"
    return res
  cmdline_config = "Configuration" + cmd_make_tree((
    ("Reproducibility", "not enforced" if args.seed < 0 else f"enforced (seed {args.seed})"),
    ("#workers", args.nb_workers),
    ("#declared Byz.", args.nb_decl_byz),
    ("#actually Byz.", args.nb_real_byz),
    ("Model", (
      ("Name", args.model),
      ("Arguments", args.model_args))),
    ("Dataset", (
      ("Name", args.dataset),
      ("Arguments", args.dataset_args),
      ("Batch size", (
        ("Training", args.batch_size or 'max'),
        ("Testing", f"{args.batch_size_test or 'max'} × {args.test_repeat}"))),
      ("Transforms", "none" if args.no_transform else "default"))),
    ("Loss", (
      ("Name", args.loss),
      ("Arguments", args.loss_args),
      ("Regularization", (
        ("l1", "none" if args.l1_regularize is None else args.l1_regularize),
        ("l2", "none" if args.l2_regularize is None else args.l2_regularize))))),
    ("Criterion", (
      ("Name", args.criterion),
      ("Arguments", args.criterion_args))),
    ("Optimizer", (
      ("Name", "sgd"),
      ("Learning rate", (
        ("Initial", args.learning_rate),
        ("Half-decay", args.learning_rate_decay if args.learning_rate_decay > 0 else "none"),
        ("Update delta", args.learning_rate_decay_delta if args.learning_rate_decay > 0 else "n/a"))),
      ("Momentum", args.momentum),
      ("Dampening", args.dampening),
      ("Weight decay", args.weight_decay))),
    ("Attack", (
      ("Name", args.attack),
      ("Arguments", args.attack_args))),
    ("Aggregation", (
      ("Name", args.gar),
      ("Arguments", args.gar_args))),
    ("Differential privacy", (
      ("Enabled?", "yes" if args.privacy else "no"),
      ("ε constant", args.privacy_epsilon if args.privacy else "n/a"),
      ("δ constant", args.privacy_delta if args.privacy else "n/a"),
      ("l2-sensitivity", args.privacy_sensitivity if args.privacy else "n/a")))))
  print(cmdline_config)

# ---------------------------------------------------------------------------- #
# Setup
tools.success("Experiment setup...")

def result_make(name, *fields):
  """ Make and bind a new result file with a name, initialize with a header line.
  Args:
    name      Name of the result file
    fields... Name of each field, in order
  Raises:
    'KeyError' if name is already bound
    'RuntimeError' if no name can be bound
    Any exception that 'io.FileIO' can raise while opening/writing/flushing
  """
  # Check if results are to be output
  global args
  if args.result_directory is None:
    raise RuntimeError("No result is to be output")
  # Check if name is already bounds
  global result_fds
  if name in result_fds:
    raise KeyError(f"Name {name!r} is already bound to a result file")
  # Make the new file
  fd = (args.result_directory / name).open("w")
  fd.write("# " + ("\t").join(str(field) for field in fields))
  fd.flush()
  result_fds[name] = fd

def result_get(name):
  """ Get a valid descriptor to the bound result file, or 'None' if the given name is not bound.
  Args:
    name Given name
  Returns:
    Valid file descriptor, or 'None'
  """
  # Check if results are to be output
  global args
  if args.result_directory is None:
    return None
  # Return the bound descriptor, if any
  global result_fds
  return result_fds.get(name, None)

def result_store(fd, *entries):
  """ Store a line in a valid result file.
  Args:
    fd         Descriptor of the valid result file
    entries... Object(s) to convert to string and write in order in a new line
  """
  fd.write(os.linesep + ("\t").join(str(entry) for entry in entries))
  fd.flush()

with tools.Context("setup", "info"):
  # Enforce reproducibility if asked (see https://pytorch.org/docs/stable/notes/randomness.html)
  reproducible = args.seed >= 0
  if reproducible:
    torch.manual_seed(args.seed)
    import numpy
    numpy.random.seed(args.seed)
  torch.backends.cudnn.deterministic = reproducible
  torch.backends.cudnn.benchmark     = not reproducible
  # Configurations
  config = experiments.Configuration(dtype=torch.float32, device=(None if args.device.lower() == "auto" else args.device), noblock=True)
  if args.device_gar.lower() == "same":
    config_gar = config
  else:
    config_gar = experiments.Configuration(dtype=config["dtype"], device=(None if args.device_gar.lower() == "auto" else args.device_gar), noblock=config["non_blocking"])
  # Defense
  defense = aggregators.gars.get(args.gar)
  if defense is None:
    tools.fatal_unavailable(aggregators.gars, args.gar, what="aggregation rule")
  # Attack
  attack = attacks.attacks.get(args.attack)
  if attack is None:
    tools.fatal_unavailable(attacks.attacks, args.attack, what="attack")
  # Model
  model = experiments.Model(args.model, config, **args.model_args)
  # Datasets
  if args.no_transform:
    train_transforms = test_transforms = torchvision.transforms.ToTensor()
  else:
    train_transforms = test_transforms = None # Let default values
  trainset, testset = experiments.make_datasets(args.dataset, args.batch_size, args.batch_size_test, train_transforms=train_transforms, test_transforms=test_transforms, **args.dataset_args)
  model.default("trainset", trainset)
  model.default("testset", testset)
  # Loss and criterion
  loss = experiments.Loss(args.loss, **args.loss_args)
  if args.l1_regularize is not None:
    loss += args.l1_regularize * experiments.Loss("l1")
  if args.l2_regularize is not None:
    loss += args.l2_regularize * experiments.Loss("l2")
  criterion = experiments.Criterion(args.criterion, **args.criterion_args)
  model.default("loss", loss)
  model.default("criterion", criterion)
  # Optimizer
  optimizer = experiments.Optimizer("sgd", model, lr=args.learning_rate, momentum=args.momentum, dampening=args.dampening, weight_decay=args.weight_decay)
  model.default("optimizer", optimizer)
  # Privacy noise distribution
  if args.privacy:
    param = model.get()
    privacy_factor = args.privacy_sensitivity * math.sqrt(2 * math.log(1.25 / args.privacy_delta)) / args.privacy_epsilon
    grad_noise = torch.distributions.normal.Normal(torch.zeros_like(param), torch.ones_like(param).mul_(privacy_factor))
  # Make the result directory (if requested)
  if args.result_directory is not None:
    try:
      resdir = pathlib.Path(args.result_directory).resolve()
      resdir.mkdir(mode=0o755, parents=True, exist_ok=True)
      args.result_directory = resdir
    except Exception as err:
      tools.warning(f"Unable to create the result directory {str(resdir)!r} ({err}); no result will be stored")
    else:
      result_fds = dict()
      try:
        # Make evaluation file
        if args.evaluation_delta > 0:
          result_make("eval", "Step number", "Cross-accuracy")
        # Make study file
        result_make("study", "Step number", "Training point count",
          "Average loss", "l2 from origin",
          "Honest gradient deviation", "Attack gradient deviation",
          "Honest gradient norm", "Attack gradient norm", "Defense gradient norm",
          "Honest max coordinate", "Attack max coordinate", "Defense max coordinate",
          "Honest-attack cosine", "Honest-defense cosine", "Attack-defense cosine")
        # Store the configuration info and JSON representation
        (args.result_directory / "config").write_text(cmdline_config + os.linesep)
        with (args.result_directory / "config.json").open("w") as fd:
          def convert_to_supported_json_type(x):
            if type(x) in {str, int, float, bool, type(None), dict, list}:
              return x
            elif type(x) is set:
              return list(x)
            else:
              return str(x)
          datargs = dict((name, convert_to_supported_json_type(getattr(args, name))) for name in dir(args) if len(name) > 0 and name[0] != "_")
          del convert_to_supported_json_type
          json.dump(datargs, fd, ensure_ascii=False, indent="\t")
      except Exception as err:
        tools.warning(f"Unable to create some result files in directory {str(resdir)!r} ({err}); some result(s) may be missing")

# ---------------------------------------------------------------------------- #
# Training
tools.success("Training...")

def compute_avg_dev(values):
  """ Compute the arithmetic mean and standard deviation of a list of values.
  Args:
    values Iterable of values
  Returns:
    Arithmetic mean, standard deviation
  """
  avg = sum(values) / len(values)
  var = 0
  for value in values:
    var += (value - avg) ** 2
  var /= len(values) - 1
  return avg, math.sqrt(var)

class StopTrainingLoop(Exception):
  """ Local exception to signal and stop the training loop.
  """
  pass

# Training until limit or stopped
with tools.Context("training", "info"):
  was_training  = False
  current_lr    = args.learning_rate
  steps         = 0
  datapoints    = 0
  fd_eval       = result_get("eval")
  fd_study      = result_get("study")
  atc_gradient  = tools.AccumulatedTimedContext(sync=True)
  atc_noise     = tools.AccumulatedTimedContext(sync=True)
  atc_aggregate = tools.AccumulatedTimedContext(sync=True)
  atc_evaluate  = tools.AccumulatedTimedContext(sync=True)
  params_origin = model.get().clone().detach_()
  try:
    while not exit_is_requested():
      # ------------------------------------------------------------------------ #
      # Evaluate if any milestone is reached
      milestone_evaluation = args.evaluation_delta > 0 and steps % args.evaluation_delta == 0
      milestone_user_input = args.user_input_delta > 0 and steps % args.user_input_delta == 0
      milestone_any        = milestone_evaluation or milestone_user_input
      # Training notification (end)
      if milestone_any and was_training:
        print(" done.")
        was_training = False
      # Evaluation milestone reached
      if milestone_evaluation:
        print("Accuracy (step %d)..." % steps, end="", flush=True)
        with atc_evaluate:
          res = model.eval()
          for _ in range(args.test_repeat - 1):
            res += model.eval()
          acc = res[0].item() / res[1].item()
        print(" %.2f%%." % (acc * 100.))
        # Store the evaluation result
        if fd_eval is not None:
          result_store(fd_eval, steps, acc)
      # User input milestone
      if milestone_user_input:
        tools.interactive()
      # Check if reach step limit
      if args.nb_steps > 0 and steps >= args.nb_steps:
        # Training notification (end)
        if was_training:
          print(" done.")
          was_training = False
        # Leave training loop
        raise StopTrainingLoop()
      # Training notification (begin)
      if milestone_any and not was_training:
        print("Training...", end="", flush=True)
        was_training = True
      # ------------------------------------------------------------------------ #
      # Compute (honest) losses (if it makes sense), gradients and voting data
      grad_honests = list()
      loss_honests = list()
      # For each honest worker
      with atc_gradient:
        for i in range(args.nb_honests):
          grad, loss = model.backprop(outloss=True)
          grad = grad.clone().detach_()
          # Loss append
          loss_honests.append(loss.item())
          # Gradient clip
          if args.gradient_clip is not None:
            grad_norm = grad.norm().item()
            if grad_norm > args.gradient_clip:
              grad.mul_(args.gradient_clip / grad_norm)
          # Gradient append
          grad_honests.append(grad)
      # Pre-compute some quantities for study (here if necessary, since each gradient in 'grad_honests' may have privacy noise added)
      if fd_study is not None:
        # Compute average loss ('len(loss_honests) > 0' is guaranteed)
        loss_avg = sum(loss_honests) / len(loss_honests)
        # Compute the sampled and honest gradients norm average, norm deviation and max absolute coordinate
        honest_grad_avg, honest_norm_avg, honest_norm_dev, honest_norm_max = tools.compute_avg_dev_max(grad_honests)
      # Move the honest gradients to the GAR device
      if config_gar is not config:
        grad_honests_gar = list(grad.to(device=config_gar["device"], non_blocking=config_gar["non_blocking"]) for grad in grad_honests)
      else:
        grad_honests_gar = grad_honests
      # ------------------------------------------------------------------------ #
      # Add privacy noise to the 'grad_honests_gar' (might be 'grad_honests'), which are the ones sent and processed by the server
      if args.privacy:
        with atc_noise:
          for grad in grad_honests_gar:
            grad.add_(grad_noise.sample())
      # ------------------------------------------------------------------------ #
      # Compute the Byzantine gradients
      grad_attacks = attack.checked(grad_honests=grad_honests_gar, f_decl=args.nb_decl_byz, f_real=args.nb_real_byz, model=model, defense=defense, **args.attack_args)
      # ------------------------------------------------------------------------ #
      # Aggregate and update the model
      with atc_aggregate:
        grad_defense = defense.checked(gradients=(grad_honests_gar + grad_attacks), f=args.nb_decl_byz, model=model, **args.gar_args)
      # Move the defense gradient back to the main device
      if config_gar is not config:
        for grad in grad_attacks:
          grad.data = grad.to(device=config["device"], non_blocking=config["non_blocking"])
        grad_defense = grad_defense.to(device=config["device"], non_blocking=config["non_blocking"])
      # Compute l2-distance from origin (if needed for study)
      if fd_study is not None:
        l2_origin = model.get().sub(params_origin).norm().item()
      # Model update (possibly updating the learning rate)
      if args.learning_rate_decay > 0 and steps % args.learning_rate_decay_delta == 0:
        current_lr = args.learning_rate / (steps / args.learning_rate_decay + 1)
        optimizer.set_lr(current_lr)
      model.update(grad_defense)
      # ------------------------------------------------------------------------ #
      # Store study (if requested)
      if fd_study is not None:
        # Compute the sampled and honest gradients norm average, norm deviation and max absolute coordinate
        attack_grad_avg, attack_norm_avg, attack_norm_dev, attack_norm_max = tools.compute_avg_dev_max(grad_attacks)
        # Compute the defense norm average and max absolute coordinate
        defense_grad = grad_defense # (Mere renaming for consistency)
        defense_norm_avg = defense_grad.norm().item()
        defense_norm_max = defense_grad.abs().max().item()
        # Compute cosine of solid angles
        cosin_honatt = math.nan if attack_grad_avg is None else torch.dot(honest_grad_avg, attack_grad_avg).div_(honest_norm_avg).div_(attack_norm_avg).item()
        cosin_hondef = torch.dot(honest_grad_avg, defense_grad).div_(honest_norm_avg).div_(defense_norm_avg).item()
        cosin_attdef = math.nan if attack_grad_avg is None else torch.dot(attack_grad_avg, defense_grad).div_(attack_norm_avg).div_(defense_norm_avg).item()
        # Store the result (float-to-string format chosen so not to lose precision)
        float_format = {torch.float16: "%.4e", torch.float32: "%.8e", torch.float64: "%.16e"}.get(config["dtype"], "%s")
        result_store(fd_study, steps, datapoints,
          float_format % loss_avg, float_format % l2_origin,
          float_format % honest_norm_dev, float_format % attack_norm_dev,
          float_format % honest_norm_avg, float_format % attack_norm_avg, float_format % defense_norm_avg,
          float_format % honest_norm_max, float_format % attack_norm_max, float_format % defense_norm_max,
          float_format % cosin_honatt, float_format % cosin_hondef, float_format % cosin_attdef)
      # ------------------------------------------------------------------------ #
      # Increase the step counter
      steps      += 1
      datapoints += args.batch_size * args.nb_honests
  except StopTrainingLoop:
    pass
  # Training notification (end)
  if was_training:
    print(" interrupted.")

# Print and store timing counters
with tools.Context("perf", "info"):
  perfs = dict()
  perf_params = (
    (atc_gradient,  "grad",  "Gradient computation (per worker)", args.nb_honests),
    (atc_noise,     "noise", "Noise addition (per worker)",       args.nb_honests),
    (atc_aggregate, "aggr",  "Gradient aggregation", 1),
    (atc_evaluate,  "eval",  "Model evaluation",     1))
  # Compute max name length
  nlen = max(len(name) for _, _, name, _ in perf_params)
  # Print
  for atc, key, name, div in perf_params:
    acc = tools.AccumulatedTimedContext(atc.current_runtime() / div)
    print(f"{name:{nlen}s} - {acc}")
    perfs[key] = (acc.current_runtime(), name)
  # Store
  if args.result_directory:
    with (args.result_directory / "perfs.json").open("w") as fd:
      json.dump(perfs, fd, ensure_ascii=False, indent="\t")
