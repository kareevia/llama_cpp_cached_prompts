#!/usr/bin/env python3
from __future__ import annotations
from typing import * # type: ignore

# change the directory to the script location
import os, sys
os.chdir(sys.path[0])

import argparse

import llama_cpp as LlC
import lib.llama_cpp_cached_preludes as LCP
import lib.repositories_manager as ReM
import lib.repl_mode as REP

DEFAULT_MODEL = "models/saiga2-13b-ggml-model-q4_1.bin"
DEFAULT_NUMBER_OF_OUTPUTS = 5

PRELUDE_DRAFTS = "prelude_drafts"
PRELUDE_DRAFTS_DIR = "prelude_drafts"
PRELUDE_DRAFTS_OUTPUT_DIR = "prelude_drafts_outputs"

PRELUDES = "preludes"
PRELUDES_DIR = "preludes"

TEST_PROMPTS = "test_prompts"
TEST_PROMPTS_DIR = "test_prompts"
TEST_PROMPTS_OUTPUT_DIR = "test_prompts_outputs"
TEST_PROMPTS_PRELUDES_FROM = PRELUDES

CACHE_DIR = "llm_cache"

if (not os.path.exists(CACHE_DIR)):
    os.mkdir(CACHE_DIR)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--full-help", help="show general help and help for all commands (run it "+
    "without any other arguments)",
    action="store_true")

parser.add_argument("-q", "--quiet", help="don't send info messages to `stderr`",
    action="store_true")

parser.add_argument("--lp-model-path", type=str, default=DEFAULT_MODEL,
    metavar="path", help="Path to a Llama_CPP model binary")

parser.add_argument("--lp-n-ctx", type=int, default = 4096, metavar = "int",
    help="Llama_CPP's `n_ctx`")

parser.add_argument("--n_threads", type=int, default = None, metavar = "int",
    help="Llama_CPP's `n_threads` (default -- auto-detect)")
    

llama_cpp_inference_parameters = [
    ("top_k", int, 40),
    ("top_p", float, 0.95),
    ("temp", float, 0.80),
    ("repeat_penalty", float, 1.1),
    ("frequency_penalty", float, 0.0),
    ("presence_penalty", float, 0.0),
    ("tfs_z", float, 1.0),
    ("mirostat_mode", int, 0),
    ("mirostat_tau", float, 5.0),
    ("mirostat_eta", float, 0.1)
]

for (par, typ, defval) in llama_cpp_inference_parameters:
    parm = par.replace("_", "-")
    
    parser.add_argument(f"--lp-{parm}", type=typ, default=defval, metavar=f"{typ.__name__}",
        help=f"Llama_CPP's `{parm}`")

subparsers = parser.add_subparsers(dest="command", title="command", required=True)



cur = subparsers.add_parser("outputs-for-drafts", help="Generate several randomized outputs for "+
    f"all `.txt` files from \"{PRELUDE_DRAFTS_DIR}\" directory and put them into "+
    f"\"{PRELUDE_DRAFTS_OUTPUT_DIR}\".",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

cur.add_argument("-n", "--number-of-outputs", type=int, default=DEFAULT_NUMBER_OF_OUTPUTS, 
    help="Number of randomized outputs to be generated.")

cur.add_argument("-f", "--force-outputs", action="store_true",
    help="Regenerate an output even if its modify time is not older than of its prelude")



cur = subparsers.add_parser("outputs-test-prompts", help="Generate several randomized outputs or "+
    f" one non-randomized (see `--deterministic`) for all `.txt` files from "+
    f"\"{TEST_PROMPTS_DIR}\" directory treating them as prompts appending corresponding preludes "+
    f"from \"{PRELUDES_DIR}\" and put them into \"{TEST_PROMPTS_DIR}\". The names of input files "+
    f"in \"{TEST_PROMPTS_DIR}\" must be of pattern "+
    f"`<prelude_filename>---<arbitraty_prompt_id>.txt`.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

grp = cur.add_mutually_exclusive_group()

grp.add_argument("-n", "--number-of-outputs", type=int, default=DEFAULT_NUMBER_OF_OUTPUTS, 
    help="Number of randomized outputs to be generated.")

grp.add_argument("-d", "--deterministic", action="store_true", 
    help="Generate only one non-random (with `temperature` = 0) output.")

cur.add_argument("-f", "--force-outputs", action="store_true",
    help="Regenerate an output even if its modify time is not older than of its prelude")



cur = subparsers.add_parser("show-stats-on-tokens-number", help="Show stats on tokens number for "+
    "all files in repositories. It could be useful to see what part of the context length each of"+
    " preludes / prompts consume.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)



cur = subparsers.add_parser("precache-preludes", help="Precache all `.txt` prelude files in "+
    f"\"{PRELUDES_DIR}\".",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)



cur = subparsers.add_parser("stdio-json-repl", help="Continuously serve stdio json-formatted "+
    "commands. Command format: {command: \"<command>\", arguments: <arguments>}. Response format: "+
    "{status: \"<`Ok`|`SystemError`>\", value: <value>, message: \"<human readable message>\"}.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)



if len(sys.argv) == 2  and  sys.argv[1] == "--full-help":
    print(parser.format_help())

    subparsers_actions: List[argparse._SubParsersAction[Any]] = [ # type: ignore 
        action 
        for action in parser._actions 
        if isinstance(action, argparse._SubParsersAction) # type: ignore 
    ]

    for subparsers_action in subparsers_actions:

        for choice, subparser in subparsers_action.choices.items():
            print("\n\n# For command `{}`\n".format(choice))
            print(subparser.format_help())

    sys.exit(0)

args = parser.parse_args()

inference_parameters: dict[Any, Any] = {
    par: args.__dict__[f"lp_{par}"]
    for (par, _, _) in llama_cpp_inference_parameters
}

llama = LlC.Llama(
    model_path=args.lp_model_path,
    n_ctx=args.lp_n_ctx,
    n_threads = args.n_threads,
    verbose=not args.quiet,
)

lccp = LCP.LlamaCachedPreludes(llama, CACHE_DIR, not args.quiet)
rep_man = ReM.RepositoriesManager(lccp, not args.quiet)

rep_man.add_repository(PRELUDE_DRAFTS, ReM.Repository(PRELUDE_DRAFTS_DIR))
rep_man.add_repository(PRELUDES, ReM.Repository(PRELUDES_DIR))
rep_man.add_repository(TEST_PROMPTS, ReM.Repository(TEST_PROMPTS_DIR, TEST_PROMPTS_PRELUDES_FROM))

if args.command == "outputs-for-drafts":

    rep_man.generate_outputs_for_preludes(PRELUDE_DRAFTS, PRELUDE_DRAFTS_OUTPUT_DIR, 
        range(args.number_of_outputs), inference_parameters, args.force_outputs)


elif args.command == "outputs-test-prompts":
    
    if args.deterministic:
        outputs_index = ["deterministic"]
        inference_parameters["temp"] = 0.
    else:
        outputs_index = range(args.number_of_outputs)

    rep_man.generate_outputs_as_from_prompts(TEST_PROMPTS, TEST_PROMPTS_OUTPUT_DIR, 
        outputs_index, inference_parameters, args.force_outputs)
    
elif args.command == "show-stats-on-tokens-number":
    print(f"Context length: {args.lp_n_ctx}")

    print(f"\n# Repository \"{PRELUDE_DRAFTS}\"\n")
    rep_man.show_stats_on_tokens_number(PRELUDE_DRAFTS, PRELUDE_DRAFTS_OUTPUT_DIR)

    print(f"\n# Repository \"{PRELUDES}\"\n")
    rep_man.show_stats_on_tokens_number(PRELUDES, None)

    print(f"\n# Repository \"{TEST_PROMPTS}\"\n")
    rep_man.show_stats_on_tokens_number(TEST_PROMPTS, TEST_PROMPTS_OUTPUT_DIR)

elif args.command == "precache-preludes":
    rep_man.precache(PRELUDES)

elif args.command == "stdio-json-repl":
    repl = REP.REPLMode(rep_man, lccp)
    repl.loop()
