import argparse
import sys
import yaml
import itertools

import incl_lib


FINISHED_STATUS = [
    "finished",
    "stopped",
    "interrupted",
    "aborted",
    "crashed"
]


def load_parameters(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def run_experiment(incl_params, hyperparams):
    command = ["python", "codes/main.py"]
    for key, value in hyperparams.items():
        command.append(f"--{key}")
        command.append(str(value))
    
    breakpoint()
    job_id = incl_lib.run(
        script=" ".join(command),
        **incl_params,
    )

    return job_id


def run_grid_search(incl_params, exp_params):
    keys, values = zip(*exp_params.items())
    hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    jobs = {}
    for hyperparams in hyperparam_combinations:
        job_id = run_experiment(incl_params, hyperparams)
        jobs[job_id] = hyperparams

    return jobs


def print_jobs(jobs):
    # Clear previous output
    sys.stdout.write("\033c")

    for job_id, hyperparams in jobs.items():
        status = incl_lib.get_status(job_id)

        sys.stdout.write(f"Job ID: {job_id} / Status: {status} / Hyperparams: {hyperparams}\n")

    # Ensure output is flushed immediately
    sys.stdout.flush()


def wait_for_jobs(jobs):
    while any(incl_lib.get_status(job_id) not in FINISHED_STATUS for job_id in jobs.keys()):
        print_jobs(jobs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_path", type=str, required=True)
    args = parser.parse_args()

    params = load_parameters(args.param_path)
    incl_params = params["incl_params"]
    exp_params = params["exp_params"]

    jobs = run_grid_search(incl_params, exp_params)
    wait_for_jobs(jobs)


if __name__ == "__main__":
    main()
