# import fire
import sys
import pdb

from exec_HE.human_eval_evaluation import evaluate_functional_correctness
from exec_HE.humaneval import HUMAN_EVAL

def execute_HE(
    sample_file,
    k: str = "1,10,100",
    n_workers: int = 1,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    # pdb.set_trace()
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    return results


# def main():
#     fire.Fire(execute_HE)


# sys.exit(main())
