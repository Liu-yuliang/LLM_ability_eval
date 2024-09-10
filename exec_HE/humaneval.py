
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)
from tqdm import tqdm
import itertools
import typing
from typing import Iterable, Dict
import gzip
import json
import os
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
HUMAN_EVAL = os.path.join(ROOT, "../datasets/Code/Humaneval",  "HumanEval.jsonl")


def read_problems(evalset_file: str = HUMAN_EVAL) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))
BatchGenerator = typing.Callable[
    [PreTrainedModel, PreTrainedTokenizer, str, int], list[str]
]



def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]


def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")


def split_batch(samples: list[str], size=4):
    mini_batches = []

    for i in range(0, len(samples), size):
        mini_batches.append(samples[i : i + size])

    return mini_batches


def run_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_samples_per_task: int,
    out_path: str,
    generate_batch_completion: BatchGenerator,
    format_tabs: bool = False,
):
    problems = read_problems()
    # problems = dict(itertools.islice(problems.items(), 20))
    samples = []
    pbar = tqdm(total=len(problems) * num_samples_per_task)

    for task_id in problems:
        if format_tabs:
            prompt = problems[task_id]["prompt"].replace("    ", "\t")
        else:
            prompt = problems[task_id]["prompt"]

        batch_completions = generate_batch_completion(
            model, tokenizer, prompt, num_samples_per_task
        )

        for sample in batch_completions:
            result = dict(
                task_id=task_id,
                completion=sample,
            )

            samples += [result]

        pbar.update(num_samples_per_task)

    write_jsonl(out_path, samples)




TOKEN = ""


@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt, batch_size
) -> list[str]:
    input_batch = [prompt for _ in range(batch_size)]
    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # model has no pad token
    )

    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True,
    )

    return [filter_code(fix_indents(completion)) for completion in batch_completions]


if __name__ == "__main__":
    # adjust for n = 10 etc
    num_samples_per_task = 10
    out_path = "./tmp/humaneval.jsonl"
    #os.makedirs("results/openllama", exist_ok=True)
    kwargs = {"torch_dtype": torch.float16, "offload_folder": f"huggyllama/llama-7b/offload"}
    num_gpus = 4
    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"20GiB" for i in range(num_gpus)} })
    tokenizer = AutoTokenizer.from_pretrained('/data1/lyl/LLMs/llama_2_7B')
    model = AutoModelForCausalLM.from_pretrained('/data1/lyl/LLMs/llama_2_7B',low_cpu_mem_usage=True, **kwargs)
    model_base = AutoModelForCausalLM.from_pretrained('/data1/lyl/LLMs/Llama-2-7b-chat-hf',low_cpu_mem_usage=True, **kwargs)
    '''for i in range(32):
                if i>1 and i<11:
                    model.model.layers[i].self_attn.q_proj = copy.deepcopy(model_base.model.layers[i].self_attn.q_proj)
                    model.model.layers[i].self_attn.k_proj = copy.deepcopy(model_base.model.layers[i].self_attn.k_proj)
                    model.model.layers[i].self_attn.v_proj = copy.deepcopy(model_base.model.layers[i].self_attn.v_proj)
                    model.model.layers[i].self_attn.o_proj = copy.deepcopy(model_base.model.layers[i].self_attn.o_proj)
                    model.model.layers[i].self_attn.rotary_emb = copy.deepcopy(model_base.model.layers[i].self_attn.rotary_emb)
                    model.model.layers[i].mlp.gate_proj = copy.deepcopy(model_base.model.layers[i].mlp.gate_proj)
                    model.model.layers[i].mlp.up_proj = copy.deepcopy(model_base.model.layers[i].mlp.up_proj)
                    model.model.layers[i].mlp.down_proj = copy.deepcopy(model_base.model.layers[i].mlp.down_proj)
                if i==31:
                    model.model.layers[i].self_attn.q_proj = copy.deepcopy(model_base.model.layers[i].self_attn.q_proj)
                    model.model.layers[i].self_attn.k_proj = copy.deepcopy(model_base.model.layers[i].self_attn.k_proj)
                    model.model.layers[i].self_attn.v_proj = copy.deepcopy(model_base.model.layers[i].self_attn.v_proj)
                    model.model.layers[i].self_attn.o_proj = copy.deepcopy(model_base.model.layers[i].self_attn.o_proj)
                    model.model.layers[i].self_attn.rotary_emb = copy.deepcopy(model_base.model.layers[i].self_attn.rotary_emb)
                    model.model.layers[i].mlp.gate_proj = copy.deepcopy(model_base.model.layers[i].mlp.gate_proj)
                    model.model.layers[i].mlp.up_proj = copy.deepcopy(model_base.model.layers[i].mlp.up_proj)
                    model.model.layers[i].mlp.down_proj = copy.deepcopy(model_base.model.layers[i].mlp.down_proj)
            
        
        model.model.embed_tokens = copy.deepcopy(model_base.model.embed_tokens)
        model.lm_head= copy.deepcopy(model_base.lm_head)'''
    run_eval(
        model,
        tokenizer,
        num_samples_per_task,
        out_path,
        generate_batch_completion,
        True,
    )
