import os
import pdb
import tqdm
import time
import json
import argparse
import numpy as np
from collections import defaultdict
import re
import torch
from exec_HE.human_eval_evaluate_functional_correctness import execute_HE
from constants import *
from data_process.preprocess_utils import build_dataset, read_answer_dataset, read_jsonl
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
import nltk
from utils.LooGLE.automatic_metrics import (
    get_bleu_score,
    get_rouge_score,
    get_meteor_score,
    get_bertscore,
    get_exact_match,
    get_partial_match
)

nltk.data.path.append('./nltk_data')

class LLMBasicAbilityTester:
    """
    This class is used to test the LLM basic ability.
    """
    def __init__(self,
                 all_datasets_dir="./datasets",
                 model_path=None,
                 context_lengths_max=32000,
                 test_set='all',
                 eval_times=None,
                 vllm=None,
                 method=None,
                 prompt_len=None,
                 run_label=None, 
                 tag=None
                 ):


        if not model_path:
            raise ValueError("Model path must be provided.")

        if not context_lengths_max:
            raise ValueError("Max context lengths must be provided.")


        self.model_path = model_path
        self.all_datasets_dir = all_datasets_dir
        self.method = method
        self.vllm = vllm
        self.run_label = run_label
        self.prompt_len = prompt_len

        self.enc = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        # self.enc.eos_token_id= self.enc.encode("\n")
        print("loading from %s" % self.model_path)

        if self.vllm == 'True':
            self.model_to_test = LLM(model=self.model_path, tensor_parallel_size=1, trust_remote_code=True, max_model_len=context_lengths_max, enforce_eager=True,
                                    gpu_memory_utilization=0.8)

        else:
            self.model_to_test = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto",
                use_cache=True,
                attn_implementation='eager'
            )

            # pdb.set_trace()
            if self.method is not None:
                method = self.method 
                from pyramidkv.monkeypatch import replace_llama
                replace_llama(method)


                max_capacity_prompts = self.prompt_len

                # default to True
                if method == "DynamicKV":
                    output_attentions = True
                else:
                    output_attentions=False

                if max_capacity_prompts != -1:
                    max_capacity_prompts = max_capacity_prompts
                elif max_capacity_prompts_ratio != -1:
                    max_capacity_prompts = round(batch_input_ids.shape[1] * max_capacity_prompts_ratio)


                if method.lower() == "pyramidkv":
                    window_sizes = 8
                elif method.lower() in ["snapkv","streamingllm","h2o"]:
                    window_sizes = 32

                kernel_sizes = 7
                pooling = "maxpool"

                layers = len(self.model_to_test.model.layers)
                # check if window_sizes is a list
                if not isinstance(window_sizes, list):
                    window_sizes = [window_sizes] * layers
                if not isinstance(max_capacity_prompts, list):
                    max_capacity_prompts = [max_capacity_prompts] * layers
                if not isinstance(kernel_sizes, list):
                    kernel_sizes = [kernel_sizes] * layers
                for i in range(layers):
                    self.model_to_test.model.layers[i].self_attn.config.window_size = window_sizes[i]
                    self.model_to_test.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
                    self.model_to_test.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                    self.model_to_test.model.layers[i].self_attn.config.pooling = pooling


        self.model_name = model_path.split('/')[-1] + self.run_label
        self.tag = tag
        self.test_set = test_set
        self.eval_times = eval_times
        self.context_lengths_max = context_lengths_max


    def print_start_test_summary(self):
        print("\n")
        print("Starting LLM Basic Ability Testing...")
        print(f"- Model: {self.model_name}")
        print(f"- Test Set: {self.test_set}")
        print("\n\n")


    def check_result_exists(self, task, time_count):
        if os.path.exists('./results/' + self.model_name + '_' + self.tag + '/' + str(time_count) + '/' + task + '.jsonl'):
            # print('\n')
            # print(f'Evaluate results exits, skip prediction')
            # print(f"- Skipped Model: {self.model_name}")
            # print(f"- Skipped Task: {task}")
            # print(f"- Skipped Times: {time_count}")
            # print("\n\n")
            return True
        # return False

    # def check_score_exists(self, task, time_count):
    #     if os.path.exists('./results/' + self.model_name + '_' + self.tag + '/' + str(time_count) + '/' + task + 'test_score.jsonl'):
    #         return True
        # return False



    def task_start_log(self, task, time_count):
        if not self.check_result_exists(task, time_count): # and not self.check_score_exists(task, time_count):
        # if self.check_result_exists(task, time_count):
            print("\n")
            print(f"Starting Times: {time_count}")
            print(f"- Task: {task}")
            print(f"- Model: {self.model_name}")
            print("\n\n")
            return True
        else:
            print('\n')
            print(f'Evaluate score exits, skip prediction')
            print(f"- Skipped Model: {self.model_name}")
            print(f"- Skipped Task: {task}")
            print(f"- Skipped Times: {time_count}")
            print("\n\n")
            return False


    def run_test(self, time_count, total_times):

        # pred first then eval, for each task

        task_list = ['GSM8k', 'HumanEval', 'MMLU', 'LongBench', 'LooGLE']
        # task_list = ['GSM8k', 'HumanEval', 'MMLU', 'LongBench']
        # task_list = ['GSM8k', 'HumanEval', 'MMLU']
        tasks = self.test_set if self.test_set != 'all' else task_list

        if type(tasks) != list:
            if tasks not in task_list:
                raise ValueError(f"{tasks} must in {task_list}.")
            task = tasks
            self.max_tokens = TASK_TO_MAX_NEW_TOKENS[task]
            if self.task_start_log(tasks, time_count):
                pred_file = './results/' + self.model_name + '_' + self.tag + '/' + str(time_count) + '/' + task + '.jsonl'
                if not self.check_result_exists(tasks, time_count):
                    results = self.pred(task, pred_file, time_count)
                else:results = 0
                score = self.eval(task, results, pred_file)
                self.log_result(pred_file, score)
        else:
            for task in tasks:
                if task in ['LooGLE'] and time_count!=total_times-1:continue
                if task in ['LongBench'] and time_count!=total_times-1:continue
                self.max_tokens = TASK_TO_MAX_NEW_TOKENS[task]
                if not self.task_start_log(task, time_count):
                    continue
                pred_file = './results/' + self.model_name + '_' + self.tag + '/' + str(time_count) + '/' + task + '.jsonl'
                if not self.check_result_exists(task, time_count):
                    results = self.pred(task, pred_file, time_count)
                else: results = 0
                score = self.eval(task, results, pred_file)
                self.log_result(pred_file, score)
        if time_count == total_times-1:
            self.merge_result('./results/' + self.model_name + '_' + self.tag)

    # TODO 分数计算
    def statistica_score(self, path):
        total_score = {'GSM8k': 0, 
               'HE': 0, 
               'MMLU': 0,
               'LongBench': 0, 
               'LooGLE': 0
               }
        scores = []
        with open(path + '/merged_results.jsonl', 'r') as fr:
            for data in fr.readlines():
                scores.append(json.loads(data))
        for score in scores:
            if 'MMLU' in score['file']:
                total_score['MMLU'] += score['score']['score']
            if 'HumanEval' in score['file']:
                total_score['HE'] += score['score']['score']['pass@1']
            if 'GSM8k' in score['file']:
                # import pdb; pdb.set_trace()
                total_score['GSM8k'] += score['score']['score']
            if 'LongBench' in score['file']:
                # import pdb; pdb.set_trace()
                for sco in score['score']['score']:
                    total_score['LongBench'] += sco['score']

        with open(path + '/merged_results_score.jsonl', 'a+') as fw:
            for score in total_score:
                fw.write(json.dumps(score))
                fw.write('\n')

    def merge_result(self, path):

        def find_files_with_name(directory, file_name):
            result_files = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file_name in file:
                        result_files.append(os.path.join(root, file))
            return result_files

        result_files = find_files_with_name(path, 'test_score.jsonl')
        with open(path + '/merged_results.jsonl', 'a+') as fw:
            for file in result_files:
                with open(file, 'r+') as fr:
                    score = json.loads(fr.readlines()[0])

                fw.write(json.dumps({
                    'file': file,
                    'score': score
                }))
                fw.write('\n')

    def start_test(self, args):
        self.print_start_test_summary()
        # TODO
        for time_count in range(0, self.eval_times):
            # if time_count <= /
            self.run_test(time_count, total_times=args.eval_times)

    def remove_content_after_first_triple_newline(self, s):
        index = s.find('\n\n\n')
        if index != -1:
            s = s[:index]
        return s

    def pred(self, task, pred_file, time_count):

        dir_path = os.path.dirname(pred_file)
        os.makedirs(dir_path, exist_ok=True)

        test_samples = build_dataset(task)
        results = []
        print(f'Testing {task}! times: {time_count}')

        with open(pred_file, 'at', encoding="utf-8", buffering=1) as fout:
            if task not in ['LooGLE', 'LongBench']:
                sampling_params = SamplingParams(temperature=0.5, max_tokens=self.max_tokens, seed=time_count)
            # pdb.set_trace()
            if task not in ['MMLU', 'HumanEval', 'LongBench', 'LooGLE']:
                for idx in tqdm.tqdm(range(len(test_samples))):
                    if self.vllm == 'True':
                        # import pdb; pdb.set_trace()
                        response = self.model_to_test.generate([test_samples[idx]], sampling_params, use_tqdm=False)[0].outputs[0].text
                    else:
                        prompt = self.enc(test_samples[idx], return_tensors="pt")
                        input_ids = prompt['input_ids'].to(self.model_to_test.device)
                        print(input_ids.shape)

                        output_ids = self.model_to_test.generate(input_ids, max_new_tokens=self.max_tokens)
                        response = self.enc.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

                    result = {
                        'index': idx,
                        'test_sample': test_samples[idx],
                        'response': response
                        }
                    results.append(result)
                    fout.write(json.dumps(result) + '\n')
            elif task == 'MMLU':
                for idx in tqdm.tqdm(range(len(test_samples))):
                    if self.vllm == 'True':
                        response = self.model_to_test.generate([test_samples[idx]['prompt']], sampling_params, use_tqdm=False)[0].outputs[0].text
                    else:
                        prompt = self.enc(test_samples[idx]['prompt'], return_tensors="pt")
                        input_ids = prompt['input_ids'].to(self.model_to_test.device)

                        output_ids = self.model_to_test.generate(input_ids, max_new_tokens=self.max_tokens)
                        response = self.enc.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


                    result = {
                        'index': idx,
                        'test_sample': test_samples[idx]['prompt'],
                        'response': response,
                        'subject': test_samples[idx]['subject'],
                        'answer': test_samples[idx]['answer']
                        }
                    results.append(result)
                    fout.write(json.dumps(result) + '\n')

            elif task == 'HumanEval':
                for idx in tqdm.tqdm(range(len(test_samples))):
                    if self.vllm == 'True':
                        response = self.model_to_test.generate([test_samples[idx]['prompt']], sampling_params, use_tqdm=False)[0].outputs[0].text
                    else:
                        prompt = self.enc(test_samples[idx]['prompt'], return_tensors="pt")
                        input_ids = prompt['input_ids'].to(self.model_to_test.device)

                        output_ids = self.model_to_test.generate(input_ids, max_new_tokens=self.max_tokens)
                        response = self.enc.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)


                    result = {
                        'task_id': test_samples[idx]['idx'],
                        'test_sample': test_samples[idx]['prompt'],
                        'response': self.remove_content_after_first_triple_newline(response),
                        }
                    results.append(result)
                    fout.write(json.dumps(result) + '\n')


            elif task == 'LongBench':
                for idx in tqdm.tqdm(range(len(test_samples))):

                    sampling_params = SamplingParams(temperature=0.5, max_tokens=test_samples[idx]['gen_len'], seed=time_count)


                    if self.vllm == 'True':
                        response = self.model_to_test.generate([test_samples[idx]['prompt']], sampling_params, use_tqdm=False)[0].outputs[0].text
                    else:
                        prompt = self.enc(test_samples[idx]['prompt'], return_tensors="pt")
                        input_ids = prompt['input_ids'].to(self.model_to_test.device)

                        output_ids = self.model_to_test.generate(input_ids, output_attentions=False, max_new_tokens=self.max_tokens, num_beams=1,do_sample=False,temperature=1.0,eos_token_id=[self.enc.eos_token_id, self.enc.encode("\n", add_special_tokens=False)[-1]])
                        response = self.enc.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


                    result = {
                        'answer': test_samples[idx]['answer'],
                        'test_sample': test_samples[idx]['prompt'],
                        'response': response,
                        'language': test_samples[idx]['language'],
                        'subset': test_samples[idx]['subset']
                        }
                    results.append(result)
                    fout.write(json.dumps(result) + '\n')


            elif task == 'LooGLE':
                max_length = 32000
                for idx in tqdm.tqdm(range(len(test_samples))):
                    sampling_params = SamplingParams(temperature=0.5, max_tokens=test_samples[idx]['gen_len'], seed=time_count)
                    prompt_format = LooGLE_TASK2PROMPT[test_samples[idx]['subset']]
                    if self.vllm == 'True':
                        ans, groundtruth = [], []
                        result = {}
                        raw_inputs = test_samples[idx]['prompt']
                        # pdb.set_trace()
                        if test_samples[idx]['qa_pairs'] == 'none':
                            result['qa_pairs'] = test_samples[idx]['qa_pairs']
                            json_obj = {'input': raw_inputs}

                            prompt = prompt_format.format(**json_obj)
                            tokenized_prompt = self.enc(prompt, truncation=False, return_tensors="pt").input_ids[0]
                            if len(tokenized_prompt) > max_length:
                                half = int(max_length/2)
                                prompt = self.enc.decode(tokenized_prompt[:half], skip_special_tokens=True)+self.enc.decode(tokenized_prompt[-half:], skip_special_tokens=True)


                            pred = self.model_to_test.generate([prompt], sampling_params, use_tqdm=False)[0].outputs[0].text

                            ans.append(pred)
                            groundtruth.append(test_samples[idx]["output"])

                        else:
                            result['qa_pairs'] = eval(test_samples[idx]['qa_pairs'])
                            for j in eval(test_samples[idx]['qa_pairs']):

                                json_obj = {'Q':j['Q'], 'input': raw_inputs}
                                # pdb.set_trace()
                                prompt = prompt_format.format(**json_obj)
                                tokenized_prompt = self.enc(prompt, truncation=False, return_tensors="pt").input_ids[0]
                                if len(tokenized_prompt) > max_length:
                                    half = int(max_length/2)
                                    prompt = self.enc.decode(tokenized_prompt[:half], skip_special_tokens=True)+self.enc.decode(tokenized_prompt[-half:], skip_special_tokens=True)

                                pred = self.model_to_test.generate([prompt], sampling_params, use_tqdm=False)[0].outputs[0].text

                                ans.append(pred)
                                groundtruth.append(j['A'])

                        result['response'] = ans
                        result['ground_truth'] = groundtruth
                        result['subset'] = test_samples[idx]['subset']
                        results.append(result)
                        fout.write(json.dumps(result) + '\n')
                # pdb.set_trace()
        return results

    def split_by_subset(self, results):
        grouped_dicts = defaultdict(list)

        # 分组
        for d in results:
            grouped_dicts[d['subset']].append(d)

        # 转换成列表列表
        result = list(grouped_dicts.values())

        return result

    def eval(self, task, results, pred_file):

        if task == 'GSM8k':
            raw_datasets = read_answer_dataset(task)
            results = read_jsonl(pred_file)
            if len(raw_datasets) != len(results):
                raise ValueError(f"{task} lack of test samples")
            true_count = 0
            for answer, result in zip(raw_datasets, results):
                pattern = r'#### (\d+)'
                try:
                    # pdb.set_trace()
                    ans = re.findall(pattern, answer['answer'])[0]
                    res = re.findall(pattern, result['response'])[0]
                    if ans == res:
                        true_count += 1
                except:
                    pass
            return true_count / len(raw_datasets)

        elif task == 'HumanEval':
            score = execute_HE(sample_file=pred_file, k='1')
            return score

        elif task == 'PiQA':
            count = 0
            labels = read_answer_dataset(task)
            results = read_jsonl(pred_file)
            for answer, result in zip(labels, results):
                if (answer[0] == '0' and result['response'][3] == '1') or (answer[0] == '1' and result['response'][3] == '2'):
                    count += 1
            return count / len(labels)

        elif task == 'LongBench':
            res_scores = []
            results = read_jsonl(pred_file)
            # pdb.set_trace()
            list_of_results = self.split_by_subset(results)
            # self.eval_by_subset(task, list_of_results)

            for longbench_subsets in list_of_results:
                total_score = 0.
                dataset = longbench_subsets[0]['subset']
                for longbench_sample in longbench_subsets:
                    score = 0.
                    if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
                        prediction = longbench_sample['response'].lstrip('\n').split('\n')[0]
                    else:
                        prediction = longbench_sample['response']
                    for ground_truth in longbench_sample['answer']:
                        score = max(score, LongBench_DATASET2METRIC[dataset](prediction, ground_truth, all_classes='null')) # longbench_sample['all_classes']
                    total_score += score
                res_scores.append({
                    'subset': dataset,
                    'score': round(100 * total_score / len(longbench_subsets), 2)})
            return res_scores

        elif task == 'LooGLE':
            res_scores = []
            results = read_jsonl(pred_file)
            list_of_results = self.split_by_subset(results)
            for loogle_subsets in tqdm.tqdm(list_of_results):
                # print(len(list_of_results))

                score_result = {}
                dataset = loogle_subsets[0]['subset']
                # print(dataset)
                if dataset == 'shortdep_cloze':
                    eval_functions = ["get_exact_match", "get_partial_match"]
                else:
                    eval_functions = [
                        "get_bleu_score",
                        # "get_rouge_score",
                        "get_meteor_score",
                        "get_bertscore"
                    ]
                for loogle_sample in loogle_subsets:
                    # pdb.set_trace()
                    score_result = self.loogle_evaluation(loogle_sample, score_result, eval_functions, dataset)

                if dataset != "shortdep_cloze":
                    res_scores.append({
                        'subset': dataset,
                        'score': self.loogle_get_semantic_matching(score_result, eval_functions)})
                else:
                    res_scores.append({
                        'subset': dataset,
                        'score': self.loogle_get_match_score(score_result, eval_functions)})
                # print(dataset)

            return res_scores

        elif task == 'MMLU':
            results = read_jsonl(pred_file)
            count = 0
            for result in results:
                if result['response'][1] == result['answer']:
                    count += 1
            return count / len(results)


    def log_result(self, pred_file, score):
        score_file = pred_file[:-6] + '_test_score.jsonl'
        with open(score_file, 'a+') as fw:
            fw.write(json.dumps({'score': score
            }))


    def loogle_evaluation(self, data, scores, functions, task):
        fail_count = 0
        def contains_no_alphabet(s):
            return not any(char.isalpha() for char in s)
        for i in range(len(data["ground_truth"])):
            hyp, ref = data["response"][i], data["ground_truth"][i]
            # print(hyp, ref)
            if hyp == '':#or contains_no_alphabet(hyp):
                hyp = 'None'
            if "qa_pairs" in data:
                if data["qa_pairs"] != "none":
                    question = data["qa_pairs"][i]["Q"]
                else:
                    question = ""

            for j in functions:
                # if j not in scores:
                scores[j] = []
                try:
                    scores[j].append(eval(j)(question, ref, hyp, task))
                except:
                    fail_count+=1
                    print(j)
                    print("heree")
        return scores


    def loogle_get_semantic_matching(self, result, functions):
        final_score = {}
        for i in functions:
            if type(result[i][0]) is tuple:
                l = result[i]
                final_score[i] = [np.mean([i[j] for i in l]) for j in range(len(l[0]))]
            else:
                final_score[i] = np.mean(result[i])
        return final_score


    def loogle_get_match_score(self, result, functions):
        final_score = {}
        # pdb.set_trace()
        for i in functions:
            try:
                match_count = np.sum([j[0] for j in result[i]])
            except:
                pdb.set_trace()
            all_count = np.sum([j[1] for j in result[i]])
            final_score[i] = round(match_count / all_count, 4)
        return final_score


    def print_score(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, help='path to model')
    parser.add_argument('--ability', type=str, default='all', help='abilities, all/GSM8k/HumanEval/PiQA/NaturalQA')
    parser.add_argument('--eval_times', type=int, default=1, help='pred times to get average score')
    parser.add_argument('--vllm', type=str, default='True', help='')
    parser.add_argument('--prompt_len', type=int, default=0, help='')
    parser.add_argument('--method', type=str, default='', help='')
    parser.add_argument('--run_label', type=str, default='', help='')
    parser.add_argument('--tag', type=str, default='', help='')
    args = parser.parse_args()

    # pdb.set_trace()
    ht = LLMBasicAbilityTester(model_path=args.model_path,
                                 test_set=args.ability,
                                 eval_times=args.eval_times,
                                 vllm=args.vllm,
                                 prompt_len=args.prompt_len,
                                 method=args.method,
                                 run_label=args.run_label, 
                                 tag=args.tag
                                 )

    ht.start_test(args)