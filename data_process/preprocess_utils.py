import os
import pdb
import json
import random
from datasets import load_dataset
from constants import *
from data_process.MMLU_preproces import mmlu_preprocess

def read_jsonl_in_dir(folder_path):
    datas = []

    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory.")
        return

    for filename in os.listdir(folder_path):
        if filename.endswith('.jsonl') or filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r+') as fr:
                for data in fr.readlines():
                    new_data = json.loads(data)
                    new_data['file_name'] = file_path
                    datas.append(new_data)
    return datas

def read_jsonl(file_name):
    datas = []
    with open(file_name, 'r+') as fr:
        for data in fr.readlines():
            datas.append(json.loads(data))
    return datas

def read_lst(task):
    current_path = os.getcwd()
    dataset_path = current_path + '/datasets' + TASK_TO_PATH[task]
    if task == 'PiQA':
        dataset_path = dataset_path[:-11] + 'valid-labels.lst'
    with open(dataset_path, 'r+') as fr:
        return fr.readlines()
    

def read_dataset(task, shot=None):
    
    current_path = os.getcwd()
    dataset_path = current_path + '/datasets' + TASK_TO_PATH[task]
    
    if task == 'GSM8k':
        return read_jsonl(dataset_path)
    elif task == 'HumanEval':
        return read_jsonl(dataset_path)
    elif task == 'PiQA':
        return read_jsonl(dataset_path)
    elif task == 'LongBench':
        return read_jsonl_in_dir(dataset_path)
    elif task == 'LooGLE':
        return read_jsonl_in_dir(dataset_path)
    elif task == 'MMLU':
        return mmlu_preprocess(dataset_path, shot)


def read_answer_dataset(task):
    
    if task == 'GSM8k':
        answer_datas = read_dataset(task)
    elif task == 'HumanEval':
        answer_datas = read_dataset(task)
    elif task == 'PiQA':
        answer_datas = read_lst(task)
    elif task == 'NaturalQA':
        pass
    elif task == 'MMLU':
        answer_datas = None
    return answer_datas


def build_dataset(task, shot=None):
    # 用于统一字段名，和复杂的数据处理，前者如LongBench/LooGLE，后者如MMLU
    if task not in ['MMLU']:
        raw_datas = read_dataset(task)
    
    if task == 'GSM8k':
        shot = 8 if not shot else shot
        print('Preprocessing' + task + f'{shot}-shot datas')
        test_samples = build_gsm8k_few_shot_examples(raw_datas, shot)
    elif task == 'HumanEval':
        shot = 0 if not shot else shot
        print('Preprocessing' + task + f'{shot}-shot datas')
        test_samples = build_human_eval_zero_shot_examples(raw_datas, shot=0)
    elif task == 'PiQA':
        shot = 0 if not shot else shot
        print('Preprocessing' + task + f'{shot}-shot datas')
        test_samples = build_PiQA_zero_shot_examples(raw_datas, shot=0)
    elif task == 'LooGLE':
        # pdb.set_trace()
        test_samples = []
        for data in raw_datas:
            
            if 'longdep_qa' in data['file_name']:
                subset = 'longdep_qa'
            elif 'longdep_summarization' in data['file_name']: 
                subset = 'longdep_summarization'
            elif 'shortdep_cloze' in data['file_name']: 
                subset = 'shortdep_cloze'
            elif 'shortdep_qa' in data['file_name']: 
                subset = 'shortdep_qa'
            length = LooGLE_TASK2MAXLEN[subset]
            test_samples.append({
                'prompt': data['input'], 
                'subset': subset,
                'qa_pairs': data['qa_pairs'], 
                'output': data['output'], 
                'title': data['title'], 
                'gen_len': length
            })

    elif task == 'LongBench':
        test_samples = []
        for data in raw_datas:
            prompt_format = LongBench_TASK2PROMPT[data['dataset']]
            length = LongBench_TASK2MAXLEN[data['dataset']]
            test_samples.append({'prompt': prompt_format.format(context=data['context'], input=data['input']),  
                                'language': data['language'], 
                                'subset': data['dataset'], 
                                'answer': data['answers'], 
                                'gen_len': length, 
                                'all_classes': data['all_classes']
                                })
    elif task == 'MMLU':
        test_samples = read_dataset(task, shot=5)
    

    return test_samples

def random_sample_excluding_index(lst, exclude_index, sample_size):

    if sample_size > len(lst) - 1:
        raise ValueError("sample_size is too big")

    lst_excluding_index = lst[:exclude_index] + lst[exclude_index+1:]

    random_sample = random.sample(lst_excluding_index, sample_size)
    return random_sample

# def build_LOOGLE_samples()


def build_PiQA_zero_shot_examples(raw_datas, shot=0):
    test_samples = []
    for data in raw_datas:
        prompt_template = "Which solution do you think are better to the following question:\n"
        prompt_template += data['goal'] + "\nsol1:\n" + data['sol1'] + "\nsol2:\n" + data['sol2']
        prompt_template += "\nYou should only answer sol1 or sol2, your answer is:\n"
        test_samples.append(prompt_template)
    return test_samples

def build_gsm8k_few_shot_examples(raw_datas, shot=8):
    test_samples = []
    for idx, data in enumerate(raw_datas):    
        # pdb.set_trace()
        random_samples = random_sample_excluding_index(raw_datas, idx, shot)
        prompt_template = ""
        for sample in random_samples:
            prompt_template = prompt_template + "Q: " + sample['question'] + "\n" + "A: " + sample['answer'] + "\n\n"

        prompt_template += "Q: " + data['question'] + "\n" + "A: "
    
        test_samples.append(prompt_template)
        
    return test_samples


def build_human_eval_zero_shot_examples(raw_datas, shot=0):
    test_samples = []
    if shot != 0:
        for idx, data in enumerate(raw_datas):    
            random_samples = random_sample_excluding_index(raw_datas, idx, shot)
            prompt_template = ""
            for sample in random_samples:
                prompt_template = prompt_template + "Q: " + sample['question'] + "\n" + "A: " + sample['answer'] + "\n\n"
            prompt_template += "Q: " + data['question'] + "\n" + "A: "
            test_samples.append(prompt_template)
    else:
        for idx, data in enumerate(raw_datas):  
            prompt_template = ""
            prompt_template += data['prompt']
            # pdb.set_trace()
            test_samples.append({
                'idx': data['task_id'], 
                'prompt': prompt_template})
            
    return test_samples