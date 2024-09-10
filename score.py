import json

scores = []

with open('./merged_results.jsonl', 'r') as fr:
    for data in fr.readlines():
        scores.append(json.loads(data))

# import pdb; pdb.set_trace()
total_score = {'GSM8k': 0, 
               'HE': 0, 
               'MMLU': 0,
               'LongBench': 0,
               'LooGLE': 0
               }
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
    if 'LooGLE' in score['file']:
        # import pdb; pdb.set_trace()
        for sco in score['score']['score']:
            try:
                total_score['LooGLE'] += sco['score']['get_bertscore']
            except:
                pass
            
print('GSM8k: ', total_score['GSM8k'] / 3 * 100)
print('HumanEval: ', total_score['HE'] / 3 * 100)
print('MMLU: ', total_score['MMLU'] / 3 * 100)
print('LongBench: ', total_score['LongBench'] / 21)
print('LooGLE: ', total_score['LooGLE'] / 3 * 100)