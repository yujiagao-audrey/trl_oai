from thefuzz import fuzz
import json

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def are_strings_similar(string1, string2, threshold=5):
    """
    Check if two strings are very similar to each other.
    
    Args:
    string1 (str): The first string to compare.
    string2 (str): The second string to compare.
    threshold (int): The similarity threshold (0-100). Default is 80.
    
    Returns:
    bool: True if the strings are similar, False otherwise.
    """
    similarity_ratio = fuzz.ratio(string1, string2)
    return similarity_ratio >= threshold

def compare_completions(data1, data2, threshold):
    completions1 = data1.get("completions", [])
    completions2 = data2.get("completions", [])

    if len(completions1) != len(completions2):
        print("The number of completions in the two files does not match.")
        return

    unsimilar_data1 = []
    unsimilar_data2 = []
    results = []
    for i, (comp1, comp2) in enumerate(zip(completions1, completions2)):
        similarity = are_strings_similar(comp1, comp2, threshold)
        if not similarity:
            unsimilar_data1.append(comp1)
            unsimilar_data2.append(comp2)
            results.append(best_idxs[i])
            
            print('-'*30)
            print('prompt: ' + prompts[i])
            print('chosen index: ' + str(best_idxs[i]))
            print('response 0: ' + comp1)
            print('response 1: ' + comp2)
            print('-'*30)

    return unsimilar_data1, unsimilar_data2, results

# File paths for the two JSON files
task = 'helpful' # TODO: task type
model1 = 'ours' # TODO: the model we want to compare
base_model_type = 'Qwen2.5-7B-Instruct' # TODO: the base model to test
in_domain = True # TODO: the domain for test
threshold = 10 # TODO: smaller threshold, smaller data

if in_domain:
    file1_path = f'/evaluate/evaluate_result/{task}/{base_model_type}-{model1}_completions_in_domain.json'
    file2_path = f'/evaluate/evaluate_result/{task}/{base_model_type}_completions_in_domain.json'
    file3_path = f'/evaluate/evaluate_result/{task}/{base_model_type}-{model1}_against_{base_model_type}_on_{task}.json'
    file4_path = f'/evaluate/evaluate_result/{task}/in_domain_prompt.json'
else:
    file1_path = f'/evaluate/evaluate_result/{task}/{base_model_type}-{model1}_completions.json'
    file2_path = f'/evaluate/evaluate_result/{task}/{base_model_type}_completions.json'
    file3_path = f'/evaluate/evaluate_result/{task}/{base_model_type}-{model1}_against_{base_model_type}_on_{task}.json'
    file4_path = f'/evaluate/evaluate_result/{task}/prompt.json'

# Load the JSON files
data1 = load_json_file(file1_path)
data2 = load_json_file(file2_path)
comparison_reuslts = load_json_file(file3_path)
prompts = load_json_file(file4_path)

model1 = data1['model_name']
model2 = data2['model_name']
best_idxs = comparison_reuslts['best_idxs']

# Compare the completions
unsimilar_data1, unsimilar_data2, results = compare_completions(data1, data2, threshold)

# Test the function
print(len(results))
print(sum(results) / len(results))
