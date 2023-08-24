from torch.utils.data import Dataset
from transformers import AutoTokenizer
from collections import defaultdict

class CodePairsXLCostDataset(Dataset):
    def __init__(self, base_path,
                 split="train",
                 format="snip",
                 model_name = "Salesforce/codet5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


        # Read the Java code samples
        with open(f"{base_path}/{split}-Java-Python-tok.java", 'r') as f:
            self.java_code_lines = f.readlines()

        # Read the Python code samples
        with open(f"{base_path}/{split}-Java-Python-tok.py", 'r') as f:
            self.python_code_lines = f.readlines()

        # Read Java mapping
        java_problem_to_lines = defaultdict(list)

        if format == "snip":
            with open(f"{base_path}/{split}-Java-map.jsonl", 'r') as f:
                for line in f.readlines():
                    problem_id, _, idx = line.strip().split('-')
                    idx = int(idx) - 1  # convert to zero-based indexing
                    java_problem_to_lines[problem_id].append(self.java_code_lines[idx])
        else: # for "full"
            with open(f"{base_path}/{split}-Java-map.jsonl", 'r') as f:
                for idx, problem_id in enumerate(f.readlines()):
                    java_problem_to_lines[problem_id.strip()].append(self.java_code_lines[idx])

        # Read Python mapping
        python_problem_to_lines = defaultdict(list)

        if format == "snip":
            with open(f"{base_path}/{split}-Python-map.jsonl", 'r') as f:
                for line in f.readlines():
                    problem_id, _, idx = line.strip().split('-')
                    idx = int(idx) - 1  # convert to zero-based indexing
                    python_problem_to_lines[problem_id].append(self.python_code_lines[idx])
        else: # for "full"
            with open(f"{base_path}/{split}-Python-map.jsonl", 'r') as f:
                for idx, problem_id in enumerate(f.readlines()):
                    python_problem_to_lines[problem_id.strip()].append(self.python_code_lines[idx])

        # Aggregate code by problem ID
        self.java_samples = [''.join(java_problem_to_lines[pid]) for pid in java_problem_to_lines]
        self.python_samples = [''.join(python_problem_to_lines[pid]) for pid in python_problem_to_lines]

        # Generate asts
        self.java_ast_data = [generate_java_ast(x, java_parser) for x in self.java_samples]
        self.python_ast_data = [generate_python_ast(x, python_parser) for x in self.python_samples]

        python_vocab_builder = PythonAstVocabularyBuilder()
        for x in self.python_samples:
          python_vocab_builder.update_vocabulary(x)

        java_vocab_builder = JavaAstVocabularyBuilder()
        for x in self.java_ast_data:
          java_vocab_builder.update_vocabulary(x)

    def __len__(self):
        return len(self.java_samples)

    def __getitem__(self, idx):
        tokenizer_args = {
          'truncation': True,
          'max_length': 512,
          'padding': 'max_length',
          'return_tensors': 'pt'
        }

        code_ast_data = {
            'java_code': self.java_samples[idx],
            'java_ast': self.java_ast_data[idx],
            'python_code': self.python_samples[idx],
            'python_ast': self.python_ast_data[idx]
            }

        # Encode the source and target code strings
        source_encoded = self.tokenizer(code_ast_data['java_code'].replace('(','').replace(')','').replace(':',''),
                                        **tokenizer_args)
        target_encoded = self.tokenizer(code_ast_data['python_code'].replace('(','').replace(')','').replace(':',''),
                                        **tokenizer_args)

        # Encode the ASTs
        source_ast_encoded = self.tokenizer(code_ast_data['java_ast'], **tokenizer_args)
        target_ast_encoded = self.tokenizer(code_ast_data['python_ast'], **tokenizer_args)

        return {
            'source_input_ids': source_encoded['input_ids'].squeeze().clone().detach(),
            'source_attention_mask': source_encoded['attention_mask'].squeeze().clone().detach(),
            'source_program': code_ast_data['java_code'],   # Return raw code string
            'source_ast_input_ids': source_ast_encoded['input_ids'].squeeze().clone().detach(),
            'source_ast_attention_mask': source_ast_encoded['attention_mask'].squeeze().clone().detach(),
            'target_input_ids': target_encoded['input_ids'].squeeze().clone().detach(),
            'decoder_input_ids': target_encoded['input_ids'].squeeze().clone().detach(),
            'target_attention_mask': target_encoded['attention_mask'].squeeze().clone().detach(),
            'target_program': code_ast_data['python_code'],   # Return raw code string
            'target_ast_input_ids': target_ast_encoded['input_ids'].squeeze().clone().detach(),
            'labels': target_encoded['input_ids'].squeeze().clone().detach()
        }