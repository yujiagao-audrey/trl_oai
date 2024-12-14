from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Get the name of the base model
base_model_name = "EleutherAI/pythia-1b"

# Get the name of the adapter model
adapter_model_name = "XueyingJia/pythia-1b-sft-SG"

# Define the name of the combined model
combined_model_name = "XueyingJia/pythia-1b-sft-SG-merge"

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load the LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_model_name)

# Merge LoRA weights into the base model
model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(adapter_model_name)

# Save the combined model
model.push_to_hub(combined_model_name)
tokenizer.push_to_hub(combined_model_name)