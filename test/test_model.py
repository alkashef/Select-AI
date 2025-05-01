import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import torch

# Global constants
MAX_LENGTH = 1000  # Maximum length for generated responses
MODELS_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
DEFAULT_PROMPT = "Write a Python function to calculate the factorial of a number."
MODEL_NAME = "code-llama-7b-instruct"  # Name of the model directory
DEFAULT_PAD_TOKEN = "[PAD]"  # Default padding token for testing


def check_model_files(model_path: str) -> List[str]:
    """
    Check if all required files for the model exist in the specified directory.

    :param model_path: Path to the locally saved model directory.
    :return: List of files found in the model directory.
    """
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory '{model_path}' does not exist.")

    existing_files = os.listdir(model_path)
    missing_files = [file for file in required_files if file not in existing_files]

    # Check for at least one .bin file
    bin_files = [file for file in existing_files if file.endswith(".bin")]
    if not bin_files:
        missing_files.append("pytorch_model.bin (or equivalent .bin files)")

    if missing_files:
        raise FileNotFoundError(
            f"The following required files are missing in '{model_path}':\n" +
            "\n".join(missing_files) + "\nEnsure the model is downloaded correctly using the 'download_model.py' script."
        )

    print(f"All required files are present in '{model_path}':")
    for file in existing_files:
        print(file)  # Print each file on a new line
    return existing_files


def load_model_and_tokenizer(model_path: str):
    """
    Load the model and tokenizer from the specified local path.

    :param model_path: Path to the locally saved model directory.
    :return: Loaded model and tokenizer.
    """
    print(f"Loading model and tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    # Ensure the tokenizer already has a pad_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
        tokenizer.save_pretrained(model_path)

    # Load the model in PyTorch format
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, local_files_only=True)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on device: {device}")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str) -> str:
    """
    Generate a response from the model based on the given prompt.

    :param model: The loaded model.
    :param tokenizer: The loaded tokenizer.
    :param prompt: The input prompt for the model.
    :return: The generated response as a string.
    """
    print(f"Prompt: {prompt}")

    # Tokenize the input and move to the same device as the model
    device = next(model.parameters()).device
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH  # Explicitly set max_length for tokenization
    ).to(device)

    # Generate the response with attention_mask
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=MAX_LENGTH,  # Use global MAX_LENGTH
        pad_token_id=tokenizer.pad_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def read_prompt_from_file(file_path: str) -> str:
    """
    Read prompt content from a file.
    
    Args:
        file_path: Path to the file containing the prompt
        
    Returns:
        Content of the file as a string
        
    Raises:
        FileNotFoundError: If the file does not exist
        IOError: If there's an error reading the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        print(f"Prompt loaded from file: {file_path}")
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    except IOError as e:
        raise IOError(f"Error reading prompt file: {e}")


def main():
    """
    Main function to test the downloaded model with a given prompt.
    """
    parser = argparse.ArgumentParser(description="Test the downloaded model with a prompt.")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("prompt", type=str, nargs="?", default=None, 
                             help="The input prompt for the model.")
    prompt_group.add_argument("-f", "--file", type=str, 
                             help="Path to a file containing the prompt.")
    args = parser.parse_args()

    # Determine the prompt source
    if args.file:
        args.prompt = read_prompt_from_file(args.file)
    elif args.prompt is None:
        print(f"No prompt provided. Using the default prompt:\n\n'{DEFAULT_PROMPT}'\n")
        proceed = input("Do you want to continue with this example prompt? (y/n): ").strip().lower()
        if proceed != "y":
            print("Exiting...")
            return
        args.prompt = DEFAULT_PROMPT

    # Path to the locally saved model
    model_path = os.path.join(MODELS_DIRECTORY, MODEL_NAME)

    # Check if all required files exist
    check_model_files(model_path)

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Generate and print the response
    response = generate_response(model, tokenizer, args.prompt)
    print("\nModel Response:")
    print(response)


if __name__ == "__main__":
    main()