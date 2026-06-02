import os
import torch
from data_loader import get_data_loaders
from model import TinyGPT
from train import train_one_epoch
from generate import sample_gpt

def main():
    file_path = "/workspaces/GPT-2.0/Todd G. Buchholz - New Ideas from Dead Economists_ The Introduction to Modern Economic Thought, 4th Edition (2021, Penguin Publishing Group).txt" # Make sure your dataset text file is named correctly
    model_path = "tiny_gpt_economists.pth"
    block_size = 64
    batch_size = 64
    epochs = 10000
    max_steps_per_epoch = 300

    print("Loading data...")
    loader, vocab_size, stoi, itos = get_data_loaders(file_path, block_size, batch_size)

    if loader is None:
        print("Please ensure your dataset file is named correctly and exists in the root directory.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = TinyGPT(vocab_size, block_size).to(device)
    
    # ---------------------------------------------------------
    # NEW: Ask user about loading the pretrained model
    # ---------------------------------------------------------
    use_pretrained = input("Would you like to use pretained model? Type Y or N: ").strip().upper()
    
    if use_pretrained == 'Y':
        if os.path.exists(model_path):
            # Load the weights into the model
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"\nSuccessfully loaded weights from {model_path}. Skipping training.")
        else:
            print(f"\nError: '{model_path}' not found in the current directory. Proceeding to train from scratch.")
            use_pretrained = 'N' # Fallback to training mode

    # ---------------------------------------------------------
    # Execute training loop ONLY if we didn't load the pretrained model
    # ---------------------------------------------------------
    if use_pretrained != 'Y':
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        print("\nStarting training. Press Ctrl+C to stop early.")
        try:
            for epoch in range(epochs):
                train_loss = train_one_epoch(model, loader, optimizer, device, max_steps=max_steps_per_epoch)
                print(f"epoch {epoch:2d} | train loss {train_loss:.4f}")
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Retaining current weights.")

    # ---------------------------------------------------------
    # NEW: Dynamic starting text input
    # ---------------------------------------------------------
    print("\n--- Text Generation Playground ---")
    print("Press Ctrl+C at any time to exit the playground.")
    
    try:
        while True:
            start_text = input("\nEnter a starting prompt for the model: ")
            
            # Fallback just in case you hit Enter without typing anything
            if not start_text:
                start_text = "Milton Friedman"
                print(f"Empty input detected. Defaulting to: '{start_text}'")

            print(f"\nGenerating sample text based on '{start_text}'...")
            
            # Generate and print the output
            generated_text = sample_gpt(model, block_size, stoi, itos, device, start_text=start_text, max_new_tokens=500)
            
            print("\n" + "="*50)
            print(generated_text)
            print("="*50)
            
    except KeyboardInterrupt:
        print("\n\nExiting text generation. Goodbye!")

if __name__ == "__main__":
    main()
#     print("\n--- Text Generation ---")
#     start_text = input("Enter a starting prompt for the model: ")
    
#     # Fallback just in case you hit Enter without typing anything
#     if not start_text:
#         start_text = "Milton Friedman"
#         print(f"Empty input detected. Defaulting to: '{start_text}'")

#     print(f"\nGenerating sample text based on '{start_text}'...")
    
#     # Generate and print the output
#     generated_text = sample_gpt(model, block_size, stoi, itos, device, start_text=start_text, max_new_tokens=500)
    
#     print("\n" + "="*50)
#     print(generated_text)
#     print("="*50 + "\n")

# if __name__ == "__main__":
#     main()
