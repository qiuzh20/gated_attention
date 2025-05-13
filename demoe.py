import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np

# Set device
base_dir = './' # change to your hf model directory
device = "cuda" if torch.cuda.is_available() else "cpu"

# Loop through different model variants
for name in ['baseline', 'gate_elementwise', 'gate_headwise']:
    
    # Load model and tokenizer
    model_name_or_path = f"{base_dir}/1B_{name}"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).to(device)

    # Input text
    prompt = "Sparse gating mechanism mitigates attention sink."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Forward pass with output_attentions=True to retrieve attention scores
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_attentions=True  # Retrieve attention scores
        )

    # Extract attention scores
    attentions = outputs.attentions  # tuple of tensors: (layer) -> (batch, head, seq_len, seq_len)

    # Function to average attention scores across all heads for each layer
    def average_heads(attentions):
        averaged = []
        for layer_attn in attentions:
            # layer_attn: (batch, head, seq_len, seq_len)
            avg_attn = layer_attn.mean(dim=1).cpu().numpy()  # (batch, seq_len, seq_len)
            averaged.append(avg_attn[0])  # Take the first sample
        return averaged

    averaged_attentions = average_heads(attentions)

    # Get tokens for axis labels
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Visualize attention maps of selected layers
    layers_to_visualize = [0, 6, 20, 27]  # Python indices start at 0, corresponds to 1st, 7th, 21st, 28th layers
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, layer_idx in enumerate(layers_to_visualize):
        attn_map = averaged_attentions[layer_idx]

        # Plot attention map
        ax = axes[idx]
        im = ax.imshow(attn_map, cmap="viridis")

        # Add colorbar
        fig.colorbar(im, ax=ax)

        # Set title
        ax.set_title(f"Layer {layer_idx + 1}")

        # Set ticks and labels
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticklabels(tokens)

        # Hide tick marks
        ax.tick_params(axis='both', which='both', length=0)

    plt.tight_layout()
    plt.savefig(f"{name}_selected_layer_attention_maps.png")
    plt.show()
