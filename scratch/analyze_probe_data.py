import json
import numpy as np

def analyze():
    filepath = "c:/Users/iamsa/Documents/HTSPC/HTSPC-H3/translation_probe.json"
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=== META ===")
    for k, v in data["meta"].items():
        print(f"{k}: {v}")

    print("\n=== LAYER SUMMARIES ===")
    for k, v in data["layer_summaries"].items():
        print(f"Layer: {k}")
        for sk, sv in v.items():
            print(f"  {sk}: {sv:.5f}")

    print("\n=== TIMESCALES ===")
    for layer, info in data["timescales"].items():
        tau = np.array(info["tau"])
        print(f"Layer: {layer}")
        print(f"  Tau: mean={tau.mean():.2f}, std={tau.std():.2f}, min={tau.min():.2f}, max={tau.max():.2f}")

    print("\n=== WRITE SALIENCE TOP 15 ===")
    sal = data["write_salience"]
    sorted_sal = sorted(sal, key=lambda x: x["salience"], reverse=True)
    for i, item in enumerate(sorted_sal[:15]):
        print(f"{i+1:2d}. Pos: {item['pos']:3d} | Char: {repr(item['char']):6s} | Byte: {item['byte']:3d} | Salience: {item['salience']:.5f}")

    print("\n=== LAYER ACTIVATIONS DETAILED ANALYSIS ===")
    # Let's check for any potential anomalies, like values being nan, inf, or stuck at exactly 0 or 1
    # Check for dead heads (alpha very close to 0 or 1 across all steps, or beta very close to 0)
    for layer, activations in data["layer_activations"].items():
        alpha = np.array(activations["alpha"])  # (T, H)
        beta = np.array(activations["beta"])    # (T, H)
        og = np.array(activations["out_gate_norm"]) # (T, H)
        state_norms = np.array(activations["state_norms"]) # (T, H)
        write_mag = np.array(activations["write_magnitudes"]) # (T, H)
        T, H = alpha.shape
        print(f"\nLayer: {layer} (T={T}, H={H})")
        # Check dead alpha heads
        # alpha near 1 means infinite memory, alpha near 0 means no memory
        for h in range(H):
            h_alpha = alpha[:, h]
            h_beta = beta[:, h]
            h_state = state_norms[:, h]
            print(f"  Head {h}: alpha_mean={h_alpha.mean():.4f} (std={h_alpha.std():.4f}), beta_mean={h_beta.mean():.4f} (std={h_beta.std():.4f}), last_state_norm={h_state[-1]:.4f}")

    # Let's look at the correlation between salience/beta and token types (e.g., vowels, spaces, punctuation)
    # Check punctuation, spaces, alphanumeric
    categories = {"space": [], "punctuation": [], "letter": [], "digit": []}
    for item in sal:
        char = item["char"]
        salience = item["salience"]
        if char == " ":
            categories["space"].append(salience)
        elif not char.isalnum():
            categories["punctuation"].append(salience)
        elif char.isdigit():
            categories["digit"].append(salience)
        else:
            categories["letter"].append(salience)
            
    print("\n=== SALIENCE BY CHARACTER CATEGORY ===")
    for cat, vals in categories.items():
        if vals:
            print(f"Category: {cat:<12} | Count: {len(vals):3d} | Mean Salience: {np.mean(vals):.4f} | Max Salience: {np.max(vals):.4f}")

if __name__ == "__main__":
    analyze()
