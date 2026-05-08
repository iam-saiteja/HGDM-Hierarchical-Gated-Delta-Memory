import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def render_results():
    print("Generating visual plots from raw byte hallucinations for the paper...")
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. AUDIO (PCM Waveform Plot)
    if os.path.exists("generated_audio.raw"):
        with open("generated_audio.raw", "rb") as f:
            raw_audio = f.read()
            
        # Unpack the raw bytes into 16-bit integers
        num_samples = len(raw_audio) // 2
        try:
            samples = struct.unpack(f'<{num_samples}h', raw_audio[:num_samples*2])
            axs[0].plot(samples, color='blue', linewidth=0.5)
            axs[0].set_title("Hallucinated Audio (PCM Waveform)")
            axs[0].set_xlabel("Time (Samples)")
            axs[0].set_ylabel("Amplitude")
        except Exception as e:
            axs[0].set_title("Audio Data Incomplete")
    else:
        axs[0].set_title("generated_audio.raw not found")

    # 2. IMAGE (RGB Pixel Array Plot)
    if os.path.exists("generated_image.raw"):
        with open("generated_image.raw", "rb") as f:
            raw_image = f.read()
            
        # Group bytes into RGB triplets
        pixels = [b for b in raw_image]
        # Calculate largest square we can form
        num_rgb = len(pixels) // 3
        side_len = int(np.sqrt(num_rgb))
        
        if side_len > 0:
            usable_bytes = side_len * side_len * 3
            img_array = np.array(pixels[:usable_bytes]).reshape((side_len, side_len, 3))
            axs[1].imshow(img_array.astype(np.uint8))
            axs[1].set_title(f"Hallucinated Image ({side_len}x{side_len} RGB)")
            axs[1].axis('off')
        else:
            axs[1].set_title("Not enough Image bytes")
    else:
        axs[1].set_title("generated_image.raw not found")

    # 3. VIDEO (RGB Pixel Array Plot - Partial Frame)
    if os.path.exists("generated_video.raw"):
        with open("generated_video.raw", "rb") as f:
            raw_video = f.read()
            
        pixels = [b for b in raw_video]
        num_rgb = len(pixels) // 3
        side_len = int(np.sqrt(num_rgb))
        
        if side_len > 0:
            usable_bytes = side_len * side_len * 3
            vid_array = np.array(pixels[:usable_bytes]).reshape((side_len, side_len, 3))
            axs[2].imshow(vid_array.astype(np.uint8))
            axs[2].set_title(f"Hallucinated Video Frame ({side_len}x{side_len})")
            axs[2].axis('off')
        else:
            axs[2].set_title("Not enough Video bytes")
    else:
        axs[2].set_title("generated_video.raw not found")

    plt.tight_layout()
    plt.savefig("multimodal_hallucination_proof.png", dpi=300)
    print("Success! Saved visualization to 'multimodal_hallucination_proof.png'")

if __name__ == "__main__":
    render_results()
