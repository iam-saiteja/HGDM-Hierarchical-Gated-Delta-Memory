import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def render_results():
    print("Generating visual plots from raw byte hallucinations for the paper...")
    
    # We use a custom layout: Audio (Spectrogram), Image (Full), Video (4 Frames)
    fig = plt.figure(figsize=(18, 5))
    
    # 1. AUDIO (Spectrogram Plot)
    ax0 = fig.add_subplot(1, 3, 1)
    if os.path.exists("generated_audio.raw"):
        with open("generated_audio.raw", "rb") as f:
            raw_audio = f.read()
            
        num_samples = len(raw_audio) // 2
        try:
            samples = struct.unpack(f'<{num_samples}h', raw_audio[:num_samples*2])
            samples = np.array(samples).astype(np.float32)
            
            # Plot Spectrogram
            ax0.specgram(samples, Fs=44100, NFFT=1024, noverlap=512, cmap='viridis')
            ax0.set_title("Hallucinated Audio (Spectrogram)")
            ax0.set_xlabel("Time (s)")
            ax0.set_ylabel("Frequency (Hz)")
        except Exception as e:
            ax0.set_title(f"Audio Error: {e}")
    else:
        ax0.set_title("generated_audio.raw not found")

    # 2. IMAGE (Full 256x256 RGB)
    ax1 = fig.add_subplot(1, 3, 2)
    if os.path.exists("generated_image.raw"):
        with open("generated_image.raw", "rb") as f:
            raw_image = f.read()
            
        pixels = np.frombuffer(raw_image, dtype=np.uint8)
        # Attempt to reshape as 256x256x3
        try:
            img_array = pixels[:256*256*3].reshape((256, 256, 3))
            ax1.imshow(img_array)
            ax1.set_title("Hallucinated Image (256x256 Fractal)")
            ax1.axis('off')
        except Exception as e:
            ax1.set_title(f"Image Error: {e}")
    else:
        ax1.set_title("generated_image.raw not found")

    # 3. VIDEO (Sequence of 4 Frames)
    # We create a sub-grid for the 4 video frames
    ax2 = fig.add_subplot(1, 3, 3)
    ax2.axis('off')
    ax2.set_title("Hallucinated Video (4 Frames)")
    
    if os.path.exists("generated_video.raw"):
        with open("generated_video.raw", "rb") as f:
            raw_video = f.read()
            
        pixels = np.frombuffer(raw_video, dtype=np.uint8)
        frame_size = 64 * 64 * 3
        
        for i in range(4):
            try:
                start = i * frame_size
                end = (i + 1) * frame_size
                if end <= len(pixels):
                    frame = pixels[start:end].reshape((64, 64, 3))
                    # Plot in a small grid within ax2 or just use inset
                    sub_ax = fig.add_axes([0.68 + (i%2)*0.13, 0.15 + (1-i//2)*0.3, 0.12, 0.3])
                    sub_ax.imshow(frame)
                    sub_ax.axis('off')
            except:
                pass
    else:
        ax2.set_title("generated_video.raw not found")

    plt.tight_layout()
    plt.savefig("hallucination_proof.png", dpi=300)
    print("Success! Saved updated visualization to 'hallucination_proof.png'")

if __name__ == "__main__":
    render_results()
