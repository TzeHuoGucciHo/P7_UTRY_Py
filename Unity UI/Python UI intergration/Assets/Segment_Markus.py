import cv2
import mediapipe as mp
import numpy as np
from tkinter import Tk, filedialog

# -----------------------------
# File Picker
# -----------------------------
def select_image():
    root = Tk()
    root.withdraw()  # hide tkinter window
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*")
        ]
    )
    return file_path


# -----------------------------
# Crop + Resize
# -----------------------------
def crop_with_fixed_height(image_path, output_path, padding=30, target_height=2077):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Could not load image.")

    h, w = img.shape[:2]

    # MediaPipe segmentation
    mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    result = mp_selfie.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    mask = (result.segmentation_mask > 0.3).astype(np.uint8)

    ys, xs = np.where(mask == 1)

    if len(xs) == 0:
        raise ValueError("Could not segment a person in this image.")

    # Bounding box of the person
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    # Crop
    cropped = img[y1:y2, x1:x2]

    # Resize height to 2077px
    ch, cw = cropped.shape[:2]
    scale = target_height / ch
    new_width = int(cw * scale)

    resized = cv2.resize(cropped, (new_width, target_height), interpolation=cv2.INTER_LANCZOS4)

    cv2.imwrite(output_path, resized)
    print(f"Saved cropped image to {output_path}")


# -----------------------------
# Main Program
# -----------------------------
def main():
    print("Select an image to crop...")
    path = select_image()

    if not path:
        print("No image selected.")
        return

    output_path = "cropped_output.png"
    crop_with_fixed_height(path, output_path)

    print("Done! Cropped image saved.")

if __name__ == "__main__":
    main()
