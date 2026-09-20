import os
import cv2
import matplotlib.pyplot as plt


def parse_annotation(txt_path, img_width, img_height):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(',')
        class_id = float(parts[0])

        # Only extract the box if it is the Face class (1.0)
        if class_id == 1.0:
            x_c, y_c, w, h = map(float, parts[1:5])

            x_center = int(x_c * img_width)
            y_center = int(y_c * img_height)
            width = int(w * img_width)
            height = int(h * img_height)

            x1 = max(0, int(x_center - width / 2))
            y1 = max(0, int(y_center - height / 2))
            x2 = min(img_width, int(x_center + width / 2))
            y2 = min(img_height, int(y_center + height / 2))

            return x1, y1, x2, y2

    return None  # In case no face is detected


def test_load_and_plot(video_path, annot_folder, frame_number=1):
    # Extract "001_E_1" from "Dataset/001_E_1.mp4"
    video_prefix = os.path.splitext(os.path.basename(video_path))[0]

    # Construct exact text file name: e.g., "001_E_1_10.txt"
    txt_name = f"{video_prefix}_{frame_number}.txt"
    txt_path = os.path.join(annot_folder, txt_name)

    if not os.path.exists(txt_path):
        print(f"Annotation {txt_path} not found.")
        return

    # OpenCV frames are 0-indexed. Text file "1" corresponds to frame index "0".
    cv2_frame_index = frame_number - 1

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, cv2_frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Failed to read frame {cv2_frame_index} from video.")
        return

    img_height, img_width = frame.shape[:2]

    x1, y1, x2, y2 = parse_annotation(txt_path, img_width, img_height)

    # Crop the face using array slicing
    cropped_face = frame[y1:y2, x1:x2]

    # Convert BGR (OpenCV default) to RGB for Matplotlib visualization
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cropped_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(frame_rgb)
    # Draw the bounding box for verification
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2))
    plt.title(f"Original Frame: {txt_name}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cropped_rgb)
    plt.title("Cropped Face")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sample_video = "../Dataset/001_E_1.mp4"
    sample_annotation_folder = "../annotations/rois/001_E_1"

    # Testing the 10th frame
    test_load_and_plot(sample_video, sample_annotation_folder, frame_number=10)