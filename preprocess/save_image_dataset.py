import os
import cv2

IMG_SIZE = 256


def parse_face_annotation(txt_path, img_width, img_height):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(',')
        if not parts or len(parts) < 5:
            continue

        class_id = float(parts[0])
        if class_id == 1.0:  # 1.0 is the Face class
            x_c, y_c, w, h = map(float, parts[1:5])

            cx = int(x_c * img_width)
            cy = int(y_c * img_height)
            width = int(w * img_width)
            height = int(h * img_height)

            return cx, cy, width, height

    return None


def crop_and_square_face(frame, cx, cy, width, height):
    img_height, img_width = frame.shape[:2]

    # 1. Create a square using the largest dimension + 10% margin
    box_size = int(max(width, height) * 1.1)
    half_size = box_size // 2

    # 2. Calculate ideal square coordinates
    x1 = cx - half_size
    y1 = cy - half_size
    x2 = cx + half_size
    y2 = cy + half_size

    # 3. Handle boundaries (padding with black if it goes off-screen)
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - img_width)
    pad_bottom = max(0, y2 - img_height)

    safe_x1 = max(0, x1)
    safe_y1 = max(0, y1)
    safe_x2 = min(img_width, x2)
    safe_y2 = min(img_height, y2)

    cropped_face = frame[safe_y1:safe_y2, safe_x1:safe_x2]

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        cropped_face = cv2.copyMakeBorder(
            cropped_face, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

    return cropped_face


def process_paired_datasets(videos_dir, annotations_base_dir, output_base_dir):
    error_log = []
    total_saved_pairs = 0

    # Iterate through all 159 people, 2 sessions each
    for person_id in range(1, 160):
        person_str = f"{person_id:03d}"

        for session_id in [1, 2]:
            vid_e_prefix = f"{person_str}_E_{session_id}"
            vid_u_prefix = f"{person_str}_U_{session_id}"

            vid_e_path = os.path.join(videos_dir, f"{vid_e_prefix}.mp4")
            vid_u_path = os.path.join(videos_dir, f"{vid_u_prefix}.mp4")

            # Check if videos exist
            if not os.path.exists(vid_e_path) or not os.path.exists(vid_u_path):
                error_log.append(f"MISSING VIDEO PAIR: {person_str} Session {session_id}")
                continue

            print(f"Processing Pair: {vid_e_prefix} & {vid_u_prefix}...")

            cap_e = cv2.VideoCapture(vid_e_path)
            cap_u = cv2.VideoCapture(vid_u_path)

            # Setup output directories
            save_dir_e = os.path.join(output_base_dir, vid_e_prefix)
            save_dir_u = os.path.join(output_base_dir, vid_u_prefix)
            os.makedirs(save_dir_e, exist_ok=True)
            os.makedirs(save_dir_u, exist_ok=True)

            frame_idx = 1
            while True:
                ret_e, frame_e = cap_e.read()
                ret_u, frame_u = cap_u.read()

                # Check for video frame mismatches
                if not ret_e or not ret_u:
                    if ret_e != ret_u:
                        error_log.append(
                            f"LENGTH MISMATCH: {person_str} Session {session_id} ended unevenly at frame {frame_idx}")
                    break  # Reached the end of one or both videos

                annot_e_path = os.path.join(annotations_base_dir, vid_e_prefix, f"{vid_e_prefix}_{frame_idx}.txt")
                annot_u_path = os.path.join(annotations_base_dir, vid_u_prefix, f"{vid_u_prefix}_{frame_idx}.txt")

                # Check 1: Do BOTH annotation text files exist?
                if not os.path.exists(annot_e_path) or not os.path.exists(annot_u_path):
                    error_log.append(
                        f"MISSING ANNOTATION: {person_str} Session {session_id} Frame {frame_idx}. Discarded.")
                    frame_idx += 1
                    continue

                h_e, w_e = frame_e.shape[:2]
                h_u, w_u = frame_u.shape[:2]

                bbox_e = parse_face_annotation(annot_e_path, w_e, h_e)
                bbox_u = parse_face_annotation(annot_u_path, w_u, h_u)

                # Check 2: Was a Face (1.0) successfully parsed in BOTH?
                if not bbox_e or not bbox_u:
                    error_log.append(
                        f"NO FACE DETECTED: {person_str} Session {session_id} Frame {frame_idx}. Discarded.")
                    frame_idx += 1
                    continue

                # If we passed all checks, process both to squares
                sq_e = crop_and_square_face(frame_e, *bbox_e)
                sq_u = crop_and_square_face(frame_u, *bbox_u)

                # Resize to target dimension (256x256)
                final_e = cv2.resize(sq_e, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                final_u = cv2.resize(sq_u, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

                # Save to disk
                cv2.imwrite(os.path.join(save_dir_e, f"{vid_e_prefix}_{frame_idx}.jpg"), final_e)
                cv2.imwrite(os.path.join(save_dir_u, f"{vid_u_prefix}_{frame_idx}.jpg"), final_u)

                total_saved_pairs += 1
                frame_idx += 1

            cap_e.release()
            cap_u.release()

    # Save the error log
    with open("processing_errors_log.txt", "w") as f:
        for error in error_log:
            f.write(error + "\n")

    print(f"\nProcessing Complete!")
    print(f"Total Perfectly Paired Frames Saved: {total_saved_pairs} (x2 images)")
    print(f"Total Discarded Errors: {len(error_log)}")
    print(f"Check 'processing_errors_log.txt' for details on discarded frames.")


if __name__ == "__main__":
    # Ensure these paths match your environment
    videos_dir = "../Dataset"
    annotations_base_dir = "../annotations/rois"
    output_base_dir = "../Dataset_img"

    process_paired_datasets(videos_dir, annotations_base_dir, output_base_dir)