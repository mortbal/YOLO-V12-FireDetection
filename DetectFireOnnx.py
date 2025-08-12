import cv2
import numpy as np
import onnxruntime as ort

# ==== CONFIGURATION ====
onnx_path = "yolov12_fire_smoke.onnx"  # Your YOLOv12 ONNX file
input_video = "fire.mp4"
output_video = "fire_detected.mp4"
confidence_threshold = 0.25
nms_threshold = 0.45
target_classes = ["fire", "smoke"]  # Class names in training order

# ==== LOAD MODEL ====
session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# ==== HELPER FUNCTIONS ====
def preprocess(frame, img_size=640):
    h, w = frame.shape[:2]
    scale = img_size / max(h, w)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
    new_frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    new_frame[:resized.shape[0], :resized.shape[1]] = resized
    blob = new_frame[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB, HWC → CHW
    blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
    return blob, scale

def postprocess(pred, scale, orig_shape):
    boxes, scores, class_ids = [], [], []
    for det in pred:
        if det[4] * det[5] > confidence_threshold:
            x, y, w, h = det[:4] / scale
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            cls_id = int(det[6])
            boxes.append([x1, y1, x2, y2])
            scores.append(det[4] * det[5])
            class_ids.append(cls_id)
    idxs = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, nms_threshold)
    if len(idxs) > 0:
        idxs = idxs.flatten()
    return [(boxes[i], scores[i], class_ids[i]) for i in idxs]

# ==== VIDEO PROCESSING ====
cap = cv2.VideoCapture(input_video)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blob, scale = preprocess(frame)
    blob = np.expand_dims(blob, axis=0)

    pred = session.run(None, {input_name: blob})[0][0]  # YOLOv12 output

    detections = postprocess(pred, scale, frame.shape[:2])
    for (box, score, cls_id) in detections:
        if target_classes[cls_id] in ["fire", "smoke"]:
            color = (0, 0, 255) if target_classes[cls_id] == "fire" else (255, 255, 0)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, f"{target_classes[cls_id]} {score:.2f}",
                        (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)

cap.release()
out.release()
print(f"✅ Detection complete! Output saved to {output_video}")
