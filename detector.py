"""实时人脸检测与识别：通过摄像头实时识别人脸，绿色方框框出并显示人名

技术栈：
- InsightFace：高精度人脸检测（640px）+ 特征提取与比对（内置人脸对齐）
- OpenCV：摄像头采集与画面渲染
- Pillow：中文字体渲染
"""

import os
import pickle
import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, ImageFont

from config import (
    CAMERA_ID,
    SIMILARITY_THRESHOLD,
    DETECTION_INTERVAL,
    ENCODINGS_FILE,
    INSIGHTFACE_MODEL,
    DET_SIZE,
    FONT_PATH,
    FONT_SIZE,
    BOX_COLOR,
    BOX_THICKNESS,
    LABEL_BG_COLOR,
    LABEL_TEXT_COLOR,
    WINDOW_NAME,
)


def load_known_faces():
    """加载预编码的已知人脸数据"""
    if not os.path.exists(ENCODINGS_FILE):
        print(f"[WARN] 编码文件 '{ENCODINGS_FILE}' 不存在。")
        print(f"       请先运行: python encoder.py")
        return [], []

    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)

    print(f"[INFO] 已加载 {len(data['names'])} 个已知人脸：{', '.join(data['names'])}")
    return data["encodings"], data["names"]


def init_insightface():
    """初始化 InsightFace 人脸分析器（同时用于检测和识别）"""
    print(f"[INFO] 正在加载 InsightFace 模型 '{INSIGHTFACE_MODEL}' ...")
    app = FaceAnalysis(name=INSIGHTFACE_MODEL, providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    print(f"[INFO] 模型加载完成（检测分辨率: {DET_SIZE}）。")
    return app


def load_font():
    """加载中文字体，失败时回退到默认字体"""
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        print(f"[INFO] 中文字体加载成功: {FONT_PATH}")
        return font
    except (OSError, IOError):
        print(f"[WARN] 无法加载字体 '{FONT_PATH}'，使用默认字体（可能不支持中文）。")
        return ImageFont.load_default()


def draw_label_pillow(frame, text, top, left, bottom, right, font):
    """使用 Pillow 绘制中文人名标签，贴回 OpenCV 画面"""
    # 用 Pillow 测量文字尺寸
    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # 标签背景位置（在框顶部上方）
    padding = 8
    label_height = text_height + padding * 2
    label_width = text_width + padding * 2
    label_top = top - label_height
    if label_top < 0:
        label_top = bottom + 4  # 上方空间不够，放框下方

    label_left = left
    label_right = left + label_width

    # 确保标签不超出画面边界
    h, w = frame.shape[:2]
    label_left = max(0, label_left)
    label_right = min(w, label_right)
    label_top = max(0, label_top)
    label_bottom = min(h, label_top + label_height)

    # 在帧上绘制背景矩形
    cv2.rectangle(frame, (label_left, label_top), (label_right, label_bottom),
                  LABEL_BG_COLOR, cv2.FILLED)

    # 用 Pillow 渲染中文到裁剪区域，然后贴回 OpenCV frame
    roi = frame[label_top:label_bottom, label_left:label_right]
    if roi.size == 0:
        return

    # BGR → RGB（Pillow 用 RGB）
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(roi_rgb)
    draw = ImageDraw.Draw(pil_img)

    # 绘制文字（白色）
    text_x = padding
    text_y = padding
    draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))

    # RGB → BGR 贴回
    roi_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    frame[label_top:label_bottom, label_left:label_right] = roi_bgr


def main():
    """主循环：打开摄像头，实时检测和识别人脸"""
    # 加载已知人脸编码
    known_encodings, known_names = load_known_faces()
    has_known_faces = len(known_encodings) > 0

    # 初始化 InsightFace（同时负责检测和识别）
    insightface_app = init_insightface()

    # 加载中文字体
    font = load_font()

    # 打开摄像头
    print(f"[INFO] 正在打开摄像头 (ID={CAMERA_ID})...")
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 (ID={CAMERA_ID})！")
        print("        请检查摄像头连接，或修改 config.py 中的 CAMERA_ID。")
        return

    print("[INFO] 摄像头已打开，按 'q' 键退出。")
    print(f"[INFO] 已知人脸数量: {len(known_names)}")
    print(f"[INFO] 匹配阈值: {SIMILARITY_THRESHOLD} | 检测间隔: 每 {DETECTION_INTERVAL} 帧")

    # 上一轮识别结果缓存
    prev_face_locations = []
    prev_face_names = []
    frame_count = 0
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] 读取摄像头画面失败！")
                break

            frame_count += 1

            current_face_locations = []
            current_face_names = []

            # 每隔 N 帧执行完整的人脸检测+识别
            if frame_count % DETECTION_INTERVAL == 0:
                # InsightFace 一次性完成检测 + 特征提取（内置人脸对齐）
                faces = insightface_app.get(frame)

                for face in faces:
                    # 解析人脸框坐标
                    bbox = face.bbox.astype(int)
                    left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]

                    # 边界检查
                    h, w = frame.shape[:2]
                    left = max(0, left)
                    top = max(0, top)
                    right = min(w, right)
                    bottom = min(h, bottom)

                    name = "Unknown"
                    confidence = ""

                    if has_known_faces and hasattr(face, 'embedding') and face.embedding is not None:
                        # 归一化特征向量
                        embedding = face.embedding / np.linalg.norm(face.embedding)

                        # 与已知人脸比对（计算余弦相似度 — 比欧氏距离更准确）
                        # InsightFace 官方推荐使用余弦相似度进行人脸比对
                        similarities = [np.dot(embedding, ke) for ke in known_encodings]
                        max_similarity = max(similarities)
                        max_index = similarities.index(max_similarity)

                        if max_similarity > SIMILARITY_THRESHOLD:
                            name = known_names[max_index]
                            # 计算匹配置信度
                            confidence = f" {max_similarity:.0%}"

                        # 调试：打印最近的匹配信息
                        print(f"    [DEBUG] 最近匹配: {known_names[max_index]} "
                              f"相似度={max_similarity:.4f} "
                              f"({'命中' if name != 'Unknown' else '未命中'} "
                              f"阈值={SIMILARITY_THRESHOLD})")

                    current_face_locations.append((top, right, bottom, left))
                    current_face_names.append(name + confidence)

                # 更新缓存
                prev_face_locations = current_face_locations
                prev_face_names = current_face_names
            else:
                # 非识别帧：复用上次结果
                current_face_locations = prev_face_locations
                current_face_names = prev_face_names

            # 在原图上绘制结果
            for (top, right, bottom, left), name in zip(
                current_face_locations, current_face_names
            ):
                # 绘制绿色人脸框
                cv2.rectangle(frame, (left, top), (right, bottom),
                              BOX_COLOR, BOX_THICKNESS)

                # 使用 Pillow 绘制中文人名标签（含置信度）
                draw_label_pillow(frame, name, top, left, bottom, right, font)

            # 计算并显示 FPS
            fps_frame_count += 1
            if fps_frame_count >= 30:
                elapsed = time.time() - fps_start_time
                fps_display = fps_frame_count / elapsed if elapsed > 0 else 0
                fps_start_time = time.time()
                fps_frame_count = 0

            # 在画面左上角显示状态信息
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Known: {len(known_names)} | Faces: {len(current_face_locations)}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # 显示画面
            cv2.imshow(WINDOW_NAME, frame)

            # 按键处理：'q' 或 ESC 退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[INFO] 用户退出。")
                break

    except KeyboardInterrupt:
        print("\n[INFO] 收到中断信号，正在退出...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] 摄像头已释放，程序结束。")


if __name__ == "__main__":
    main()
