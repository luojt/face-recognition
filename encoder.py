"""编码器：将 known_faces 目录下的人脸图片编码为特征向量并保存

使用 InsightFace 模型提取人脸特征向量，支持：
- 一人多图：同名前缀的图片自动合并为多个独立特征向量
  比对时只要命中任意一张即匹配成功，比取平均向量更准确
- 人脸对齐：buffalo_l 模型内置关键点对齐，提升侧脸/偏转识别率
"""

import os
import pickle
import re
from collections import defaultdict

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from config import (
    KNOWN_FACES_DIR,
    ENCODINGS_FILE,
    SUPPORTED_IMAGE_FORMATS,
    INSIGHTFACE_MODEL,
    DET_SIZE,
)


def init_insightface():
    """初始化 InsightFace 人脸分析器"""
    print(f"[INFO] 正在加载 InsightFace 模型 '{INSIGHTFACE_MODEL}' ...")
    app = FaceAnalysis(name=INSIGHTFACE_MODEL)
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    print(f"[INFO] 模型加载完成（检测分辨率: {DET_SIZE}）。")
    return app


def parse_name(filename):
    """
    从文件名解析人名。
    支持一人多图：
      张三.jpg          → 张三
      张三-正面.jpg     → 张三
      张三 (1).jpg      → 张三
      张三_侧脸.png     → 张三
      李四.jpg          → 李四

    规则：取 '-' '(' '_' ' ' 之前的纯文件名部分作为人名。
    """
    # 去掉扩展名
    name_no_ext = os.path.splitext(filename)[0]
    # 按分隔符取第一部分作为人名
    base_name = re.split(r'[-_(（\s]', name_no_ext, maxsplit=1)[0].strip()
    return base_name if base_name else name_no_ext


def encode_known_faces():
    """遍历 known_faces 目录，编码所有图片中的人脸，一人多图自动合并"""
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"[INFO] '{KNOWN_FACES_DIR}' 目录不存在，正在创建...")
        os.makedirs(KNOWN_FACES_DIR)
        print(f"[INFO] 请将已知人脸的图片放入 '{KNOWN_FACES_DIR}/' 目录中，")
        print(f"       文件名即为人名（例如：张三.jpg 或 张三-正面.jpg、张三-侧脸.jpg）")
        return

    # 遍历目录中的所有图片
    image_files = [
        f for f in os.listdir(KNOWN_FACES_DIR)
        if f.lower().endswith(SUPPORTED_IMAGE_FORMATS)
    ]

    if not image_files:
        print(f"[WARN] '{KNOWN_FACES_DIR}/' 目录中没有找到图片文件。")
        print(f"       支持的格式：{', '.join(SUPPORTED_IMAGE_FORMATS)}")
        print(f"       请放入图片后重新运行。")
        return

    print(f"[INFO] 找到 {len(image_files)} 张图片，初始化模型...")

    # 初始化 InsightFace
    app = init_insightface()

    # 按人名分组：{ "张三": [encoding1, encoding2, ...], "李四": [encoding1], ... }
    person_encodings = defaultdict(list)

    print(f"[INFO] 开始编码（支持一人多图自动合并）...")

    for filename in image_files:
        filepath = os.path.join(KNOWN_FACES_DIR, filename)
        name = parse_name(filename)

        print(f"  - 正在处理: {filename} → 人物: {name}")

        try:
            image = cv2.imread(filepath)
            if image is None:
                print(f"    [ERROR] 无法读取图片，跳过。")
                continue

            # 使用 InsightFace 检测人脸并提取特征（buffalo_l 内置人脸对齐）
            faces = app.get(image)

            if len(faces) == 0:
                print(f"    [WARN] 未检测到人脸，跳过此图片。")
                continue

            if len(faces) > 1:
                print(f"    [INFO] 检测到 {len(faces)} 张人脸，使用置信度最高的一张。")
                faces.sort(key=lambda x: x.get('det_score', 0), reverse=True)

            # 使用 embedding（特征向量）
            face = faces[0]
            if hasattr(face, 'embedding') and face.embedding is not None:
                # 归一化特征向量
                embedding = face.embedding / np.linalg.norm(face.embedding)
                person_encodings[name].append(embedding)
                print(f"    [OK] 编码成功 (置信度: {face.get('det_score', 0):.2%})")
            else:
                print(f"    [WARN] 无法提取特征向量，跳过。")

        except Exception as e:
            print(f"    [ERROR] 处理失败: {e}")
            continue

    if not person_encodings:
        print("[ERROR] 没有成功编码任何人脸，请检查图片质量。")
        return

    # 一人多图：保留所有独立特征向量，比对时命中任意一张即匹配
    known_encodings = []
    known_names = []

    for name in sorted(person_encodings.keys()):
        encodings = person_encodings[name]
        for i, enc in enumerate(encodings):
            known_encodings.append(enc)
            known_names.append(name)
        if len(encodings) > 1:
            print(f"[INFO] {name}: 保存 {len(encodings)} 个独立特征向量")
        else:
            print(f"[INFO] {name}: 保存 1 个特征向量")

    # 保存到 pickle 文件
    data = {
        "encodings": known_encodings,
        "names": known_names,
    }

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    # 统计信息
    total_photos = sum(len(v) for v in person_encodings.values())
    print(f"\n{'='*50}")
    print(f"[DONE] 编码完成！")
    print(f"  - 人物数量: {len(known_names)} 人")
    print(f"  - 照片总数: {total_photos} 张")
    print(f"  - 人物列表: {', '.join(dict.fromkeys(known_names))}")
    print(f"  - 编码文件: {ENCODINGS_FILE}")
    print(f"{'='*50}")


if __name__ == "__main__":
    encode_known_faces()
