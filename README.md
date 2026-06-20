# 实时人脸识别系统

基于摄像头实时检测人脸，用**绿色方框**框出人脸并显示人名。

## 效果演示

```
┌──────────────────────────────────────────┐
│  FPS: 25.3                               │
│  Known: 3 | Faces: 2                     │
│                                          │
│        ┌──────────────┐                  │
│        │  张三 85%     │                  │
│        └──────────────┘                  │
│        ┌──────────────┐                  │
│        │  Unknown      │                  │
│        └──────────────┘                  │
└──────────────────────────────────────────┘
```

## 技术栈

| 组件 | 用途 |
|------|------|
| **InsightFace** | 人脸检测 + 特征提取与比对（buffalo_l 模型，640px 检测分辨率） |
| **OpenCV** | 摄像头采集、图像渲染、方框绘制 |
| **Pillow** | 中文人名渲染 |
| **NumPy** | 特征向量计算 |
| **ONNX Runtime** | 模型推理引擎 |

### 识别优化

| 优化项 | 说明 |
|--------|------|
| **高分辨率检测** | 640×640 检测分辨率，比默认 320 更精准 |
| **一人多图** | 每人可放多张不同角度/光照的照片，保留独立特征向量，命中任意一张即匹配 |
| **人脸对齐** | buffalo_l 模型内置 106 点关键点对齐，侧脸识别率大幅提升 |
| **余弦相似度** | 使用余弦相似度替代欧氏距离，InsightFace 官方推荐的比对算法 |

## 环境要求

- Python >= 3.9
- macOS / Windows / Linux
- 摄像头

## 安装

```bash
cd python-deal/face-recognition
pip install -r requirements.txt
```

## 快速开始

### 第一步：添加已知人脸

将人物照片放入 `known_faces/` 目录，**文件名即为人名**（不含扩展名）：

```
known_faces/
├── 张三.jpg           ← 识别后显示 "张三"
├── 张三-微笑.jpg      ← 自动合并到 "张三"
├── 张三-侧脸.jpg      ← 自动合并到 "张三"
├── 李四.png           ← 识别后显示 "李四"
└── alice.jpg          ← 识别后显示 "alice"
```

> **提高识别率的关键**：每人放 3-5 张不同角度、不同光照、不同表情的照片，文件名前缀相同即可自动合并。支持的分隔符：`-` `_` `(` `（` 空格。

### 第二步：编码人脸

```bash
python encoder.py
```

程序会：
- 自动下载 InsightFace 模型（首次运行，约 22MB，只需下载一次）
- 读取 `known_faces/` 下所有图片，自动按人名分组
- 提取每张图片的人脸特征向量（内置人脸对齐）
- 同一人物的多张照片取平均向量，提升泛化能力
- 生成 `known_encodings.pkl` 编码文件

### 第三步：启动识别

```bash
python detector.py
```

- 摄像头自动打开
- 检测到的人脸用**绿色方框**框出
- 已知人员显示**人名 + 置信度**（如 "张三 85%"），未知人员显示 **Unknown**
- 左上角显示实时 FPS 和识别状态
- 按 **`q`** 或 **`ESC`** 键退出

## 配置说明

编辑 `config.py` 可调整以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CAMERA_ID` | `0` | 摄像头设备 ID（外接摄像头改为 `1`） |
| `DET_SIZE` | `(640, 640)` | 人脸检测分辨率，越大越准（320/640/1280） |
| `SIMILARITY_THRESHOLD` | `0.35` | 人脸匹配阈值（余弦相似度），越高越严格（0.25~0.50） |
| `DETECTION_INTERVAL` | `3` | 每 N 帧做一次完整识别，值越大性能越好但延迟越高 |
| `INSIGHTFACE_MODEL` | `"buffalo_l"` | 识别模型（`buffalo_l` 精度高，`buffalo_s` 速度快） |
| `FONT_SIZE` | `28` | 人名字体大小（像素） |
| `BOX_COLOR` | `(0, 255, 0)` | 人脸框颜色（BGR 格式，默认绿色） |

## 文件说明

```
face-recognition/
├── requirements.txt       # Python 依赖
├── config.py              # 配置参数
├── encoder.py             # 人脸编码工具（支持一人多图）
├── detector.py            # 实时检测与识别主程序
├── fonts/                 # 中文字体（文泉驿微米黑，开源免费）
│   └── wqy-microhei.ttc
├── known_faces/           # 已知人脸图片目录
│   ├── 张三.jpg
│   ├── 张三-侧脸.jpg
│   └── 李四.png
└── known_encodings.pkl    # 编码文件（运行 encoder.py 后生成）
```

## 常见问题

**Q: macOS 上摄像头打不开，提示 `not authorized to capture video`？**

A: 这是 macOS 隐私权限限制。解决方法：
1. 打开 **系统设置** → **隐私与安全性** → **摄像头**
2. 找到你运行 Python 的终端应用（如终端、iTerm、VS Code），勾选允许
3. **完全退出终端后重新打开**，再运行 `python detector.py`

> 如果用的是 VS Code 内置终端，需要授权 **Visual Studio Code**。

**Q: 首次运行报错 `InvalidProtobuf` 或模型加载失败？**

A: InsightFace 模型文件下载不完整，删除后重新下载：
```bash
rm -rf ~/.insightface/models/buffalo_l
python encoder.py
```

**Q: 识别不准 / 经常识别成 Unknown？**

A: 按优先级尝试：
1. **查看调试输出**：运行 `detector.py` 时终端会打印 `[DEBUG]` 相似度值，观察你本人匹配时的相似度是多少
2. **调整阈值**：编辑 `config.py`，将 `SIMILARITY_THRESHOLD` 设为比你实际相似度低一点的值（如实际 0.40 就设 0.35）
3. **增加照片**：每人放 3-5 张不同角度/光照的照片，重新运行 `encoder.py`
4. **提高分辨率**：将 `DET_SIZE` 改为 `(1280, 1280)`（更准但更慢）

**Q: 不同人被识别成同一个人？**

A: 说明阈值太宽松。编辑 `config.py`，将 `SIMILARITY_THRESHOLD` 调高（如从 0.35 调到 0.45）。

**Q: 运行卡顿？**

A: 增大 `DETECTION_INTERVAL`（如改为 `5`），或将 `DET_SIZE` 改为 `(320, 320)`，或换用 `buffalo_s` 模型。

**Q: 未检测到人脸？**

A: 确保环境光线充足，人脸正对摄像头，距离在 2 米以内。
