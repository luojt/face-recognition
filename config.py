"""配置文件：人脸识别参数"""

# 摄像头设备 ID（0=默认摄像头，外接摄像头可改为 1 或 2）
CAMERA_ID = 0

# InsightFace 模型名称（buffalo_l 精度最高，buffalo_s 速度更快）
INSIGHTFACE_MODEL = "buffalo_l"

# 人脸检测分辨率（越大检测越准，但速度越慢）
DET_SIZE = (640, 640)

# 人脸匹配相似度阈值（余弦相似度，越高越严格，推荐 0.35）
# 0=完全不同, 1=完全相同, 典型范围 0.25~0.50
SIMILARITY_THRESHOLD = 0.35

# 检测帧间隔：每 N 帧执行一次完整的人脸识别，中间帧复用上次结果（提升性能）
DETECTION_INTERVAL = 3

# 已知人脸编码文件路径
ENCODINGS_FILE = "known_encodings.pkl"

# 已知人脸图片目录
KNOWN_FACES_DIR = "known_faces"

# 人脸框颜色（BGR 格式：绿色）
BOX_COLOR = (0, 255, 0)

# 人脸框线宽（像素）
BOX_THICKNESS = 2

# 标签背景颜色（BGR 格式：深绿色）
LABEL_BG_COLOR = (0, 200, 0)

# 标签文字颜色（BGR 格式：白色）
LABEL_TEXT_COLOR = (255, 255, 255)

# 中文字体路径（项目内置文泉驿微米黑，无需额外安装）
FONT_PATH = "fonts/wqy-microhei.ttc"

# 标签字体大小（像素）
FONT_SIZE = 28

# 窗口名称
WINDOW_NAME = "Face Recognition - Press 'q' to quit"

# 支持的图片格式
SUPPORTED_IMAGE_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
