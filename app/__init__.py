from flask import Flask
import os

# 创建Flask应用实例
app = Flask(__name__)

# 配置项
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['STATIC_FOLDER'] = 'app/static'
app.config['UPLOAD_FOLDER'] = os.path.join('app', 'static', 'uploads')
app.config['SEGMENTATION_RESULT_FOLDER'] = os.path.join('app', 'static', 'segmentation_results')
app.config['DETECTION_RESULT_FOLDER'] = os.path.join('app', 'static', 'detection_results')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB文件大小限制

# 确保使用HTTP而不是HTTPS
app.config['SESSION_COOKIE_SECURE'] = False  # 允许在HTTP上使用会话cookie
app.config['PREFERRED_URL_SCHEME'] = 'http'  # 优先使用HTTP

# 导入路由
from app import routes

# 导入模型加载函数
from app.model_loader import load_models

# 在应用启动时加载模型和创建目录
with app.app_context():
    load_models()
    # 确保必要的目录存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['SEGMENTATION_RESULT_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DETECTION_RESULT_FOLDER'], exist_ok=True)
    # 创建对比结果目录
    comparison_folder = os.path.join(app.config['STATIC_FOLDER'], 'comparison_results')
    os.makedirs(comparison_folder, exist_ok=True)