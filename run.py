from app import app
import os
import sys

# 确保目录存在
def ensure_directories():
    # 创建必要的目录
    directories = [
        app.config['UPLOAD_FOLDER'],
        app.config['SEGMENTATION_RESULT_FOLDER'],
        app.config['DETECTION_RESULT_FOLDER']
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"创建目录: {directory}")
            except Exception as e:
                print(f"创建目录失败 {directory}: {e}")

# 检查Python版本
def check_python_version():
    required_version = (3, 6)
    current_version = sys.version_info
    if current_version < required_version:
        print(f"警告: Python版本 {current_version[0]}.{current_version[1]} 可能不支持文件夹上传功能。推荐使用Python 3.6或更高版本。")

if __name__ == '__main__':
    # 检查Python版本
    check_python_version()
    
    # 确保目录存在
    ensure_directories()
    
    # 启动应用
    app.run(debug=True, host='0.0.0.0', port=5000)