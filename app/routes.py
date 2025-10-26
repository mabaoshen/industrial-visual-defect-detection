from flask import render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import os
from app import app
from app.utils import apply_segmentation, apply_detection, allowed_file

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和分析"""
    # 检查请求中是否包含文件部分
    if 'file' not in request.files:
        flash('没有文件部分')
        return redirect(request.url)
    
    file = request.files['file']
    
    # 如果用户没有选择文件
    if file.filename == '':
        flash('未选择文件')
        return redirect(request.url)
    
    # 检查文件类型
    if not allowed_file(file.filename):
        flash('不支持的文件类型')
        return redirect(request.url)
    
    # 保存上传的文件
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # 获取处理类型
    process_type = request.form.get('process_type', 'segmentation')
    
    # 根据选择的处理类型执行相应的处理
    if process_type == 'segmentation':
        # 图像分割
        result_filename = f"segmented_{filename}"
        result_path = os.path.join(app.config['SEGMENTATION_RESULT_FOLDER'], result_filename)
        success, message = apply_segmentation(filepath, result_path)
        
        if success:
            # 构建结果URL
            upload_url = url_for('uploaded_file', folder='uploads', filename=filename)
            result_url = url_for('uploaded_file', folder='segmentation_results', filename=result_filename)
            
            return render_template('index.html', 
                                original_image=upload_url,
                                result_image=result_url,
                                result_type='segmentation',
                                message="图像分割完成",
                                success=True)
        else:
            flash(message)
            return redirect(url_for('index'))
    
    elif process_type == 'detection':
        # 目标检测
        result_filename = f"detected_{filename}"
        result_path = os.path.join(app.config['DETECTION_RESULT_FOLDER'], result_filename)
        success, message = apply_detection(filepath, result_path)
        
        # 构建URL
        upload_url = url_for('uploaded_file', folder='uploads', filename=filename)
        result_url = url_for('uploaded_file', folder='detection_results', filename=result_filename)
        
        # 无论成功与否，都渲染结果页面，提供更好的用户体验
        if success:
            # 如果message是列表，计算检测到的数量
            if isinstance(message, list):
                detection_count = len(message)
                detection_message = f"检测到 {detection_count} 个目标"
            else:
                detection_message = message
            
            return render_template('index.html', 
                                original_image=upload_url,
                                result_image=result_url,
                                result_type='detection',
                                message=detection_message,
                                success=True)
        else:
            # 即使失败，也显示结果页面，提供更多信息
            # 判断是否是模型未加载的情况
            is_model_not_loaded = "模型未加载" in message
            return render_template('index.html', 
                                original_image=upload_url,
                                result_image=result_url,
                                result_type='detection',
                                message=message,
                                success=False,
                                is_model_not_loaded=is_model_not_loaded)
    
    else:
        flash('无效的处理类型')
        return redirect(url_for('index'))

# 设置静态文件目录
app.config['STATIC_FOLDER'] = 'app/static'

# 确保Flask能够正确识别静态文件夹
import werkzeug.urls

@app.route('/static/<folder>/<filename>')
def uploaded_file(folder, filename):
    """提供上传的文件和处理结果"""
    # 使用绝对路径确保文件能被正确找到
    directory = os.path.join(app.root_path, 'static', folder)
    if os.path.exists(os.path.join(directory, filename)):
        return send_from_directory(directory, filename)
    else:
        # 返回错误信息，便于调试
        return f"文件不存在: {os.path.join(directory, filename)}", 404