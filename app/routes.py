from flask import render_template, request, redirect, url_for, flash, send_from_directory, send_file, make_response, get_flashed_messages
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import io
from app import app
from app.utils import apply_segmentation, apply_detection, allowed_file

@app.route('/')
def index():
    """主页面"""
    # 获取flash消息
    from flask import get_flashed_messages
    messages = get_flashed_messages(with_categories=True)
    message = None
    if messages:
        message = messages[0][1]  # 获取最新的消息
    return render_template('index.html', system_status='ready', message=message)

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和分析"""
    # 检查请求中是否包含文件部分
    if 'file' not in request.files:
        flash('没有文件部分')
        return render_template('index.html', system_status='error', message='没有文件部分')
    
    file = request.files['file']
    
    # 如果用户没有选择文件
    if file.filename == '':
        flash('未选择文件')
        return render_template('index.html', system_status='error', message='未选择文件')
    
    # 检查文件类型
    if not allowed_file(file.filename):
        flash('不支持的文件类型')
        return render_template('index.html', system_status='error', message='不支持的文件类型')
    
    # 保存上传的文件
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # 获取处理类型
    process_type = request.form.get('process_type', 'segmentation')
    
    # 构建上传URL
    upload_url = url_for('uploaded_file', folder='uploads', filename=filename)
    
    # 根据选择的处理类型执行相应的处理
    if process_type == 'segmentation':
        # 图像分割
        result_filename = f"segmented_{filename}"
        result_path = os.path.join(app.config['SEGMENTATION_RESULT_FOLDER'], result_filename)
        success, message = apply_segmentation(filepath, result_path)
        
        if success:
            # 构建结果URL
            result_url = url_for('uploaded_file', folder='segmentation_results', filename=result_filename)
            
            # 计算缺陷区域数量
            if isinstance(message, dict) and 'defect_count' in message:
                defect_count = message['defect_count']
                status_message = f"检测到 {defect_count} 个缺陷区域"
            else:
                status_message = "图像分割完成"
            
            return render_template('index.html', 
                                original_image=upload_url,
                                result_image=result_url,
                                result_type='segmentation',
                                message=status_message,
                                success=True,
                                system_status='completed')
        else:
            # 分割失败时显示错误状态
            return render_template('index.html', 
                                original_image=upload_url,
                                result_image=None,
                                result_type='segmentation',
                                message=message,
                                success=False,
                                system_status='error')
    
    elif process_type == 'detection':
        # 目标检测
        result_filename = f"detected_{filename}"
        result_path = os.path.join(app.config['DETECTION_RESULT_FOLDER'], result_filename)
        success, message = apply_detection(filepath, result_path)
        
        # 构建结果URL
        result_url = url_for('uploaded_file', folder='detection_results', filename=result_filename)
        
        # 无论成功与否，都渲染结果页面，提供更好的用户体验
        if success:
            # 如果message是列表，计算检测到的数量
            if isinstance(message, list):
                detection_count = len(message)
                detection_message = f"检测到 {detection_count} 个目标"
            elif isinstance(message, dict) and 'defect_count' in message:
                detection_count = message['defect_count']
                detection_message = f"检测到 {detection_count} 个缺陷"
            else:
                detection_message = message
            
            return render_template('index.html', 
                                original_image=upload_url,
                                result_image=result_url,
                                result_type='detection',
                                message=detection_message,
                                success=True,
                                system_status='completed')
        else:
            # 即使失败，也显示结果页面，提供更多信息
            # 判断是否是模型未加载的情况
            is_model_not_loaded = "模型未加载" in str(message)
            return render_template('index.html', 
                                original_image=upload_url,
                                result_image=result_url if os.path.exists(result_path) else None,
                                result_type='detection',
                                message=message,
                                success=False,
                                is_model_not_loaded=is_model_not_loaded,
                                system_status='error')
    
    else:
        # 无效的处理类型，显示错误状态
        return render_template('index.html',
                            original_image=upload_url,
                            result_type=None,
                            message='无效的处理类型',
                            success=False,
                            system_status='error')

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

@app.route('/download/result/<filename>')
def download_result(filename):
    """下载处理结果图片"""
    try:
        # 确定是分割结果还是检测结果
        if filename.startswith('segmented_'):
            folder = 'segmentation_results'
            # 使用ASCII安全的文件名，避免编码问题
            base_name = filename.replace('segmented_', '')
            display_name = f"result_{base_name}"
        elif filename.startswith('detected_'):
            folder = 'detection_results'
            # 使用ASCII安全的文件名，避免编码问题
            base_name = filename.replace('detected_', '')
            display_name = f"result_{base_name}"
        else:
            return "Invalid result filename", 400
        
        directory = os.path.join(app.root_path, 'static', folder)
        file_path = os.path.join(directory, filename)
        
        if not os.path.exists(file_path):
            return "Result file not found", 404
            
        # 使用send_file发送文件，并设置下载文件名
        return send_file(
            file_path, 
            as_attachment=True,
            download_name=display_name,
            mimetype='image/jpeg'
        )
    except Exception as e:
        return f"Download failed: {str(e)}", 500

@app.route('/download/comparison/<upload_folder>/<upload_filename>/<result_folder>/<result_filename>')
def download_comparison(upload_folder, upload_filename, result_folder, result_filename):
    """生成并下载对比图片"""
    try:
        # 获取原始图片和结果图片的完整路径
        upload_path = os.path.join(app.root_path, 'static', upload_folder, upload_filename)
        result_path = os.path.join(app.root_path, 'static', result_folder, result_filename)
        
        # 检查文件是否存在
        if not os.path.exists(upload_path):
            return "原始图片不存在", 404
        if not os.path.exists(result_path):
            return "结果图片不存在", 404
        
        # 读取图片
        original_img = cv2.imread(upload_path)
        result_img = cv2.imread(result_path)
        
        if original_img is None or result_img is None:
            return "无法读取图片", 500
        
        # 调整结果图片尺寸以匹配原始图片的高度（保持比例）
        original_h, original_w = original_img.shape[:2]
        result_h, result_w = result_img.shape[:2]
        
        # 计算新的宽度以保持比例
        new_result_w = int((original_h / result_h) * result_w)
        result_img_resized = cv2.resize(result_img, (new_result_w, original_h))
        
        # 创建并排对比图片
        spacing = 20  # 图片间距
        margin_top = 40  # 顶部边距（用于文字）
        margin_bottom = 20  # 底部边距（用于时间戳）
        
        total_width = original_w + new_result_w + spacing
        total_height = original_h + margin_top + margin_bottom
        
        # 创建空白画布（白色背景）
        comparison_img = 255 * np.ones((total_height, total_width, 3), dtype=np.uint8)
        
        # 放置图片（使用try-except确保不会出现尺寸不匹配错误）
        try:
            # 确保图片放置区域在画布范围内
            original_end_h = min(margin_top + original_h, total_height)
            original_end_w = min(10 + original_w, total_width)
            comparison_img[margin_top:original_end_h, 0:original_end_w] = original_img[:original_end_h-margin_top, :original_end_w-0]
            
            # 放置结果图片
            result_start_w = original_w + spacing
            result_end_h = min(margin_top + original_h, total_height)
            result_end_w = min(result_start_w + new_result_w, total_width)
            comparison_img[margin_top:result_end_h, result_start_w:result_end_w] = result_img_resized[:result_end_h-margin_top, :result_end_w-result_start_w]
        except Exception as e:
            # 如果放置图片失败，尝试调整策略
            # 直接使用两个图片的原始尺寸，确保不超出画布
            comparison_img = 255 * np.ones((max(original_h, result_h) + margin_top + margin_bottom, 
                                          original_w + result_w + spacing, 3), dtype=np.uint8)
            comparison_img[margin_top:margin_top+original_h, 0:original_w] = original_img
            comparison_img[margin_top:margin_top+result_h, original_w+spacing:original_w+spacing+result_w] = result_img
        
        # 添加文字标签 - 使用英文避免编码问题
        cv2.putText(comparison_img, 'Original', (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(comparison_img, 'Result', (original_w + spacing + 10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        # 添加时间戳
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(comparison_img, f'Generated: {timestamp}', (10, total_height-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # 将OpenCV图像转换为字节流
        success, buffer = cv2.imencode('.jpg', comparison_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success:
            return "生成对比图片失败", 500
        
        # 创建IO缓冲区
        buf = io.BytesIO(buffer)
        buf.seek(0)
        
        # 发送文件 - 使用ASCII安全的文件名和正确编码的响应头
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(upload_filename)[0]
        safe_filename = f"comparison_{base_name}_{timestamp}.jpg"
        
        # 使用Werkzeug的secure_filename确保文件名安全
        from werkzeug.utils import secure_filename as werkzeug_secure
        final_filename = werkzeug_secure(safe_filename)
        
        # 创建响应
        response = make_response(send_file(buf, mimetype='image/jpeg'))
        # 使用ASCII安全的方式设置Content-Disposition
        response.headers['Content-Disposition'] = 'attachment; filename="%s"' % final_filename
        return response
        
    except Exception as e:
        return f"Failed to generate comparison image: {str(e)}", 500