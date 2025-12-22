#!/usr/bin/env python3
"""
生成 Chrome 扩展图标
需要安装 Pillow: pip install Pillow
"""

try:
    from PIL import Image, ImageDraw, ImageFont
    
    def create_icon(size, filename):
        """创建图标"""
        # 创建紫色背景
        img = Image.new('RGB', (size, size), color='#9333ea')
        draw = ImageDraw.Draw(img)
        
        # 绘制简单的 "YT" 文字
        try:
            # 尝试使用系统字体
            font_size = max(8, size // 3)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            # 如果找不到字体，使用默认字体
            font = ImageFont.load_default()
        
        text = "YT"
        # 计算文字位置（居中）
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((size - text_width) // 2, (size - text_height) // 2)
        
        # 绘制白色文字
        draw.text(position, text, fill='white', font=font)
        
        img.save(filename)
        print(f"✓ 已创建 {filename} ({size}x{size})")
    
    # 创建所有尺寸的图标
    sizes = [16, 48, 128]
    for size in sizes:
        create_icon(size, f'icons/icon{size}.png')
    
    print("\n所有图标已生成完成！")
    
except ImportError:
    print("错误: 需要安装 Pillow 库")
    print("请运行: pip install Pillow")
except Exception as e:
    print(f"生成图标时出错: {e}")

