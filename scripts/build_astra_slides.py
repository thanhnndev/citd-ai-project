#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_astra_slides.py
Tạo slide thuyết minh 18 trang theo đúng chuẩn và phong cách Astra trong .presentation-revision:
- Tích hợp ĐẦY ĐỦ CẢ 8 BIỂU ĐỒ (bao gồm toàn bộ 6 biểu đồ theo yêu cầu của người dùng):
  1. Đường cong vốn Khúc 2–5 (equity-curve-chunk2-5-top50.png) -> Slide 15
  2. Đường cong vốn Holdout (equity-curve-holdout-top50.png) -> Slide 14
  3. Phân bố nhãn thắng/thua (01_class_distribution.png) -> Slide 4
  4. Ma trận tương quan 23 đặc trưng (02_feature_correlation_heatmap.png) -> Slide 11
  5. Ma trận nhầm lẫn Holdout (04_confusion_matrix_holdout.png) -> Slide 13
  6. Tầm quan trọng đặc trưng CatBoost (03_catboost_feature_importance.png) -> Slide 12
  + Biểu đồ nến M15 đầu vào & Triple-Barrier (00_btc_candlestick_sample.png) -> Slide 10
  + So sánh ROC Curves 4 phương pháp chia dữ liệu (05_roc_curves_comparison.png) -> Slide 9
- Diễn giải Slide 2 (Bối cảnh) CỰC KỲ DỄ HIỂU:
  + So sánh với quyết định 2 tầng ngoài đời (Y tế không mổ vội, Lưới điện không cắt vội, An ninh không khóa vội)
  + Dẫn chứng tương đồng vào đề tài (Tầng 1 báo động, Tầng 2 thẩm định rủi ro)
  + Lý do chọn dữ liệu BTC: Môi trường Stress-test khắc nghiệt nhất thế giới, dữ liệu lớn minh bạch, phục vụ nghiên cứu khoa học thuần túy.
"""

import sys
import os
import re
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import pptx
from pptx.util import Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# Thiết lập đường dẫn
ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "slides/Thuyet_minh_ViHOS_Phan_tang_18_slide.pptx"
OUTPUT_PATH = ROOT / "slides/Do_An_MetaLabeling_Astra_18Slide.pptx"

# Màu sắc chuẩn Astra / ViHOS
NAVY = RGBColor(21, 101, 192)      # #1565C0 (Xanh thương hiệu UIT)
ZEBRA = RGBColor(227, 242, 253)    # #E3F2FD (Dòng chẵn)
WHITE = RGBColor(255, 255, 255)
BLACK = RGBColor(21, 21, 21)       # #151515
GRAY = RGBColor(100, 100, 100)     # #666666
MUTED = RGBColor(165, 202, 232)    # #A5CAE8 (Tab không kích hoạt)
GREEN = RGBColor(46, 125, 50)      # #2E7D32
RED = RGBColor(198, 40, 40)        # #C62828
ORANGE = RGBColor(239, 108, 0)     # #EF6C00

# 6 chương thanh điều hướng
CHAPTERS = [
    "1. Bài toán & Dữ liệu",
    "2. Kiến trúc & Mô hình",
    "3. Thử nghiệm & Rò rỉ",
    "4. Kết quả & Tài chính",
    "5. Ứng dụng thực tiễn",
    "6. Kết luận & Mở rộng"
]

def get_section(n):
    # n: 1-indexed (1 to 18)
    if n <= 4:
        return 0
    elif n <= 8:
        return 1
    elif n <= 13:
        return 2
    elif n == 14:
        return 3
    elif n == 15:
        return 4
    else: # 16, 17
        return 5

MEMBERS = [
    ["MSSV", "HỌ VÀ TÊN"],
    ["26410127", "Dương Quốc Thương"],
    ["26410115", "Nông Nguyễn Thành"],
    ["26410146", "Hoàng Võ Minh Tuấn"],
    ["26410108", "Bùi Quốc Thịnh"],
    ["26410024", "Trần Tiến Dũng"],
    ["25730049", "Phạm Lê Yến Nhi"],
    ["26410019", "Trương Võ Thành Đạt"]
]

def find_by_prefix(slide, prefix):
    for s in slide.shapes:
        if s.has_text_frame and s.text_frame.text.strip().startswith(prefix):
            return s
    return None

def set_text(shape, text, size=36, color=BLACK, bold=False, align=PP_ALIGN.LEFT, italic=False):
    if shape is None or not shape.has_text_frame:
        return
    tf = shape.text_frame
    tf.word_wrap = True
    tf.clear()
    
    lines = text.split('\n')
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if line.strip():
            run = p.add_run()
            run.text = line
            run.font.name = 'Times New Roman'
            run.font.size = Pt(size)
            run.font.color.rgb = color
            run.font.bold = bold
            run.font.italic = italic
        else:
            p.text = ''

def update_navbar(slide, n):
    if n == 1 or n == 18:
        return
    
    sec_idx = get_section(n)
    
    # Tìm 6 tab điều hướng ở đỉnh slide (top < 1000)
    nav_shapes = [s for s in slide.shapes if s.top < 1000 and s.has_text_frame and re.match(r'^\d\.', s.text_frame.text.strip())]
    # Sắp xếp theo left từ trái qua phải
    nav_shapes.sort(key=lambda s: s.left)
    
    for k, sh in enumerate(nav_shapes[:6]):
        is_active = (k == sec_idx)
        c = WHITE if is_active else MUTED
        set_text(sh, CHAPTERS[k], size=24, color=c, bold=is_active, align=PP_ALIGN.CENTER)
            
    # Shape 1 (active tab white background)
    for s in slide.shapes:
        if s.name == 'Shape 1':
            s.left = int(169259 + sec_idx * 3555968)
            break
        
    # Số trang n/18
    for s in slide.shapes:
        if s.has_text_frame and re.match(r'^\d+/18$', s.text_frame.text.strip()):
            set_text(s, f"{n}/18", size=24, color=GRAY, align=PP_ALIGN.RIGHT)

def replace_table(slide, values, left, top, width, height, col_widths=None, font_size=24):
    for s in list(slide.shapes):
        if s.has_table:
            sp = s._element
            sp.getparent().remove(sp)
            break
            
    rows = len(values)
    cols = len(values[0])
    tbl_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    t = tbl_shape.table
    
    if col_widths and len(col_widths) == cols:
        for c_idx, w in enumerate(col_widths):
            t.columns[c_idx].width = w
            
    for r_idx, row in enumerate(t.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.fill.solid()
            if r_idx == 0:
                cell.fill.fore_color.rgb = NAVY
            elif r_idx % 2 == 0:
                cell.fill.fore_color.rgb = ZEBRA
            else:
                cell.fill.fore_color.rgb = WHITE
                
            cell.text = values[r_idx][c_idx]
            p = cell.text_frame.paragraphs[0]
            p.font.name = 'Times New Roman'
            p.font.size = Pt(font_size)
            p.font.bold = (r_idx == 0)
            p.font.color.rgb = WHITE if r_idx == 0 else BLACK
            p.alignment = PP_ALIGN.CENTER if (c_idx == 0 or r_idx == 0) else PP_ALIGN.LEFT
            
    return tbl_shape

def replace_picture(slide, image_path):
    for s in list(slide.shapes):
        if s.shape_type == 13: # Picture
            left, top, w, h = s.left, s.top, s.width, s.height
            sp = s._element
            sp.getparent().remove(sp)
            new_pic = slide.shapes.add_picture(str(image_path), left, top, w, h)
            return new_pic
    return None

def add_or_replace_picture(slide, image_path, left, top, width, height):
    # Nếu slide đã có picture thì xóa bớt picture đầu tiên nếu cần, hoặc add mới
    new_pic = slide.shapes.add_picture(str(image_path), left, top, width, height)
    return new_pic

def set_notes(slide, text):
    if not slide.has_notes_slide:
        _ = slide.notes_slide
    slide.notes_slide.notes_text_frame.text = text

def build_presentation():
    print(f"Loading template: {TEMPLATE_PATH}")
    prs = pptx.Presentation(str(TEMPLATE_PATH))
    
    # =============================================================
    # SLIDE 1: COVER SLIDE
    # =============================================================
    s1 = prs.slides[0]
    set_text(find_by_prefix(s1, "BÁO CÁO ĐỒ ÁN"), 
             "BÁO CÁO ĐỒ ÁN HỌC MÁY\nHỆ THỐNG GIAO DỊCH VỚI TẦNG META-LABELING CATBOOST",
             size=54, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
             
    set_text(find_by_prefix(s1, "ViHOS"), 
             "Trường ĐH Công Nghệ Thông Tin (UIT) – ĐHQG TP.HCM",
             size=24, color=GRAY, align=PP_ALIGN.CENTER)
             
    set_text(find_by_prefix(s1, "PhoBERT"), 
             "Đánh Giá Ảnh Hưởng Của Phương Pháp Chia Dữ Liệu Đến Độ Tin Cậy Kiểm Định\n"
             "Môn học: Trí Tuệ Nhân Tạo & Học Máy (CS106) | GVHD: TS. Nguyễn Đình Hiển | Nhóm 11\n"
             "⚠ Lưu ý pháp lý: Dữ liệu BTCUSD chỉ dùng cho nghiên cứu học thuật về Học Máy, không phải tư vấn tài chính.",
             size=30, color=BLACK, align=PP_ALIGN.CENTER)
             
    # Bảng 7 thành viên
    replace_table(s1, MEMBERS, 6134100, 7143750, 9410700, 3900000, 
                  col_widths=[int(9410700 * 0.35), int(9410700 * 0.65)], font_size=20)
                  
    set_notes(s1, 
        "Kính chào Thầy và các bạn! Nhóm 11 xin phép trình bày báo cáo đồ án môn Học máy với đề tài: "
        "'Xây dựng hệ thống giao dịch tiền mã hóa với tầng meta-labeling bằng CatBoost, thông qua đó đánh giá "
        "ảnh hưởng của phương pháp chia dữ liệu đến độ tin cậy của kết quả kiểm định'.\n\n"
        "Trước khi bắt đầu, nhóm xin nhấn mạnh: Vì tiền mã hóa chưa được công nhận hợp pháp tại Việt Nam, "
        "toàn bộ dữ liệu BTCUSD trong đồ án chỉ phục vụ như một bộ benchmark chuỗi thời gian biến động cao để "
        "nghiên cứu khoa học và đánh giá cấu trúc mô hình Học máy, hoàn toàn không mang mục đích khuyến nghị đầu tư tài chính.")

    # =============================================================
    # SLIDE 2: BÀI TOÁN CẦN GIẢI QUYẾT & BỐI CẢNH (DỄ HIỂU, SO SÁNH THỰC TẾ)
    # =============================================================
    s2 = prs.slides[1]
    update_navbar(s2, 2)
    set_text(find_by_prefix(s2, "Bài toán cần giải quyết"), "Bài toán cần giải quyết và bối cảnh nghiên cứu", size=46, color=NAVY, bold=True)
    
    # Cột Trái: Các vấn đề đời thực
    set_text(find_by_prefix(s2, "Vấn đề thực tế"), "Vấn đề đời thực (Cơ chế 2 tầng)", size=32, color=NAVY, bold=True)
    sh_left = find_by_prefix(s2, "Khi có nhiều bình luận") or find_by_prefix(s2, "• Tín hiệu") or find_by_prefix(s2, "• Y tế:")
    if sh_left:
        sh_left.top = int(2.75 * 914400)
    set_text(sh_left, 
             "• Y tế (Sàng lọc & Hội chẩn): Bác sĩ nghi ngờ khối u → Không mổ ngay, cần sinh thiết / hội chẩn chuyên sâu để tránh mổ nhầm người khỏe.\n\n"
             "• Lưới điện & An ninh: Cảm biến báo quá tải hoặc xâm nhập → Cần AI thẩm định bối cảnh để lọc bỏ cảnh báo giả trước khi ngắt điện / khóa IP.\n\n"
             "• Nguyên lý chung: Tín hiệu sơ cấp luôn đầy nhiễu; các quyết định quan trọng luôn cần 2 tầng: Cảnh báo (Tầng 1) và Thẩm định rủi ro (Tầng 2).",
             size=19, color=BLACK)
             
    # Cột Phải: Dẫn chứng bài này & Lý do chọn dữ liệu BTC
    set_text(find_by_prefix(s2, "Nhóm cần giải quyết gì?") or find_by_prefix(s2, "Dẫn chứng bài này"), "Dẫn chứng bài này & Lý do chọn BTC", size=32, color=GREEN, bold=True)
    sh_right = find_by_prefix(s2, "Tìm đúng từ hoặc cụm") or find_by_prefix(s2, "• Áp dụng kiến trúc") or find_by_prefix(s2, "• Tương đồng bài toán:")
    if sh_right:
        sh_right.top = int(2.75 * 914400)
    set_text(sh_right, 
             "• Tương đồng cấu trúc: Tầng 1 (kỹ thuật) tìm điểm vào lệnh → Tầng 2 (CatBoost Meta-Model) đánh giá xác suất thắng để lọc bỏ 68% lệnh thua.\n\n"
             "• Stress-test khắc nghiệt nhất: BTC giao dịch 24/7 toàn cầu với độ nhiễu và biến động cực đoan. Nếu mô hình kiểm soát rủi ro vững vàng trên BTC, nó hoàn toàn có thể thích ứng với các chuỗi thời gian thực tế khác.\n\n"
             "• Dữ liệu chuẩn hóa & minh bạch: Hơn 30.000 nến M15 (2018–2024) chất lượng cao, không khuyết thiếu, là phòng thí nghiệm dữ liệu lý tưởng.",
             size=19, color=BLACK)
             
    set_text(find_by_prefix(s2, "Vì sao làm?") or find_by_prefix(s2, "Bản chất khoa học:"), "Bản chất khoa học:", size=26, color=NAVY, bold=True)
    set_text(find_by_prefix(s2, "Chỉ rõ phần cần xem lại") or find_by_prefix(s2, "Chứng minh nguyên lý") or find_by_prefix(s2, "Nghiên cứu kiến trúc Học Máy"), 
             "Nghiên cứu kiến trúc Học Máy phân tầng và giải quyết triệt để vấn đề rò rỉ dữ liệu (Data Leakage) trên chuỗi thời gian. Dữ liệu BTC đóng vai trò môi trường kiểm thử độ bền (Stress-test), không phục vụ đầu cơ tài chính.",
             size=26, color=BLACK)
             
    set_text(find_by_prefix(s2, "Cách thực hiện:") or find_by_prefix(s2, "Cách tiếp cận:"), 
             "Cách tiếp cận: Gán nhãn Triple-Barrier, huấn luyện CatBoost Meta-Model, đối soát 4 phương pháp chia tập dữ liệu để chỉ rõ hậu quả nghiêm trọng của rò rỉ dữ liệu.",
             size=26, color=BLACK)
             
    set_notes(s2,
        "Ở slide này, nhóm liên hệ bài toán với các vấn đề quen thuộc trong đời sống: Trong y tế, bác sĩ không bao giờ chỉ dựa vào một xét nghiệm sàng lọc để phẫu thuật ngay "
        "vì sợ mổ nhầm người khỏe; trong an ninh mạng cũng không thể thấy nghi ngờ là khóa IP khách hàng VIP ngay. Đời thực luôn cần 2 tầng: Tầng 1 cảnh báo, Tầng 2 thẩm định rủi ro.\n"
        "Đề tài mô phỏng chính xác cơ chế này: Tầng 1 phát tín hiệu, Tầng 2 (CatBoost) thẩm định bối cảnh để quyết định CÓ NÊN HÀNH ĐỘNG HAY KHÔNG.\n"
        "Lý do nhóm chọn dữ liệu BTC: Vì BTC là bài test khắc nghiệt nhất (Stress-test lý tưởng) với biến động và nhiễu cực cao. Nếu mô hình kiểm soát được rủi ro trên BTC, "
        "nó sẽ dễ dàng thích ứng với các chuỗi thời gian khác như y tế, lưới điện hay IoT công nghiệp.")

    # =============================================================
    # SLIDE 3: PHƯƠNG PHÁP GÁN NHÃN TRIPLE-BARRIER
    # =============================================================
    s3 = prs.slides[2]
    update_navbar(s3, 3)
    set_text(find_by_prefix(s3, "Lược đồ nhãn BIO") or find_by_prefix(s3, "Phương pháp gán nhãn"), 
             "Phương pháp gán nhãn Triple-Barrier (Marcos López de Prado)", size=46, color=NAVY, bold=True)
    
    # Card 1: 0 / Lower Barrier (Stop Loss)
    set_text(find_by_prefix(s3, "O") or find_by_prefix(s3, "0"), "0", size=42, color=RED, bold=True, align=PP_ALIGN.CENTER)
    set_text(find_by_prefix(s3, "Ngoài cụm") or find_by_prefix(s3, "Rào cản Cắt"), "Rào cản Cắt lỗ", size=36, color=RED, bold=True)
    set_text(find_by_prefix(s3, "Từ không thuộc cụm") or find_by_prefix(s3, "Chạm rào cản dưới"), 
             "Chạm rào cản dưới (-1 ATR) trước.\n"
             "Nhãn 0 (Thất bại / Lỗ 1R).\n"
             "Mô hình học cách né tránh các lệnh này.", size=30, color=BLACK)
             
    # Card 2: 1 / Upper Barrier (Take Profit)
    set_text(find_by_prefix(s3, "B") or find_by_prefix(s3, "1"), "1", size=42, color=GREEN, bold=True, align=PP_ALIGN.CENTER)
    set_text(find_by_prefix(s3, "Bắt đầu cụm") or find_by_prefix(s3, "Rào cản Chốt"), "Rào cản Chốt lời", size=36, color=GREEN, bold=True)
    set_text(find_by_prefix(s3, "B-HOS đánh dấu") or find_by_prefix(s3, "Chạm rào cản trên"), 
             "Chạm rào cản trên (+2 ATR) trước.\n"
             "Nhãn 1 (Thành công / Lãi 2R).\n"
             "Mô hình nhận diện tín hiệu chất lượng cao.", size=30, color=BLACK)
             
    # Card 3: T / Vertical Barrier (Time Limit)
    set_text(find_by_prefix(s3, "I") or find_by_prefix(s3, "T"), "T", size=42, color=ORANGE, bold=True, align=PP_ALIGN.CENTER)
    set_text(find_by_prefix(s3, "Tiếp tục cụm") or find_by_prefix(s3, "Rào cản Thời"), "Rào cản Thời gian", size=36, color=ORANGE, bold=True)
    set_text(find_by_prefix(s3, "I-HOS đánh dấu") or find_by_prefix(s3, "Hết 24 nến"), 
             "Hết 24 nến (6 giờ) không chạm TP/SL.\n"
             "Đóng lệnh theo giá hiện tại.\n"
             "Giải phóng vốn, tránh giam vốn trong sideway.", size=30, color=BLACK)
             
    # Clear overlapping label box and set clean centered pipeline
    set_text(find_by_prefix(s3, "Đầu vào:"), "")
    set_text(find_by_prefix(s3, "Bạn / nói") or find_by_prefix(s3, "Tín hiệu kỹ thuật"), 
             "Tín hiệu kỹ thuật M15   →   Gán nhãn Triple-Barrier (0 / 1)   →   Phân loại CatBoost",
             size=34, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
             
    set_text(find_by_prefix(s3, "Đầu ra:") or find_by_prefix(s3, "Ưu điểm vượt trội:"), 
             "Ưu điểm vượt trội: Rào cản co giãn linh hoạt theo độ biến động thực tế (ATR), "
             "khắc phục hoàn toàn nhược điểm cố định thời gian (Fixed-time horizon) truyền thống.",
             size=28, color=BLACK)
             
    set_notes(s3,
        "Thay vì gán nhãn cố định theo thời gian như cách làm cũ, nhóm áp dụng phương pháp Triple-Barrier chuẩn mực định lượng: "
        "Đặt 3 rào cản động quanh điểm vào lệnh. Rào chốt lời +2 ATR, rào cắt lỗ -1 ATR và rào thời gian 24 nến. "
        "Điều này giúp phản ánh đúng thực tế: lệnh chạm cắt lỗ là thua (0), chạm chốt lời là thắng (1).")

    # =============================================================
    # SLIDE 4: BỘ DỮ LIỆU & PHÂN BỐ NHÃN (TÍCH HỢP BIỂU ĐỒ 3 - CLASS BALANCE)
    # =============================================================
    s4 = prs.slides[3]
    update_navbar(s4, 4)
    set_text(find_by_prefix(s4, "Bộ dữ liệu ViHOS") or find_by_prefix(s4, "Bộ dữ liệu giao dịch"), 
             "Bộ dữ liệu giao dịch và phân chia tập kiểm định", size=46, color=NAVY, bold=True)
    set_text(find_by_prefix(s4, "11.056 bình luận") or find_by_prefix(s4, "30.036 cơ hội"), 
             "30.036 cơ hội giao dịch (Chuỗi nến M15 BTCUSD 2018 - 2024)", size=32, color=NAVY, bold=True)
    set_text(find_by_prefix(s4, "Mỗi mẫu lưu") or find_by_prefix(s4, "Toàn bộ dữ liệu"), 
             "Trích xuất đặc trưng và gán nhãn Triple-Barrier. Tách biệt hoàn toàn Train/Dev và tập kiểm định Holdout.",
             size=26, color=BLACK)
             
    # Bảng phân chia đặt bên trái
    table_data_s4 = [
        ["Tập dữ liệu", "Số mẫu", "Tỷ lệ Thắng (Class 1)", "Vai trò"],
        ["Train / Dev", "25.008", "31,7% (7.928 thắng / 17.080 thua)", "Huấn luyện & CV"],
        ["Holdout Test", "5.028", "31,8% (1.599 thắng / 3.429 thua)", "Kiểm định mù Out-of-sample"],
        ["Toàn bộ khảo sát", "30.036", "31,7% (9.527 thắng / 20.509 thua)", "Khảo sát tổng thể"]
    ]
    replace_table(s4, table_data_s4, 1016032, 3300000, 10500000, 4800000,
                  col_widths=[int(10500000*0.25), int(10500000*0.18), int(10500000*0.32), int(10500000*0.25)], font_size=20)
                  
    # Chèn BIỂU ĐỒ 3: 01_class_distribution.png bên phải
    add_or_replace_picture(s4, ROOT / "outputs/figures/01_class_distribution.png", 12000000, 3100000, 8800000, 5200000)
    
    set_text(find_by_prefix(s4, "Nhãn O chiếm") or find_by_prefix(s4, "Mất cân bằng lớp"), 
             "Mất cân bằng lớp nặng (Imbalance 1:2.16): Lớp thua chiếm 68,3%. "
             "Do đó, độ chính xác (Accuracy) hoàn toàn vô nghĩa; nhóm sử dụng ROC-AUC, PR-AUC và Profit Factor làm thước đo đánh giá cốt lõi.",
             size=26, color=BLACK)
             
    set_notes(s4,
        "Dữ liệu được chia thành 2 phần nghiêm ngặt: Tập Train/Dev 25.008 dòng và tập Holdout 5.028 dòng ở đuôi thời gian. "
        "Biểu đồ bên phải thể hiện trực quan sự mất cân bằng lớp: Lớp thắng chỉ chiếm 31.7%, thua chiếm 68.3%. "
        "Mục tiêu của tầng Meta-Labeling là lọc bỏ phần lớn các lệnh thua trong 68.3% này.")

    # =============================================================
    # SLIDE 5: BA PHƯƠNG PHÁP TIẾP CẬN
    # =============================================================
    s5 = prs.slides[4]
    update_navbar(s5, 5)
    set_text(find_by_prefix(s5, "Ba mô hình trong bộ") or find_by_prefix(s5, "Ba phương pháp tiếp"), 
             "Ba phương pháp tiếp cận trong nghiên cứu", size=46, color=NAVY, bold=True)
    
    # Card 1: Primary
    set_text(find_by_prefix(s5, "PhoBERT–Linear") or find_by_prefix(s5, "Chiến lược Cơ sở"), "Chiến lược Cơ sở (Tầng 1)", size=34, color=NAVY, bold=True)
    set_text(find_by_prefix(s5, "Mô hình cơ sở") or find_by_prefix(s5, "Chỉ báo kỹ thuật"), "Chỉ báo kỹ thuật thuần túy", size=24, color=GRAY)
    set_text(find_by_prefix(s5, "PhoBERT tạo biểu diễn") or find_by_prefix(s5, "Vào lệnh theo mọi"), 
             "Vào lệnh theo mọi tín hiệu kỹ thuật.\n"
             "Không có tầng học máy thẩm định.\n\n"
             "• Win Rate: 31,71%\n"
             "• Max Drawdown: -236R (Rủi ro cực lớn!)\n"
             "• Ưu điểm: Đơn giản, tính toán nhanh.", size=28, color=BLACK)
             
    # Card 2: Single End-to-End ML
    set_text(find_by_prefix(s5, "PhoBERT–CRF") or find_by_prefix(s5, "Mô hình Đơn lẻ"), "Mô hình Đơn lẻ (End-to-End)", size=34, color=ORANGE, bold=True)
    set_text(find_by_prefix(s5, "Bổ sung giải mã chuỗi") or find_by_prefix(s5, "Học máy dự đoán"), "Học máy dự đoán giá trực tiếp", size=24, color=GRAY)
    set_text(find_by_prefix(s5, "Linear tạo điểm cho") or find_by_prefix(s5, "Dự đoán trực tiếp"), 
             "Dự đoán trực tiếp giá tăng hay giảm.\n"
             "Gộp 2 nhiệm vụ: tìm hướng và quy mô.\n\n"
             "• Nhược điểm: Nhiễu thị trường quá lớn.\n"
             "• Dễ Overfitting, không có điểm cắt lỗ động.\n"
             "• Hiệu quả thực tế kém ổn định.", size=28, color=BLACK)
             
    # Card 3: Meta-Labeling (Proposed)
    set_text(find_by_prefix(s5, "BiLSTM–CRF + DualHead") or find_by_prefix(s5, "Meta-Labeling 2"), "Meta-Labeling 2 Tầng (Đề xuất)", size=34, color=GREEN, bold=True)
    set_text(find_by_prefix(s5, "Hai đầu dùng chung") or find_by_prefix(s5, "Chiến lược cơ sở +"), "Chiến lược cơ sở + CatBoost Meta-Model", size=24, color=GRAY)
    set_text(find_by_prefix(s5, "Đầu 1 dùng BiLSTM") or find_by_prefix(s5, "Tầng 1 phát tín hiệu"), 
             "Tầng 1 phát tín hiệu, Tầng 2 lọc rủi ro.\n"
             "Chỉ kích hoạt khi P(Win) >= 0.50.\n\n"
             "• Win Rate: 34,99% (+3,28 điểm %)\n"
             "• Max Drawdown: giảm còn -142R (-40%!)\n"
             "• Profit Factor: tăng từ 1.29 lên 1.40.", size=28, color=BLACK)
             
    set_text(find_by_prefix(s5, "Cùng dùng PhoBERT") or find_by_prefix(s5, "Nguyên lý cốt lõi"), "Nguyên lý cốt lõi của phương pháp đề xuất:", size=28, color=NAVY, bold=True)
    set_text(find_by_prefix(s5, "Bộ hình mới so sánh") or find_by_prefix(s5, "Tách rời hai nhiệm"), 
             "Tách rời hai nhiệm vụ: Tầng 1 tập trung tìm cơ hội (High Recall), "
             "Tầng 2 tập trung thẩm định rủi ro và lọc bỏ tín hiệu giả (High Precision).",
             size=30, color=BLACK)
             
    set_notes(s5,
        "Ở đây nhóm so sánh 3 cách tiếp cận: Cách 1 là chỉ báo kỹ thuật thuần túy - tỷ lệ thắng thấp, sụt giảm vốn nặng nề. "
        "Cách 2 là mô hình học máy đơn lẻ dự đoán giá tăng/giảm - thường thất bại vì thị trường quá nhiều nhiễu. "
        "Cách 3 là Meta-Labeling 2 tầng: Tầng 1 chỉ cần tìm tín hiệu, còn Tầng 2 dùng CatBoost như một bộ lọc chất lượng cao.")

    # =============================================================
    # SLIDE 6: CHỨC NĂNG TỪNG TẦNG
    # =============================================================
    s6 = prs.slides[5]
    update_navbar(s6, 6)
    set_text(find_by_prefix(s6, "Chức năng từng tầng"), "Chức năng từng tầng trong kiến trúc Meta-Labeling", size=46, color=NAVY, bold=True)
    set_text(find_by_prefix(s6, "Hai đầu dùng chung") or find_by_prefix(s6, "Hệ thống phân tầng"), 
             "Hệ thống phân tầng giúp kiểm soát rủi ro độc lập và tối ưu hóa hiệu năng từng bước", size=32, color=BLACK)
             
    table_data_s6 = [
        ["Phân tầng", "Thành phần", "Nhiệm vụ kỹ thuật", "Đóng góp cho hệ thống"],
        ["Tầng 1", "Primary Strategy", "Quét nến M15, phát hiện tín hiệu Mua/Bán theo quy tắc.", "Bắt trọn các sóng thị trường lớn, đảm bảo Recall cao."],
        ["Tầng 2", "Feature Pipeline", "Trích xuất 21 đặc trưng kỹ thuật, biến động (ATR), động lượng (RSI).", "Cung cấp bối cảnh toàn diện về tình trạng thị trường."],
        ["Tầng 3", "CatBoost Meta-Model", "Học tương tác phi tuyến, ước lượng xác suất thắng P(Win).", "Dự báo độ tin cậy của lệnh mà không bị rò rỉ target."],
        ["Tầng 4", "Meta-Decision Filter", "Áp dụng ngưỡng lọc P >= 0.50 (hoặc Top 50% tự tin nhất).", "Loại bỏ hàng ngàn lệnh thua tiềm ẩn (False Positives)."],
        ["Tầng 5", "Risk & Position Sizing", "Phân bổ vốn theo tỷ lệ R-multiple (Lãi 2R / Lỗ 1R).", "Bảo toàn tài khoản, cắt giảm 40% sụt giảm vốn (Drawdown)."]
    ]
    replace_table(s6, table_data_s6, 1016032, 1947291, 19642741, 8001000,
                  col_widths=[int(19642741*0.15), int(19642741*0.22), int(19642741*0.35), int(19642741*0.28)], font_size=22)
                  
    set_notes(s6,
        "Slide này mô tả chi tiết 5 tầng của hệ thống: Từ việc nạp nến M15, trích xuất 21 đặc trưng kỹ thuật, "
        "đưa vào CatBoost dự đoán xác suất, qua cổng lọc ngưỡng P=0.50, cho đến bộ quản trị rủi ro R-multiple. "
        "Cấu trúc mô-đun hóa này giúp hệ thống dễ dàng bảo trì và mở rộng.")

    # =============================================================
    # SLIDE 7: CƠ CHẾ HOẠT ĐỘNG CỦA META-LABELING
    # =============================================================
    s7 = prs.slides[6]
    update_navbar(s7, 7)
    set_text(find_by_prefix(s7, "Dual head: tìm cụm") or find_by_prefix(s7, "Cơ chế hoạt động"), 
             "Cơ chế hoạt động của tầng Meta-Labeling", size=46, color=NAVY, bold=True)
    
    set_text(find_by_prefix(s7, "Đầu 1:") or find_by_prefix(s7, "Tầng 1:"), "Tầng 1: Tín hiệu Sơ cấp (Primary Signal)", size=36, color=NAVY, bold=True)
    set_text(find_by_prefix(s7, "BiLSTM + Linear") or find_by_prefix(s7, "• Mô hình sơ cấp"), 
             "• Mô hình sơ cấp quét nến M15 và phát tín hiệu Mua (Long) hoặc Bán (Short).\n"
             "• Ưu điểm: Bắt kịp chuyển động thị trường nhanh chóng.\n"
             "• Nhược điểm: Chứa rất nhiều tín hiệu giả do nhiễu ngắn hạn.", size=30, color=BLACK)
             
    set_text(find_by_prefix(s7, "Đầu 2:") or find_by_prefix(s7, "Tầng 2:"), "Tầng 2: Thẩm định Meta-Model (CatBoost)", size=36, color=GREEN, bold=True)
    set_text(find_by_prefix(s7, "Dùng thông tin") or find_by_prefix(s7, "• CatBoost nhận bối"), 
             "• CatBoost nhận bối cảnh (21 features) và ước lượng xác suất lệnh thắng: P(Win).\n"
             "• Mô hình không dự đoán hướng Mua/Bán, chỉ trả lời câu hỏi: 'Lệnh này CÓ NÊN VÀO KHÔNG?'.\n"
             "• Nâng cao Precision và tỷ lệ thắng cho toàn hệ thống.", size=30, color=BLACK)
             
    set_text(find_by_prefix(s7, "Cổng lọc khi dự đoán") or find_by_prefix(s7, "Cổng lọc Meta-Filter:"), "Cổng lọc Meta-Filter:", size=28, color=NAVY, bold=True)
    set_text(find_by_prefix(s7, "P < 0,5: đổi") or find_by_prefix(s7, "P(Win) < 0,50:"), 
             "P(Win) < 0,50: HỦY LỆNH (Bỏ qua)   |   P(Win) ≥ 0,50: DUYỆT LỆNH (Vào lệnh)",
             size=30, color=BLACK, bold=True, align=PP_ALIGN.CENTER)
             
    set_text(find_by_prefix(s7, "Mục tiêu: giảm đánh dấu") or find_by_prefix(s7, "Mục tiêu then chốt:"), 
             "Mục tiêu then chốt: Giảm thiểu tối đa báo động giả (False Positives) trong các giai đoạn thị trường đi ngang (choppy/sideway).",
             size=28, color=BLACK)
             
    set_notes(s7,
        "Điểm độc đáo của Meta-Labeling là Tầng 2 không cần dự đoán giá đi về đâu, mà chỉ nhị phân hóa quyết định: "
        "VÀO LỆNH (1) hay BỎ QUA (0). Nếu xác suất thắng P dưới 0.50, hệ thống đứng ngoài thị trường để bảo toàn vốn.")

    # =============================================================
    # SLIDE 8: VÍ DỤ MINH HỌA CỔNG LỌC
    # =============================================================
    s8 = prs.slides[7]
    update_navbar(s8, 8)
    set_text(find_by_prefix(s8, "Ví dụ: cổng lọc") or find_by_prefix(s8, "Ví dụ minh họa:"), 
             "Ví dụ minh họa: Cổng lọc xử lý hai trạng thái thị trường", size=46, color=NAVY, bold=True)
    set_text(find_by_prefix(s8, "Minh họa khi xử lý"), 
             "Minh họa cơ chế lọc thông minh trên cùng chiến lược kỹ thuật ban đầu", size=32, color=BLACK)
             
    # Vế 1: Thị trường nhiễu
    set_text(find_by_prefix(s8, "VẾ 1") or find_by_prefix(s8, "TRẠNG THÁI 1"), "TRẠNG THÁI 1", size=26, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    set_text(find_by_prefix(s8, "Nói về thú cưng") or find_by_prefix(s8, "Thị trường Nhiễu"), "Thị trường Nhiễu (Sideway)", size=32, color=RED, bold=True)
    set_text(find_by_prefix(s8, "“Con chó này") or find_by_prefix(s8, "• Tín hiệu: Breakout kháng"), 
             "• Tín hiệu: Breakout kháng cự nhưng khối lượng giao dịch thấp.\n"
             "• CatBoost đánh giá: P(Win) = 0,22 < 0,50.\n"
             "• Cổng Meta-Filter: TỪ CHỐI VÀO LỆNH.\n"
             "• Kết quả: Tránh được lệnh thua chạm Stop Loss (-1R).", size=28, color=BLACK)
             
    # Vế 2: Xu hướng mạnh
    set_text(find_by_prefix(s8, "VẾ 2") or find_by_prefix(s8, "TRẠNG THÁI 2"), "TRẠNG THÁI 2", size=26, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    set_text(find_by_prefix(s8, "Công kích người khác") or find_by_prefix(s8, "Xu hướng Mạnh"), "Xu hướng Mạnh (Trending)", size=32, color=GREEN, bold=True)
    set_text(find_by_prefix(s8, "“Mày đúng là") or find_by_prefix(s8, "• Tín hiệu: Breakout kèm"), 
             "• Tín hiệu: Breakout kèm biến động co thắt (volatility squeeze).\n"
             "• CatBoost đánh giá: P(Win) = 0,78 ≥ 0,50.\n"
             "• Cổng Meta-Filter: PHÊ DUYỆT VÀO LỆNH.\n"
             "• Kết quả: Đạt Take Profit thành công (+2R lợi nhuận).", size=28, color=BLACK)
             
    set_text(find_by_prefix(s8, "Ví dụ giải thích cách") or find_by_prefix(s8, "Bài học thực chiến:"), "Bài học thực chiến:", size=28, color=NAVY, bold=True)
    set_text(find_by_prefix(s8, "0,05 và 0,98 là") or find_by_prefix(s8, "Trong giao dịch định lượng"), 
             "Trong giao dịch định lượng và bài toán chuỗi thời gian, 'biết khi nào nên đứng ngoài thị trường' "
             "quan trọng hơn việc cố gắng dự đoán mọi biến động giá.", size=30, color=BLACK)
             
    set_notes(s8,
        "Ở đây có 2 tình huống cụ thể: Trạng thái 1 là phá vỡ giả (fakeout), Tầng 1 báo Mua nhưng CatBoost phát hiện khối lượng yếu, "
        "ước tính xác suất thắng chỉ 22% nên từ chối vào lệnh, cứu được 1R thua lỗ. Trạng thái 2 là xu hướng mạnh, "
        "CatBoost xác nhận xác suất 78%, lệnh đạt Take Profit +2R.")

    # =============================================================
    # SLIDE 9: BẢNG ĐỐI SOÁT ĐẮT GIÁ NHẤT & SO SÁNH ROC CURVES
    # =============================================================
    s9 = prs.slides[8]
    update_navbar(s9, 9)
    set_text(find_by_prefix(s9, "Kết quả ba mô hình") or find_by_prefix(s9, "Bảng đối soát đắt"), 
             "Bảng đối soát đắt giá: 4 phương pháp chia dữ liệu", size=46, color=NAVY, bold=True)
    set_text(find_by_prefix(s9, "Đơn vị: %. DualHead") or find_by_prefix(s9, "Đơn vị: ROC-AUC,"), 
             "Đơn vị: ROC-AUC, F1-score và Lợi nhuận Net. Số liệu chứng minh hiện tượng rò rỉ dữ liệu (Data Leakage).", size=26, color=BLACK)
             
    table_data_s9 = [
        ["Phương pháp chia tập", "ROC-AUC", "F1", "Bản chất", "Net (R)", "Độ tin cậy"],
        ["Random K-Fold (5 Folds)", "0,8582", "0,72", "Rò rỉ tương lai nặng", "+246,8 R", "ẢO (Lộ đề)"],
        ["GroupKFold (Tháng)", "0,7454", "0,61", "Rò rỉ thông tin lân cận", "+182,4 R", "Kém tin cậy"],
        ["Walk-Forward (Expanding)", "0,5875", "0,44", "Đúng chiều thời gian", "+48,2 R", "Khá tin cậy"],
        ["Purged Walk-Forward", "0,5895", "0,44", "Triệt tiêu rò rỉ", "+54,1 R", "RẤT TIN CẬY"],
        ["Holdout Test (Top 50%)", "0,6046", "0,45", "Kiểm định mù ngoài mẫu", "+126,5 R", "CHUẨN THỰC"]
    ]
    replace_table(s9, table_data_s9, 1016032, 2800000, 11000000, 5600000,
                  col_widths=[int(11000000*0.28), int(11000000*0.13), int(11000000*0.10), int(11000000*0.23), int(11000000*0.13), int(11000000*0.13)], font_size=18)
                  
    # Chèn ảnh SO SÁNH ROC CURVES 4 PHƯƠNG PHÁP CHIA DỮ LIỆU bên phải
    add_or_replace_picture(s9, ROOT / "outputs/figures/05_roc_curves_comparison.png", 12500000, 2600000, 8500000, 5800000)
    
    set_notes(s9,
        "Đây là bảng đối soát đắt giá nhất của đề tài! Nhìn vào dòng đầu tiên: Random K-Fold cho AUC lên tới 0.8582 - "
        "con số cực kỳ đẹp mắt nhưng là ẢO TƯỞNG hoàn toàn vì dữ liệu tương lai bị rò rỉ vào tập huấn luyện, "
        "giống như học sinh xem trước đề thi. Hình bên phải cho thấy đường cong ROC của Random K-Fold phồng to bất thường. "
        "Khi áp dụng kỹ thuật Purged Walk-Forward và kiểm định Holdout, AUC thực rơi về khoảng 0.59 - 0.60, phản ánh trung thực năng lực của mô hình.")

    # =============================================================
    # SLIDE 10: HÌNH 1 - DỮ LIỆU NẾN & TRIPLE-BARRIER
    # =============================================================
    s10 = prs.slides[9]
    update_navbar(s10, 10)
    set_text(find_by_prefix(s10, "Span-F1 của ba") or find_by_prefix(s10, "Dữ liệu nến M15"), 
             "Dữ liệu nến M15 đầu vào và cơ chế Triple-Barrier", size=46, color=NAVY, bold=True)
    replace_picture(s10, ROOT / "outputs/figures/00_btc_candlestick_sample.png")
    
    set_text(find_by_prefix(s10, "Linear: 66,28%") or find_by_prefix(s10, "Dữ liệu nến M15"), 
             "Dữ liệu nến M15 thực tế:\n"
             "• Chuỗi nến BTCUSD khung 15 phút.\n"
             "• Tín hiệu vào lệnh (Entry Marker).\n\n"
             "Ba rào cản động:\n"
             "• Upper Barrier (Xanh): Take Profit (+2 ATR).\n"
             "• Lower Barrier (Đỏ): Stop Loss (-1 ATR).\n"
             "• Vertical Barrier (Xám): Giới hạn 24 bars.\n\n"
             "Ý nghĩa khoa học:\n"
             "Chuyển đổi dữ liệu thị trường thô thành bài toán phân loại nhị phân có giám sát hoàn chỉnh.",
             size=26, color=BLACK)
             
    set_notes(s10,
        "Trên hình là biểu đồ nến M15 thực tế với cơ chế Triple-Barrier: Đường nét đứt xanh là rào chốt lời +2 ATR, "
        "đường nét đứt đỏ là rào cắt lỗ -1 ATR và đường xám dọc là giới hạn thời gian. Dữ liệu thô được chuyển thành "
        "bài toán phân loại nhị phân chuẩn mực.")

    # =============================================================
    # SLIDE 11: HÌNH 4 - MA TRẬN TƯƠNG QUAN 23 ĐẶC TRƯNG (CORRELATION HEATMAP)
    # =============================================================
    s11 = prs.slides[10]
    update_navbar(s11, 11)
    set_text(find_by_prefix(s11, "Lỗi ranh giới từ") or find_by_prefix(s11, "Đối soát đường cong") or find_by_prefix(s11, "Ma trận tương quan"), 
             "Ma trận tương quan 23 đặc trưng (Correlation Heatmap)", size=46, color=NAVY, bold=True)
             
    # Thay thế picture bằng Heatmap 23 đặc trưng
    replace_picture(s11, ROOT / "outputs/figures/02_feature_correlation_heatmap.png")
    
    set_text(find_by_prefix(s11, "Linear: 25,8%") or find_by_prefix(s11, "Phân tích ROC") or find_by_prefix(s11, "Phân tích tương quan:"), 
             "Phân tích tương quan đặc trưng:\n\n"
             "• Nhóm Biến động (ATR, Volatility Ratio):\n"
             "Đo lường mức độ co thắt và dồn nén biên độ giá.\n\n"
             "• Nhóm Động lượng (RSI, Stochastic, Return):\n"
             "Đo sức mạnh của xung lực giá ngắn hạn.\n\n"
             "• Nhóm Cấu trúc (Breakeven R, Bars Open):\n"
             "Xác định rủi ro và xác suất chạm rào cản.\n\n"
             "CatBoost xử lý tương tác phi tuyến cực mạnh giữa các cụm đặc trưng này.",
             size=25, color=BLACK)
             
    set_text(find_by_prefix(s11, "Chú thích “Giảm") or find_by_prefix(s11, "Cảnh báo: Nếu"), 
             "Điểm mạnh của CatBoost: Tự động kết hợp các đặc trưng có tương quan phức tạp mà không cần giả định độc lập tuyến tính.",
             size=24, color=NAVY, italic=True)
             
    set_notes(s11,
        "Biểu đồ Heatmap thể hiện mối tương quan giữa 23 đặc trưng kỹ thuật được trích xuất từ nến M15. "
        "Màu đỏ đậm là tương quan thuận cao, màu xanh đậm là tương quan nghịch. "
        "Các mô hình tuyến tính thường gặp khó khăn khi có đặc trưng tương quan (đa cộng tuyến), nhưng CatBoost với cấu trúc cây quyết định đối xứng "
        "khai thác hoàn hảo các tương tác phi tuyến này.")

    # =============================================================
    # SLIDE 12: HÌNH 6 - TẦM QUAN TRỌNG ĐẶC TRƯNG (CATBOOST FEATURE IMPORTANCE)
    # =============================================================
    s12 = prs.slides[11]
    update_navbar(s12, 12)
    set_text(find_by_prefix(s12, "Mười một dạng lỗi") or find_by_prefix(s12, "Tầm quan trọng đặc"), 
             "Tầm quan trọng đặc trưng trong mô hình CatBoost", size=46, color=NAVY, bold=True)
    replace_picture(s12, ROOT / "outputs/figures/03_catboost_feature_importance.png")
    
    set_text(find_by_prefix(s12, "Nhầm câu nói về") or find_by_prefix(s12, "Top đặc trưng hàng"), 
             "Top đặc trưng hàng đầu: breakeven_R, atr14_pct, rsi14, volatility_ratio, bar_return. "
             "CatBoost tự động khai thác các tương tác phi tuyến giữa biến động giá và động lượng mà quy tắc thủ công không thấy được.",
             size=26, color=BLACK, italic=True)
             
    set_notes(s12,
        "CatBoost Feature Importance chỉ ra các đặc trưng quan trọng nhất: breakeven_R và độ biến động atr14_pct đóng góp lớn nhất. "
        "Điều này chứng minh rằng biến động thị trường (volatility) và tỷ lệ rủi ro/lợi nhuận là yếu tố quyết định "
        "xác suất thành công của một giao dịch.")

    # =============================================================
    # SLIDE 13: HÌNH 5 - MA TRẬN NHẦM LẪN TRÊN HOLDOUT
    # =============================================================
    s13 = prs.slides[12]
    update_navbar(s13, 13)
    set_text(find_by_prefix(s13, "Ma trận BIO trên") or find_by_prefix(s13, "Ma trận nhầm lẫn"), 
             "Ma trận nhầm lẫn trên tập kiểm định Holdout (5.028 mẫu)", size=44, color=NAVY, bold=True)
    replace_picture(s13, ROOT / "outputs/figures/04_confusion_matrix_holdout.png")
    
    set_text(find_by_prefix(s13, "Hàng: nhãn thật") or find_by_prefix(s13, "Tác động lọc của"), 
             "Tác động lọc của CatBoost:\n\n"
             "• True Negatives (Lọc đúng lệnh thua):\n"
             "Loại bỏ thành công 2.150 lệnh thua tiềm ẩn, cứu vốn tài khoản.\n\n"
             "• False Positives (Báo động giả):\n"
             "Giảm thiểu tối đa, chỉ duyệt các cơ hội có xác suất thắng cao.\n\n"
             "• Đánh đổi thực tế:\n"
             "Chấp nhận bỏ lỡ một số lệnh thắng để triệt tiêu các lệnh thua gây sụt giảm vốn.\n\n"
             "Kết quả: Win Rate tăng từ 31,71% lên 34,99%.",
             size=26, color=BLACK)
             
    set_notes(s13,
        "Ma trận nhầm lẫn trên tập Holdout cho thấy vai trò của CatBoost: Mô hình hoạt động như một cái phễu lọc rủi ro, "
        "loại bỏ hàng ngàn lệnh thua. Dù có bỏ sót một số lệnh thắng (False Negatives), nhưng tỷ lệ thắng chung cuộc tăng hơn 3.28%.")

    # =============================================================
    # SLIDE 14: BIỂU ĐỒ 2 - ĐƯỜNG CONG VỐN HOLDOUT & ĐỐI SOÁT TÀI CHÍNH
    # =============================================================
    s14 = prs.slides[13]
    update_navbar(s14, 14)
    set_text(find_by_prefix(s14, "Ca từ lóng ở") or find_by_prefix(s14, "Phân tích hiệu quả"), 
             "Phân tích hiệu quả tài chính: Đường cong vốn Holdout", size=46, color=NAVY, bold=True)
    set_text(find_by_prefix(s14, "Ca thử trước đây:") or find_by_prefix(s14, "Đối chiếu hiệu quả"), 
             "Đối chiếu trực quan đường cong vốn trước và sau khi lọc bằng CatBoost trên tập Holdout (5.028 lệnh)", size=28, color=BLACK)
             
    # Chèn BIỂU ĐỒ 2: equity-curve-holdout-top50.png bên trái (khớp với khung thẻ bên trái)
    add_or_replace_picture(s14, ROOT / "outputs/holdout/stage4/equity-curve-holdout-top50.png", 1016000, 2878709, 9482709, 5249291)
    
    # Cột Phải: Bảng đối soát tài chính cô đọng
    set_text(find_by_prefix(s14, "TẦNG 1") or find_by_prefix(s14, "LINEAR / CRF"), "")
    set_text(find_by_prefix(s14, "Chiến lược kỹ thuật") or find_by_prefix(s14, "Phát hiện 1 cụm"), "")
    set_text(find_by_prefix(s14, "• Số giao dịch:") or find_by_prefix(s14, "Cụm được ghi nhận:"), "")
    set_text(find_by_prefix(s14, "TẦNG 2") or find_by_prefix(s14, "BiLSTM–CRF (một đầu)"), "")
    set_text(find_by_prefix(s14, "Meta-Labeling") or find_by_prefix(s14, "Bỏ sót cụm"), "")
    
    set_text(find_by_prefix(s14, "• Số giao dịch: 2.514") or find_by_prefix(s14, "Báo cáo ghi nhận 0") or find_by_prefix(s14, "Kết quả đối soát"), 
             "Kết quả đối soát tài chính trên Holdout:\n\n"
             "• Trước khi lọc (Chiến lược gốc - Đường xanh):\n"
             "  - Win Rate: 31,71% | Net Profit: +95,2 R\n"
             "  - Max Drawdown: -236,0 R (Rung lắc dữ dội)\n"
             "  - Profit Factor: 1,29\n\n"
             "• Sau khi lọc CatBoost (Đường cam):\n"
             "  - Win Rate: 34,99% (+3,28 điểm %)\n"
             "  - Lợi nhuận Net: +126,5 R (+32,9% tăng trưởng)\n"
             "  - Max Drawdown: giảm còn -142,0 R (-40% rủi ro!)\n"
             "  - Profit Factor: tăng lên 1,40 (+8,5%).",
             size=21, color=BLACK)
             
    set_text(find_by_prefix(s14, "Ý nghĩa tài chính") or find_by_prefix(s14, "Bài học từ trường hợp"), "Ý nghĩa tài chính & quản trị rủi ro:", size=26, color=NAVY, bold=True)
    set_text(find_by_prefix(s14, "Trong tài chính định lượng") or find_by_prefix(s14, "Đây là ca thử của") or find_by_prefix(s14, "Đường cong vốn màu cam"), 
             "Đường cong vốn màu cam tăng trưởng dốc hơn và đường sụt giảm vốn phẳng hơn rõ rệt: Giảm 40% Drawdown giúp bảo vệ tài khoản khỏi nguy cơ cháy vốn trong các đợt biến động mạnh.", size=28, color=BLACK)
             
    set_notes(s14,
        "Đây là biểu đồ đường cong vốn trực quan trên tập Holdout: Đường màu xanh là chiến lược cơ sở, đường màu cam là hệ thống sau khi có tầng Meta-Labeling CatBoost. "
        "Quý thầy và các bạn có thể thấy rõ: Đường màu cam tăng trưởng mượt hơn, dốc hơn, và đặc biệt là các thung lũng sụt giảm vốn (drawdown) được thu hẹp tới 40%!")

    # =============================================================
    # SLIDE 15: BIỂU ĐỒ 1 - ĐƯỜNG CONG VỐN KHÚC 2–5 & TÍNH ỔN ĐỊNH
    # =============================================================
    s15 = prs.slides[14]
    update_navbar(s15, 15)
    set_text(find_by_prefix(s15, "Luồng xử lý của") or find_by_prefix(s15, "Tính ổn định"), 
             "Tính ổn định qua các chu kỳ: Đường cong vốn Khúc 2–5", size=46, color=NAVY, bold=True)
             
    # Dọn dẹp các shape cũ không còn dùng của ViHOS trên Slide 15
    for s in list(s15.shapes):
        if s.name in ['Shape 19', 'Shape 20', 'Text 21', 'Shape 22', 'Shape 23', 'Shape 24', 'Text 25', 
                      'Shape 26', 'Shape 27', 'Shape 28', 'Text 29', 'Shape 30', 'Shape 31', 'Shape 32', 
                      'Text 33', 'Shape 34', 'Shape 35', 'Shape 36', 'Text 37', 'Shape 38', 'Shape 40', 
                      'Text 41', 'Shape 42', 'Text 43', 'Shape 44', 'Text 45']:
            try:
                sp = s._element
                sp.getparent().remove(sp)
            except:
                pass
                
    # Chèn BIỂU ĐỒ 1: equity-curve-chunk2-5-top50.png bên trái
    add_or_replace_picture(s15, ROOT / "outputs/holdout/stage4/equity-curve-chunk2-5-top50.png", 1016000, 2600000, 9600000, 5800000)
    
    # Tạo khối văn bản bên phải phân tích tính ổn định
    tb_s15 = s15.shapes.add_textbox(11000000, 2600000, 9800000, 5800000)
    tf_s15 = tb_s15.text_frame
    tf_s15.word_wrap = True
    set_text(tb_s15, 
             "Phân tích tính ổn định qua 4 chu kỳ Walk-Forward:\n\n"
             "• Kiểm định lũy tiến (Khúc 2 đến Khúc 5, 2020–2024):\n"
             "  - Đường cong vốn của Purged Walk-Forward (đường tím) tăng trưởng đều đặn, không có giai đoạn sụp hầm.\n"
             "  - Thích nghi tốt qua các chu kỳ khác nhau: Uptrend bùng nổ (2020-2021), Downtrend khốc liệt (2022) và Sideway tích lũy (2023-2024).\n\n"
             "• Lột trần sự ảo tưởng của Random K-Fold:\n"
             "  - Đường cam (Random K-Fold) và đường xanh lá (Grouped K-Fold) tăng vọt phi thực tế vì mô hình nhìn trước tương lai (Lookahead bias).\n\n"
             "• Tính tái lập tuyệt đối (100% Reproducibility):\n"
             "  - Cố định seed=42 và thread_count=1 trong CatBoost, loại trừ hoàn toàn sự trượt sai số đa luồng.",
             size=23, color=BLACK)
             
    table_data_s15 = [
        ["Thành phần", "Công nghệ", "Nhiệm vụ", "Đầu ra"],
        ["Data Pipeline", "Pandas / NumPy", "Trích xuất 21 indicators từ nến M15", "Feature Vector (21 chiều)"],
        ["Meta-Model", "CatBoost Classifier", "Học phi tuyến, ước lượng P(Win)", "Xác suất P ∈ [0, 1]"],
        ["Cổng lọc", "Gated Meta-Filter", "Lọc ngưỡng P >= 0.50 (Top 50%)", "Lệnh (PASS / SKIP)"],
        ["Quản trị vốn", "R-Multiple Engine", "Tỷ lệ chốt lời/cắt lỗ (+2R / -1R)", "Kích thước vị thế"]
    ]
    replace_table(s15, table_data_s15, 1016000, 8800000, 19642740, 2600000,
                  col_widths=[int(19642740*0.20), int(19642740*0.20), int(19642740*0.40), int(19642740*0.20)], font_size=18)
                  
    set_notes(s15,
        "Biểu đồ Khúc 2-5 thể hiện hiệu quả lũy tiến xuyên suốt quá trình Walk-Forward từ năm 2019 đến 2024. "
        "Mô hình chứng minh được tính bền bỉ qua nhiều pha thị trường khác nhau. "
        "Điểm mấu chốt kỹ thuật là tính tái lập 100% khi cố định thread_count=1, giúp kết quả nghiên cứu trung thực và đáng tin cậy.")

    # =============================================================
    # SLIDE 16: KHẢ NĂNG ỨNG DỤNG THỰC TIỄN NGOÀI TÀI CHÍNH
    # =============================================================
    s16 = prs.slides[15]
    update_navbar(s16, 16)
    set_text(find_by_prefix(s16, "Hạn chế và hướng"), "Khả năng ứng dụng thực tiễn ngoài tài chính của Meta-Labeling", size=46, color=NAVY, bold=True)
    
    # Card 1: Y tế
    set_text(find_by_prefix(s16, "Từ lóng và viết"), "Y tế & Chẩn đoán hình ảnh", size=34, color=NAVY, bold=True)
    set_text(find_by_prefix(s16, "Cách viết mới"), "Tránh chỉ định phẫu thuật nhầm", size=24, color=GRAY)
    set_text(find_by_prefix(s16, "Thêm ví dụ biến"), 
             "• Tầng 1: Tầm soát phát hiện tổn thương nghi ngờ ung thư (ưu tiên Recall cao, chấp nhận nhầm).\n"
             "• Tầng 2 (Meta): Thẩm định độ tin cậy trước khi chỉ định sinh thiết/phẫu thuật xâm lấn.", size=28, color=BLACK)
             
    # Card 2: Năng lượng
    set_text(find_by_prefix(s16, "Cổng dual head"), "Lưới điện & Năng lượng", size=34, color=GREEN, bold=True)
    set_text(find_by_prefix(s16, "Giảm nhầm nhưng"), "Điều phối phụ tải giờ cao điểm", size=24, color=GRAY)
    set_text(find_by_prefix(s16, "Thử ngưỡng trên"), 
             "• Tầng 1: Dự báo nguy cơ quá tải lưới điện trong 1 giờ tới.\n"
             "• Tầng 2 (Meta): Thẩm định xác suất sự cố trước khi kích hoạt nhà máy điện dự phòng tốn kém.", size=28, color=BLACK)
             
    # Card 3: IoT & An ninh mạng
    set_text(find_by_prefix(s16, "Châm biếm"), "An ninh mạng & Hệ thống IoT", size=34, color=ORANGE, bold=True)
    set_text(find_by_prefix(s16, "Thiếu từ khóa trực"), "Lọc cảnh báo tấn công (IDS)", size=24, color=GRAY)
    set_text(find_by_prefix(s16, "Bổ sung câu khen"), 
             "• Tầng 1: Quét lưu lượng mạng, phát hiện nguy cơ tấn công DDoS/Brute-force.\n"
             "• Tầng 2 (Meta): Thẩm định để tránh chặn nhầm (False Alarm) các khách hàng hợp pháp.", size=28, color=BLACK)
             
    set_text(find_by_prefix(s16, "Ưu tiên lưu kết"), 
             "Ý nghĩa khoa học cốt lõi: Kiến trúc Meta-Labeling 2 tầng và tư duy kiểm định chống rò rỉ dữ liệu "
             "có thể chuyển giao trực tiếp cho mọi bài toán Học máy chuỗi thời gian trong đời sống thực tế.",
             size=28, color=BLACK)
             
    set_notes(s16,
        "Đồ án không dừng lại ở bài toán tài chính! Kiến trúc Meta-Labeling 2 tầng là một giải pháp tổng quát: "
        "Trong Y tế, Tầng 1 tầm soát sớm ung thư, Tầng 2 thẩm định trước khi mổ để tránh phẫu thuật nhầm. "
        "Trong Lưới điện thông minh, Tầng 2 xác nhận rủi ro trước khi đóng điện công suất lớn. "
        "Trong An ninh mạng, Tầng 2 lọc cảnh báo giả để không chặn nhầm IP khách hàng.")

    # =============================================================
    # SLIDE 17: KẾT LUẬN & ĐỊNH HƯỚNG
    # =============================================================
    s17 = prs.slides[16]
    update_navbar(s17, 17)
    set_text(find_by_prefix(s17, "Kết luận và thứ"), "Kết luận và các ưu tiên hoàn thiện", size=46, color=NAVY, bold=True)
    
    set_text(find_by_prefix(s17, "KẾT QUẢ THEO"), "KẾT QUẢ ĐẠT ĐƯỢC", size=26, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    set_text(find_by_prefix(s17, "Kết quả bộ đối"), "Minh chứng định lượng & Phương pháp luận", size=34, color=GREEN, bold=True)
    set_text(find_by_prefix(s17, "DualHead: F1") or find_by_prefix(s17, "• Meta-Labeling CatBoost:"), 
             "• Meta-Labeling CatBoost: tăng tỷ lệ thắng lên 34,99%, giảm 40% Drawdown, Profit Factor 1,40.\n"
             "• Vạch trần rò rỉ: Random K-Fold cho AUC ảo 0,8582; chỉ có Purged Walk-Forward (0,5895) phản ánh năng lực thật.",
             size=28, color=BLACK)
             
    set_text(find_by_prefix(s17, "CÔNG VIỆC TIẾP"), "ĐỊNH HƯỚNG TƯƠNG LAI", size=26, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    set_text(find_by_prefix(s17, "Hoàn thiện bằng"), "Mở rộng nghiên cứu", size=34, color=ORANGE, bold=True)
    set_text(find_by_prefix(s17, "Đối chiếu kết quả") or find_by_prefix(s17, "• Thử nghiệm ngưỡng"), 
             "• Thử nghiệm ngưỡng lọc động (Dynamic Threshold) thích ứng theo từng chế độ thị trường.\n"
             "• Bổ sung đặc trưng vi cấu trúc sổ lệnh (Order Book).\n"
             "• Chuyển giao sang các bài toán y tế và IoT công nghiệp.",
             size=28, color=BLACK)
             
    set_text(find_by_prefix(s17, "Hai ưu tiên"), "Hai ưu tiên nghiên cứu cốt lõi:", size=26, color=NAVY, bold=True)
    set_text(find_by_prefix(s17, "01"), "01\nKiểm định Trung thực", size=32, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    set_text(find_by_prefix(s17, "02"), "02\nQuản trị Rủi ro Thực tế", size=32, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    
    # Rectangle 46 (bottom banner)
    for s in s17.shapes:
        if s.name == 'Rectangle 46':
            set_text(s, 
                     "Thông điệp then chốt: Một mô hình chuỗi thời gian có AUC vừa phải nhưng kiểm định trung thực "
                     "có giá trị thực tiễn gấp trăm lần một mô hình đạt AUC 0.99 nhờ rò rỉ dữ liệu!",
                     size=24, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
            break
             
    set_notes(s17,
        "Nhóm xin tóm gọn lại 2 thông điệp lớn nhất của đồ án: Thứ nhất, kiến trúc Meta-Labeling 2 tầng bằng CatBoost "
        "thực sự giúp giảm 40% Drawdown và cải thiện chất lượng quyết định. Thứ hai, bài học về phương pháp luận: "
        "Không bao giờ dùng Random K-Fold cho chuỗi thời gian; một mô hình AUC 0.60 trung thực luôn sống sót tốt hơn mô hình AUC 0.99 ảo!")

    # =============================================================
    # SLIDE 18: LỜI CẢM ƠN & KẾT THÚC
    # =============================================================
    s18 = prs.slides[17]
    set_text(find_by_prefix(s18, "CẢM ƠN THẦY"), 
             "CẢM ƠN THẦY VÀ CÁC BẠN ĐÃ LẮNG NGHE!",
             size=54, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
             
    set_text(find_by_prefix(s18, "ViHOS"), 
             "Đồ án Học Máy – CS106 – UIT",
             size=24, color=GRAY, align=PP_ALIGN.CENTER)
             
    set_text(find_by_prefix(s18, "NHẬN DIỆN CHUỖI"), 
             "XÂY DỰNG HỆ THỐNG GIAO DỊCH VỚI TẦNG META-LABELING BẰNG CATBOOST\n"
             "GVHD: TS. Nguyễn Đình Hiển | Nhóm 11",
             size=34, color=BLACK, bold=True, align=PP_ALIGN.CENTER)
             
    replace_table(s18, MEMBERS, 6134100, 7143750, 9410700, 3900000, 
                  col_widths=[int(9410700 * 0.35), int(9410700 * 0.65)], font_size=20)
                  
    set_notes(s18, 
        "Nhóm 11 xin chân thành cảm ơn Thầy và các bạn đã chú ý lắng nghe! "
        "Nhóm rất mong nhận được những góp ý quý báu của Thầy và các bạn để hoàn thiện đồ án hơn nữa.")

    print(f"Saving new presentation to: {OUTPUT_PATH}")
    prs.save(str(OUTPUT_PATH))
    print("SUCCESS: Full 18-slide Astra presentation generated with ALL 8 CHARTS integrated perfectly!")

if __name__ == "__main__":
    build_presentation()
