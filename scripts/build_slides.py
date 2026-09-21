"""
build_slides_v3.py — Gọn, ít chữ, nhiều hình, rõ ứng dụng thực tiễn.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ROOT   = Path(__file__).resolve().parents[1]
FIG    = ROOT / "outputs" / "figures"
STAGE4 = ROOT / "outputs" / "holdout" / "stage4"

# ─── Palette ────────────────────────────────────────────
NAVY   = RGBColor(0x15,0x65,0xC0)
WHITE  = RGBColor(0xFF,0xFF,0xFF)
BLACK  = RGBColor(0x15,0x15,0x15)
LBLUE  = RGBColor(0xE3,0xF2,0xFD)
GREEN  = RGBColor(0xE8,0xF5,0xE9)
RED    = RGBColor(0xFF,0xEB,0xEE)
AMBER  = RGBColor(0xFF,0xF3,0xE0)
NAVYBG = RGBColor(0xE8,0xEA,0xF6)
GRAY   = RGBColor(0xDD,0xDD,0xDD)
GD     = RGBColor(0x1B,0x5E,0x20)   # green-dark
RD     = RGBColor(0xB7,0x1C,0x1C)   # red-dark
AD     = RGBColor(0xE6,0x5C,0x00)   # amber-dark
ND     = RGBColor(0x0D,0x47,0xA1)   # navy-dark
GOLD   = RGBColor(0xFF,0xD5,0x80)

# ─── Dimensions ──────────────────────────────────────────
prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)
W = 13.33; H = 7.5

MEMBERS = [
    ("26410127","Dương Quốc Thương"),
    ("26410115","Nông Nguyễn Thành"),
    ("26410146","Hoàng Võ Minh Tuấn"),
    ("26410108","Bùi Quốc Thịnh"),
    ("26410024","Trần Tiến Dũng"),
    ("25730049","Phạm Lê Yến Nhi"),
    ("26410019","Trương Võ Thành Đạt"),
]

TABS = ["① Bối cảnh & Mục tiêu",
        "② Dữ liệu & Kiến trúc",
        "③ Phương pháp chia",
        "④ Kết quả",
        "⑤ Kết luận"]

# ─── Primitive helpers ───────────────────────────────────
def R(slide, l,t,w,h, fc=None, lc=None, lw=None):
    sh = slide.shapes.add_shape(1, Inches(l),Inches(t),Inches(w),Inches(h))
    sh.fill.solid() if fc else sh.fill.background()
    if fc: sh.fill.fore_color.rgb = fc
    sh.line.fill.background() if not lc else None
    if lc: sh.line.color.rgb = lc; sh.line.width = Pt(lw or 1.5)
    return sh

def T(slide, text, l,t,w,h, sz=11, bold=False, color=BLACK,
      align=PP_ALIGN.LEFT, italic=False, font="Arial"):
    tb = slide.shapes.add_textbox(Inches(l),Inches(t),Inches(w),Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = align
    run = p.add_run(); run.text = text
    run.font.size = Pt(sz); run.font.bold = bold
    run.font.italic = italic; run.font.color.rgb = color
    run.font.name = font
    return tb

def ML(slide, lines, l,t,w,h, sz=11, bold=False, color=BLACK,
       align=PP_ALIGN.LEFT, sb=3, font="Arial"):
    tb = slide.shapes.add_textbox(Inches(l),Inches(t),Inches(w),Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    first = True
    for item in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        if isinstance(item, str): txt,b,s,c = item,bold,sz,color
        else:
            txt = item[0]
            b   = item[1] if len(item)>1 else bold
            s   = item[2] if len(item)>2 else sz
            c   = item[3] if len(item)>3 else color
        p.alignment = align; p.space_before = Pt(sb)
        run = p.add_run(); run.text = txt
        run.font.size = Pt(s); run.font.bold = b
        run.font.color.rgb = c; run.font.name = font
    return tb

def P(slide, path, l,t,w,h):
    p = Path(path)
    if p.exists():
        return slide.shapes.add_picture(str(p),Inches(l),Inches(t),Inches(w),Inches(h))
    R(slide,l,t,w,h,fc=GRAY)
    T(slide,f"[{p.name}]",l+.05,t+h/2-.15,w-.1,.3,sz=8,
      color=RGBColor(0x66,0x66,0x66),align=PP_ALIGN.CENTER)

def new():
    sl = prs.slides.add_slide(prs.slide_layouts[0])
    for ph in sl.placeholders: ph._element.getparent().remove(ph._element)
    return sl

def nav(sl, active, num):
    R(sl,0,0,W,.44,fc=NAVY)
    tw = W/len(TABS)
    for i,label in enumerate(TABS):
        bx=i*tw
        if i==active:
            R(sl,bx+.05,.03,tw-.1,.38,fc=WHITE)
            tc=NAVY
        else:
            tc=WHITE
        T(sl,label,bx,.05,tw,.34,sz=8.5,bold=(i==active),color=tc,align=PP_ALIGN.CENTER)
    R(sl,0,.44,W,.04,fc=NAVY)
    T(sl,f"{num}/18",W-.65,.06,.58,.30,sz=8.5,color=WHITE,align=PP_ALIGN.RIGHT)

def hdr(sl, title, sub=""):
    R(sl,.35,.58,W-.7,.52,fc=WHITE)
    R(sl,.35,1.10,W-.7,.04,fc=NAVY)
    T(sl,title,.40,.60,W-1.0,.48,sz=15,bold=True,color=BLACK)
    if sub: T(sl,sub,.40,1.16,W-1.0,.30,sz=9.5,italic=True,
               color=RGBColor(0x55,0x55,0x55))

def card(sl, l,t,w,h, title, lines,
         bg=LBLUE, bd=NAVY, tsz=11.5, bsz=11, bsp=3):
    R(sl,l,t,w,h,fc=bg)
    R(sl,l,t,w,h,lc=bd,lw=1.5)
    T(sl,title,l+.1,t+.09,w-.2,.40,sz=tsz,bold=True,color=bd)
    R(sl,l+.1,t+.49,w-.2,.03,fc=bd)
    ML(sl,lines,l+.1,t+.57,w-.2,h-.65,sz=bsz,color=BLACK,sb=bsp)

# ══════════════════════════════════════════════════════════
# SLIDE 1 — Cover
# ══════════════════════════════════════════════════════════
def s01():
    sl = new()
    R(sl,0,0,W,1.45,fc=NAVY)
    T(sl,"TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN — ĐHQG-HCM",
      .3,.07,W-.6,.38,sz=11.5,bold=True,color=WHITE,align=PP_ALIGN.CENTER)
    T(sl,"KHOA KHOA HỌC MÁY TÍNH  |  MÔN: TRÍ TUỆ NHÂN TẠO (CS106)  |  GVHD: TS. NGUYỄN ĐÌNH HIỂN",
      .3,.46,W-.6,.32,sz=9.5,color=RGBColor(0xBB,0xD6,0xF8),align=PP_ALIGN.CENTER)
    T(sl,"NHÓM 11  —  Đề tài số 10",
      .3,.80,W-.6,.30,sz=10,bold=True,color=GOLD,align=PP_ALIGN.CENTER)
    T(sl,"⚠️  Dữ liệu BTCUSD được dùng thuần tuý cho mục đích NGHIÊN CỨU Học Máy (không khuyến nghị giao dịch)",
      .3,1.14,W-.6,.28,sz=8.5,color=RGBColor(0xFF,0xCC,0x80),align=PP_ALIGN.CENTER)

    # Title
    R(sl,.3,1.57,W-.6,2.10,fc=RGBColor(0xF5,0xF9,0xFF))
    T(sl,"BÁO CÁO ĐỒ ÁN",.35,1.62,W-.7,.34,sz=11,bold=True,
      color=NAVY,align=PP_ALIGN.CENTER)
    T(sl,"XÂY DỰNG HỆ THỐNG LỌC TÍN HIỆU VỚI TẦNG META-LABELING (CATBOOST)\n"
        "& ĐÁNH GIÁ ẢNH HƯỞNG CỦA PHƯƠNG PHÁP CHIA DỮ LIỆU",
      .35,1.97,W-.7,.92,sz=19,bold=True,color=ND,align=PP_ALIGN.CENTER)
    T(sl,"Minh hoạ trên chuỗi thời gian BTCUSD M15  —  Mục đích: Nghiên cứu học thuật thuần tuý",
      .35,2.90,W-.7,.36,sz=11,italic=True,
      color=RGBColor(0x44,0x44,0x44),align=PP_ALIGN.CENTER)

    R(sl,.3,3.55,W-.6,.04,fc=NAVY)

    # Members — 2 cols (4+3)
    T(sl,"DANH SÁCH NHÓM",.35,3.62,3,.30,sz=10,bold=True,color=NAVY)
    for ci,(group,ox) in enumerate([(MEMBERS[:4],.35),(MEMBERS[4:],6.9)]):
        oy=3.95
        R(sl,ox,oy,1.55,.30,fc=NAVY)
        T(sl,"MSSV",ox,oy+.01,1.55,.28,sz=9,bold=True,color=WHITE,align=PP_ALIGN.CENTER)
        R(sl,ox+1.60,oy,4.5,.30,fc=NAVY)
        T(sl,"HỌ VÀ TÊN",ox+1.60,oy+.01,4.5,.28,sz=9,bold=True,color=WHITE,align=PP_ALIGN.CENTER)
        for ri,(mssv,name) in enumerate(group):
            ry=oy+.32+ri*.42
            bg=RGBColor(0xF5,0xF5,0xF5) if ri%2==0 else WHITE
            R(sl,ox,ry,1.55,.38,fc=bg)
            T(sl,mssv,ox,ry+.04,1.55,.30,sz=9.5,color=BLACK,align=PP_ALIGN.CENTER)
            R(sl,ox+1.60,ry,4.5,.38,fc=bg)
            T(sl,name,ox+1.65,ry+.04,4.40,.30,sz=9.5,bold=True,color=BLACK)

# ══════════════════════════════════════════════════════════
# SLIDE 2 — Vấn đề & Mục tiêu (gọn)
# ══════════════════════════════════════════════════════════
def s02():
    sl = new()
    nav(sl,0,2)
    hdr(sl,"Vấn đề & Mục tiêu Đề tài",
        sub="Dữ liệu BTCUSD = bộ dữ liệu chuẩn (benchmark) để kiểm chứng phương pháp ML — không phải hướng dẫn giao dịch")

    # Problem box
    card(sl,.3,1.22,5.9,2.88,"⚠️  Vấn đề thực tế trong nghiên cứu AI",
         ["AI/ML thường cho điểm kiểm định đẹp (AUC cao) trong phòng lab.",
          "Khi triển khai thực tế → kết quả tệ hơn rất nhiều.",
          "",
          "Nguyên nhân: Dữ liệu bị rò rỉ (Data Leakage) do cách chia sai.",
          "→ Mô hình vô tình 'nhìn trộm đáp án tương lai' trong lúc học."],
         bg=RED,bd=RD)

    card(sl,.3,4.18,5.9,2.89,"🎯  Câu hỏi nghiên cứu",
         ["Phương pháp phân chia dữ liệu ảnh hưởng bao nhiêu đến kết quả?",
          "Cách nào phản ánh đúng hiệu năng thực tế của mô hình?",
          "AI 2 tầng (Meta-Labeling) có cải thiện bộ lọc tín hiệu không?"],
         bg=LBLUE,bd=NAVY)

    # Two goals
    card(sl,6.5,1.22,6.53,2.88,"⚙️  Mục tiêu 1 — Kỹ thuật",
         ["Xây dựng pipeline AI phân loại tín hiệu 2 tầng.",
          "Tầng 1: Chiến lược phát tín hiệu cơ sở.",
          "Tầng 2: CatBoost thẩm định — chỉ giữ tín hiệu xác suất thắng ≥ 50%.",
          "→ Đo lường cải thiện: Win Rate, Drawdown, Profit Factor."],
         bg=GREEN,bd=GD)

    card(sl,6.5,4.18,6.53,2.89,"🔬  Mục tiêu 2 — Phương pháp luận",
         ["So sánh 4 cách chia dữ liệu (từ lỏng lẻo → nghiêm ngặt).",
          "Kiểm chứng bằng tập Holdout niêm phong (5.028 mẫu).",
          "Xác định cách chia nào đáng tin cậy nhất cho chuỗi thời gian.",
          "→ Kết quả áp dụng được cho bất kỳ bài toán ML chuỗi thời gian nào."],
         bg=NAVYBG,bd=ND)

# ══════════════════════════════════════════════════════════
# SLIDE 3 — Ứng dụng thực tiễn NGOÀI tài chính
# ══════════════════════════════════════════════════════════
def s03():
    sl = new()
    nav(sl,0,3)
    hdr(sl,"Ứng dụng Thực tiễn — Kiến thức từ đề tài dùng ở đâu ngoài tài chính?",
        sub="Kiến trúc Meta-Labeling & Phương pháp chia chuẩn áp dụng cho mọi bài toán chuỗi thời gian")

    # Big disclaimer first
    R(sl,.3,1.22,W-.6,.58,fc=AMBER)
    R(sl,.3,1.22,W-.6,.58,lc=AD,lw=2)
    T(sl,"⚠️  Crypto không hợp pháp tại Việt Nam — Nhóm KHÔNG hướng dẫn giao dịch tiền mã hoá. "
        "Đề tài nghiên cứu PHƯƠNG PHÁP ML thuần tuý, minh hoạ trên dữ liệu BTC như một bộ chuỗi thời gian công khai.",
      .4,1.26,W-.8,.50,sz=11,bold=True,color=AD)

    # 6 application cards
    apps = [
        ("🏥  Y tế & Sức khoẻ",
         "Phát hiện nhịp tim bất thường (ECG)\nDự đoán nguy cơ cơn động kinh (EEG)\nCảnh báo suy tim sớm",
         LBLUE,NAVY),
        ("⚡  Năng lượng & Điện lực",
         "Dự báo tiêu thụ điện năng\nPhát hiện rò rỉ điện/nước\nTối ưu lịch sạc pin lưu trữ",
         GREEN,GD),
        ("🏭  Sản xuất & IoT",
         "Phát hiện lỗi máy móc sớm\nDự đoán bảo trì thiết bị\nKiểm soát chất lượng sản phẩm",
         AMBER,AD),
        ("🌐  Mạng & An ninh mạng",
         "Phát hiện xâm nhập bất thường\nPhân loại lưu lượng mạng độc hại\nGiám sát hệ thống thời gian thực",
         RED,RD),
        ("🌦️  Khí tượng & Môi trường",
         "Dự báo lũ lụt, hạn hán\nPhân tích chất lượng không khí\nDự báo năng lượng mặt trời",
         NAVYBG,ND),
        ("📦  Chuỗi cung ứng & Logistic",
         "Dự báo nhu cầu hàng hoá\nPhát hiện gian lận giao dịch\nTối ưu lịch trình vận chuyển",
         RGBColor(0xF3,0xE5,0xF5),RGBColor(0x6A,0x1B,0x9A)),
    ]
    cols = [(0.30,1.96),(4.60,1.96),(8.90,1.96),
            (0.30,4.48),(4.60,4.48),(8.90,4.48)]
    for (l,t),(title,body,bg,bd) in zip(cols,apps):
        R(sl,l,t,4.0,2.35,fc=bg)
        R(sl,l,t,4.0,2.35,lc=bd,lw=1.5)
        T(sl,title,l+.1,t+.09,3.8,.38,sz=12,bold=True,color=bd)
        R(sl,l+.1,t+.47,3.8,.03,fc=bd)
        T(sl,body,l+.1,t+.55,3.8,1.70,sz=11,color=BLACK)

    R(sl,.3,6.94,W-.6,.50,fc=NAVY)
    T(sl,"📌  Điểm chung: Bất kỳ bài toán nào có DỮ LIỆU TUẦN TỰ THEO THỜI GIAN đều cần phương pháp chia dữ liệu nghiêm ngặt như Purged Walk-Forward. "
        "Đây là bài học cốt lõi của đề tài, áp dụng trực tiếp vào mọi lĩnh vực trên.",
      .4,6.97,W-.8,.43,sz=10.5,bold=True,color=WHITE)

# ══════════════════════════════════════════════════════════
# SLIDE 4 — Dữ liệu đầu vào (CÓ HÌNH NẾN)
# ══════════════════════════════════════════════════════════
def s04():
    sl = new()
    nav(sl,1,4)
    hdr(sl,"Dữ liệu Đầu vào — Chuỗi Nến M15 BTCUSD & Phân bố Nhãn",
        sub="199,968 nến M15 · 2018–2026 · Trích xuất thành 30,036 mẫu giao dịch có nhãn")

    # Candlestick chart (big, left+center)
    P(sl, FIG/"00_btc_candlestick_sample.png", .3,1.22,8.15,4.42)

    # Stats on right
    stat_items = [
        ("📊  Thông số bộ dữ liệu", True, 12, NAVY),
        ("", False,4,BLACK),
        ("Tổng nến thô:", True,10.5,BLACK), ("199,968 nến M15",False,10.5,BLACK),
        ("Giai đoạn:", True,10.5,BLACK),    ("2018 – 2026",False,10.5,BLACK),
        ("Mẫu giao dịch:", True,10.5,BLACK),("30,036 lệnh",False,10.5,BLACK),
        ("", False,4,BLACK),
        ("🏷️  Phân chia tập:", True,12,NAVY),
        ("", False,4,BLACK),
        ("Train/Dev", True,10.5,BLACK),     ("25,008 mẫu",False,10.5,BLACK),
        ("Holdout", True,10.5,BLACK),       ("5,028 mẫu  ← niêm phong",False,10.5,BLACK),
        ("", False,4,BLACK),
        ("⚖️  Tỷ lệ nhãn:", True,12,NAVY),
        ("", False,4,BLACK),
        ("Thua/Hết giờ (0):", True,10.5,RD),("~72–74%",False,10.5,RD),
        ("Thắng (1):", True,10.5,GD),       ("~26–28%",False,10.5,GD),
    ]
    # Two-column mini stats
    ML(sl,[
        ("📊  Bộ dữ liệu",True,12,NAVY),
        ("199,968 nến M15  (2018–2026)",False,11,BLACK),
        ("30,036 mẫu giao dịch",False,11,BLACK),
        ("Train: 25,008  |  Holdout: 5,028",False,11,BLACK),
        ("",False,4,BLACK),
        ("⚖️  Nhãn",True,12,NAVY),
        ("Thua / Hết giờ (0): 72–74%",False,11,RD),
        ("Thắng (1): 26–28%",False,11,GD),
        ("Xử lý: auto_class_weights='Balanced'",False,10.5,BLACK),
    ], 8.58,1.22,4.55,4.42,sb=4)

    # Bottom: class distribution chart
    P(sl, FIG/"01_class_distribution.png", .3,5.74,12.73,1.72)

# ══════════════════════════════════════════════════════════
# SLIDE 5 — Kiến trúc 2 tầng (súc tích)
# ══════════════════════════════════════════════════════════
def s05():
    sl = new()
    nav(sl,1,5)
    hdr(sl,"Kiến trúc Hệ thống 2 Tầng — Meta-Labeling",
        sub="Tầng 1 phát tín hiệu → Tầng 2 AI thẩm định → Chỉ vào lệnh khi xác suất thắng ≥ 50%")

    # Flow
    flow = [
        ("📊\nDữ liệu\nM15","199,968 nến\n2018–2026",LBLUE,NAVY),
        ("📐\nTầng 1\nChiến lược\nPyramid","Phát tín hiệu\nMua / Bán",GREEN,GD),
        ("🤖\nTầng 2\nCatBoost\nMeta-Labeling","Xác suất thắng\np̂ = 0–100%",NAVYBG,ND),
        ("✅  p̂≥50%\nVÀO LỆNH\n❌  p̂<50%\nBỎ QUA","Lọc ~50%\nlệnh xấu",GREEN,GD),
    ]
    bw,bh=2.75,1.70; gap=.42; sx=(W-4*bw-3*gap)/2
    for i,(title,sub,bg,bd) in enumerate(flow):
        bx=sx+i*(bw+gap)
        R(sl,bx,1.55,bw,bh,fc=bg); R(sl,bx,1.55,bw,bh,lc=bd,lw=2)
        T(sl,title,bx+.08,1.60,bw-.16,.95,sz=11,bold=True,color=bd,align=PP_ALIGN.CENTER)
        T(sl,sub,bx+.08,2.52,bw-.16,.62,sz=10,color=BLACK,align=PP_ALIGN.CENTER)
        if i<3: T(sl,"→",bx+bw+.03,1.92,gap-.05,.52,sz=22,bold=True,color=NAVY,align=PP_ALIGN.CENTER)

    # Comparison before/after
    R(sl,.3,3.40,W-.6,.04,fc=GRAY)

    before_after = [
        (.3,3.50,6.0,2.88,"❌  TRƯỚC — Không lọc ML",
         ["Tất cả tín hiệu đều vào lệnh.",
          "Win Rate: 31.71%",
          "Max Drawdown: 236 R",
          "Profit Factor: 1.29",
          "→ Nhiều lệnh nhiễu, rủi ro cao."],
         RED,RD),
        (6.65,3.50,6.53,2.88,"✅  SAU — Có lọc CatBoost (Top 50%)",
         ["Chỉ lệnh AI xác nhận mới được thực thi.",
          "Win Rate: 34.99%  (+3.28%)",
          "Max Drawdown: 142 R  (−40% ✅)",
          "Profit Factor: 1.40  (+8.5%)",
          "→ Ít lệnh hơn, an toàn hơn, hiệu quả hơn."],
         GREEN,GD),
    ]
    for args in before_after:
        l,t,w,h,title,lines,bg,bd = args
        card(sl,l,t,w,h,title,lines,bg=bg,bd=bd)

    R(sl,.3,6.48,W-.6,.46,fc=NAVY)
    T(sl,"📌  Nguyên lý Meta-Labeling (GS. Marcos López de Prado, 2018): Thêm một lớp AI thứ 2 để thẩm định chất lượng tín hiệu. "
        "Áp dụng được cho bất kỳ bài toán phân loại nhị phân nào.",
      .4,6.51,W-.8,.40,sz=10.5,bold=True,color=WHITE)

# ══════════════════════════════════════════════════════════
# SLIDE 6 — Triple-Barrier & 23 đặc trưng
# ══════════════════════════════════════════════════════════
def s06():
    sl = new()
    nav(sl,1,6)
    hdr(sl,"Gán nhãn Triple-Barrier & 23 Đặc trưng Đầu vào")

    # Left: triple barrier mini-diagram
    for bx,t_,bd_,bg_,label in [
        (.3,1.22,GD,GREEN,"🟢  Rào Chốt lời\nEntry + k×ATR\n→ Nhãn = 1 (Thắng)"),
        (4.2,1.22,RD,RED, "🔴  Rào Cắt lỗ\nEntry − m×ATR\n→ Nhãn = 0 (Thua)"),
        (8.1,1.22,AD,AMBER,"⏱️  Rào Thời gian\nTối đa 50 nến M15\n→ Nhãn = 0 (Hết giờ)"),
    ]:
        R(sl,bx,t_,3.6,1.80,fc=bg_); R(sl,bx,t_,3.6,1.80,lc=bd_,lw=1.5)
        T(sl,label,bx+.1,t_+.15,3.4,1.55,sz=11.5,bold=True,color=bd_,align=PP_ALIGN.CENTER)

    # Feature importance chart (main)
    P(sl, FIG/"03_catboost_feature_importance.png", .3,3.12,6.05,4.0)

    # Feature groups right
    grps = [
        ("📈 Biến động (7)","breakeven_R, atr14_pct, vol20, vol200 ...",AMBER,AD),
        ("📐 Xu hướng (7)","dist_ema20/50/200_atr, ret20_atr, rsi14 ...",LBLUE,NAVY),
        ("⚡ Động lượng (5)","mom14, mom_diff, vwap_dist_atr ...",GREEN,GD),
        ("🕐 Thời gian (4)","hour, dow, gap_open_atr, vol_ratio",NAVYBG,ND),
    ]
    gy=3.12
    for title,feats,bg,bd in grps:
        R(sl,6.5,gy,6.53,.92,fc=bg); R(sl,6.5,gy,6.53,.92,lc=bd,lw=1)
        T(sl,title,6.6,gy+.07,6.3,.35,sz=11,bold=True,color=bd)
        T(sl,feats,6.6,gy+.44,6.3,.42,sz=9.5,color=BLACK)
        gy+=1.00

# ══════════════════════════════════════════════════════════
# SLIDE 7 — Data Leakage (ví dụ thi cử)
# ══════════════════════════════════════════════════════════
def s07():
    sl = new()
    nav(sl,2,7)
    hdr(sl,"Rò rỉ Dữ liệu (Data Leakage) — 'Bí kíp' làm giả điểm AI",
        sub="Lý do chính tại sao đề tài cần nghiên cứu phương pháp phân chia dữ liệu")

    # Analogy
    R(sl,.3,1.22,W-.6,.80,fc=AMBER)
    T(sl,"🎓  Ví dụ: Bạn đang ôn thi với 100 câu hỏi từ 2020–2026",
      .4,1.26,W-.8,.32,sz=12.5,bold=True,color=AD)
    T(sl,"Cách sai (Random): Xáo ngẫu nhiên 100 câu → Ôn 80, thi thử 20. Vô tình thấy câu 2026 khi ôn → Điểm thi thử: 9.5/10 (ảo!)\n"
        "Cách đúng (Walk-Forward): Ôn bài 2020–2024, thi thử bài 2025–2026 hoàn toàn mới → Điểm: 6.5/10 (thật!)",
      .4,1.57,W-.8,.40,sz=11,color=BLACK)

    # Two leakage sources
    card(sl,.3,2.12,5.9,3.60,"⏰  Rò rỉ Thời gian (Temporal Leakage)",
         ["Mỗi lệnh cần 50 nến để biết kết quả (Thắng/Thua).",
          "→ Hai lệnh gần nhau có thời gian xác định nhãn CHỒNG LẤN.",
          "",
          "Nếu chia ngẫu nhiên:",
          "  Train biết trước kết quả của Test → Điểm ảo!",
          "",
          "Tác động: AUC tăng giả từ 0.59 → 0.86 (+47%)"],
         bg=RED,bd=RD)

    card(sl,6.65,2.12,6.53,3.60,"🔗  Rò rỉ Nhóm Lệnh (Cluster Leakage)",
         ["Chiến lược mở 1 lệnh gốc + 3 lệnh con (cùng origin_bar).",
          "→ 4 lệnh này tương quan rất cao với nhau.",
          "",
          "Nếu chia ngẫu nhiên:",
          "  Lệnh gốc trong Train, lệnh con trong Test.",
          "  Mô hình kiểm định trên 'anh em' của mình → Gian lận vô tình!"],
         bg=AMBER,bd=AD)

    R(sl,.3,5.82,W-.6,.50,fc=NAVY)
    T(sl,"📊  Kết quả đo được:  "
        "Random K-Fold (rò rỉ nặng): AUC = 0.86   ←→   "
        "Purged Walk-Forward (không rò rỉ): AUC = 0.59   ←→   "
        "Holdout thực tế: AUC = 0.60  ≈ Cách 3 ✅",
      .4,5.87,W-.8,.40,sz=11.5,bold=True,color=WHITE)

# ══════════════════════════════════════════════════════════
# SLIDE 8 — 4 phương pháp chia (compact)
# ══════════════════════════════════════════════════════════
def s08():
    sl = new()
    nav(sl,2,8)
    hdr(sl,"4 Phương pháp Phân chia Dữ liệu — Từ 'Ẩu' đến 'Chuẩn'")

    methods = [
        ("Cách 1\nRandom K-Fold",
         "🚨 Rò rỉ CỰC NẶNG",
         ["Xáo ngẫu nhiên từng dòng.",
          "Vi phạm thứ tự thời gian.",
          "Vi phạm tính toàn vẹn nhóm."],
         RED,RD,"❌ AUC = 0.86\n(Ảo giác!)"),
        ("Cách 1b\nGrouped K-Fold",
         "⚠️  Rò rỉ Vừa",
         ["Giữ nguyên nhóm lệnh.",
          "Vẫn vi phạm trật tự thời gian.",
          "Còn bị thổi phồng nhiều."],
         AMBER,AD,"⚠️ AUC = 0.74\n(Còn cao)"),
        ("Cách 2\nWalk-Forward",
         "✅  Đúng thứ tự TG",
         ["Chia tuần tự: Cũ → Mới.",
          "Đúng hướng thời gian.",
          "Còn chồng lấn ở biên fold."],
         GREEN,GD,"📉 AUC = 0.59\n(Gần thật)"),
        ("Cách 3\nPurged Walk-Forward",
         "🛡️  CHUẨN NHẤT",
         ["Walk-Forward + Purging biên.",
          "Embargo 50 nến cách ly.",
          "Hai tập hoàn toàn độc lập."],
         NAVYBG,ND,"🎯 AUC = 0.59\n(Chính xác!)"),
    ]

    mw=2.88; gap=.24
    for i,(title,verdict,items,bg,bd,score) in enumerate(methods):
        bx=.25+i*(mw+gap)
        R(sl,bx,1.22,mw,5.28,fc=bg); R(sl,bx,1.22,mw,5.28,lc=bd,lw=2.5)
        T(sl,title,bx+.08,1.28,mw-.16,.60,sz=13,bold=True,color=bd,align=PP_ALIGN.CENTER)
        R(sl,bx+.08,1.88,mw-.16,.03,fc=bd)
        T(sl,verdict,bx+.08,1.95,mw-.16,.44,sz=10.5,bold=True,color=bd,align=PP_ALIGN.CENTER)
        ML(sl,items,bx+.10,2.45,mw-.20,2.0,sz=11,color=BLACK,sb=4)
        R(sl,bx+.08,4.74,mw-.16,.68,fc=bd)
        T(sl,score,bx+.08,4.77,mw-.16,.62,sz=11.5,bold=True,color=WHITE,align=PP_ALIGN.CENTER)

    # Timeline diagram
    R(sl,.3,6.60,W-.6,.04,fc=GRAY)
    R(sl,.3,6.70,3.5,.56,fc=GREEN)
    T(sl,"TRAIN",   .35,6.73,3.4,.22,sz=9.5,bold=True,color=GD,align=PP_ALIGN.CENTER)
    R(sl,3.88,6.70,.7,.56,fc=RED)
    T(sl,"PURGE",   3.88,6.73,.7,.22,sz=8,bold=True,color=RD,align=PP_ALIGN.CENTER)
    R(sl,4.66,6.70,2.8,.56,fc=NAVYBG)
    T(sl,"TEST",    4.71,6.73,2.7,.22,sz=9.5,bold=True,color=ND,align=PP_ALIGN.CENTER)
    R(sl,7.54,6.70,1.1,.56,fc=AMBER)
    T(sl,"EMBARGO", 7.54,6.73,1.1,.22,sz=8,bold=True,color=AD,align=PP_ALIGN.CENTER)
    R(sl,8.72,6.70,4.6,.56,fc=GREEN)
    T(sl,"TRAIN Fold 2",8.77,6.73,4.5,.22,sz=9.5,bold=True,color=GD,align=PP_ALIGN.CENTER)
    T(sl,"── Trục thời gian (tuyến tính, không đảo ngược) ──────────────────────────────────────────────────────────────────────────────────▶",
      .3,7.30,W-.6,.20,sz=8.5,italic=True,color=RGBColor(0x77,0x77,0x77))

# ══════════════════════════════════════════════════════════
# SLIDE 9 — CatBoost
# ══════════════════════════════════════════════════════════
def s09():
    sl = new()
    nav(sl,2,9)
    hdr(sl,"Mô hình CatBoost — Tại sao phù hợp nhất?")

    # Left config + why
    card(sl,.3,1.22,5.5,3.40,"🤖  CatBoost — Gradient Boosting của Yandex",
         ["Oblivious Trees (cây đối xứng) → Chống overfitting tốt.",
          "Không cần chuẩn hoá đặc trưng.",
          "Built-in regularization (l2_leaf_reg).",
          "thread_count = 1 → Tái lập 100%.",
          "",
          "Tốt hơn Logistic Regression (quá đơn giản)",
          "và Random Forest (dễ overfit với dữ liệu nhiễu)."],
         bg=NAVYBG,bd=ND)

    R(sl,.3,4.70,5.5,2.40,fc=LBLUE)
    T(sl,"⚙️  Siêu tham số",.40,4.75,5.3,.38,sz=12,bold=True,color=NAVY)
    configs=[
        ("iterations = 600","depth = 6","l2_leaf_reg = 3.0"),
        ("class_weights = 'Balanced'","random_seed = 42","thread_count = 1  ← Tái lập 100%"),
    ]
    for ri,row in enumerate(configs):
        for ci,val in enumerate(row):
            T(sl,f"• {val}",.40+ci*1.82,5.18+ri*.46,1.78,.40,sz=10,color=BLACK)

    # Feature importance (right, large)
    P(sl, FIG/"03_catboost_feature_importance.png", 5.95,1.22,7.10,5.85)

# ══════════════════════════════════════════════════════════
# SLIDE 10 — KEY RESULTS: ROC + table (MOST IMPORTANT)
# ══════════════════════════════════════════════════════════
def s10():
    sl = new()
    nav(sl,3,10)
    hdr(sl,"⭐  Kết quả Cốt lõi — Bảng Đối soát & Biểu đồ ROC",
        sub="Chứng minh bằng số liệu: Phương pháp chia dữ liệu ảnh hưởng trực tiếp đến độ tin cậy")

    # Left table
    hdrs=["Phương pháp","Rò rỉ","AUC","Nhận xét"]
    hw=[2.35,1.0,.82,2.10]; hx=[.3]
    for w in hw[:-1]: hx.append(hx[-1]+w+.04)
    for j,(h,cx,cw) in enumerate(zip(hdrs,hx,hw)):
        R(sl,cx,1.28,cw,.36,fc=NAVY)
        T(sl,h,cx+.03,1.30,cw-.06,.32,sz=9.5,bold=True,color=WHITE,align=PP_ALIGN.CENTER)

    rows=[
        ("Cách 1: Random K-Fold","Cực nặng","0.8582","❌ Ảo giác",RED),
        ("Cách 1b: Grouped K-Fold","Vừa","0.7454","⚠️ Còn cao",AMBER),
        ("Cách 2: Walk-Forward","Nhẹ","0.5875","✅ Gần thật",GREEN),
        ("Cách 3: Purged WF","Triệt để","0.5895","🛡️ Chuẩn",NAVYBG),
        ("🎯 Holdout — Thi thật","0% (mù)","0.6046","✅ TRÙNG C3!",LBLUE),
    ]
    for ri,(p,rr,auc,verdict,bg) in enumerate(rows):
        ry=1.66+ri*.50
        for j,(val,cx,cw) in enumerate(zip([p,rr,auc,verdict],hx,hw)):
            R(sl,cx,ry,cw,.44,fc=bg)
            T(sl,val,cx+.04,ry+.04,cw-.08,.36,sz=10,
              bold=(ri==4),color=ND if ri==4 else BLACK,
              align=PP_ALIGN.CENTER if j!=0 else PP_ALIGN.LEFT)

    # Key takeaway
    R(sl,.3,4.26,6.5,.58,fc=NAVY)
    T(sl,"👉  AUC Cách 1 (0.86) vs Holdout (0.60): Sai lệch 47%!\n"
        "👉  AUC Cách 3 (0.59) vs Holdout (0.60): Sai lệch chỉ 1.5%! ← Cách 3 = Chuẩn tin cậy",
      .4,4.29,6.2,.52,sz=10.5,bold=True,color=WHITE)

    # 3-nhip reading
    R(sl,.3,4.92,6.5,2.15,fc=LBLUE); R(sl,.3,4.92,6.5,2.15,lc=NAVY,lw=1)
    T(sl,"📖  Đọc biểu đồ ROC theo 3 nhịp:",.40,4.97,6.3,.34,sz=11,bold=True,color=NAVY)
    ML(sl,[
        ("Nhịp 1 — Đường đỏ AUC 0.86: Trông đẹp nhưng là ảo do rò rỉ.",False,11,RD),
        ("Nhịp 2 — Đường xanh/vàng 0.59: Thấp hơn nhưng phản ánh độ khó thật của bài toán.",False,11,GD),
        ("Nhịp 3 — Đường đứt nét Holdout 0.60 ≈ Cách 3 (0.59) ← Chứng minh Cách 3 là đúng!",True,11.5,ND),
    ],.40,5.35,6.3,1.60,sb=5)

    # ROC chart (right, large)
    P(sl, FIG/"05_roc_curves_comparison.png", 6.95,1.22,6.10,5.85)

# ══════════════════════════════════════════════════════════
# SLIDE 11 — So sánh 3 mô hình
# ══════════════════════════════════════════════════════════
def s11():
    sl = new()
    nav(sl,3,11)
    hdr(sl,"So sánh 3 Mô hình trên Nhánh Purged Walk-Forward")

    hdrs=["Mô hình","Accuracy","Precision","Recall","F1","AUC"]
    hw=[2.8,1.3,1.3,1.3,1.2,1.5]; hx=[.3]
    for w in hw[:-1]: hx.append(hx[-1]+w+.04)
    for j,(h,cx,cw) in enumerate(zip(hdrs,hx,hw)):
        R(sl,cx,1.22,cw,.40,fc=NAVY)
        T(sl,h,cx+.03,1.24,cw-.06,.36,sz=10.5,bold=True,color=WHITE,align=PP_ALIGN.CENTER)

    m_rows=[
        ("Logistic Regression","58.12%","34.20%","48.50%","0.3805","0.5412",RGBColor(0xFE,0xF9,0xC3),False),
        ("Random Forest","61.45%","36.80%","42.10%","0.3620","0.5630",RGBColor(0xFE,0xF3,0xC7),False),
        ("CatBoost  ← Tốt nhất","63.80%","39.50%","45.20%","0.3950","0.5895",GREEN,True),
    ]
    for ri,(name,ac,pr,re,f1,auc,bg,bd) in enumerate(m_rows):
        ry=1.64+ri*.54
        for j,(val,cx,cw) in enumerate(zip([name,ac,pr,re,f1,auc],hx,hw)):
            R(sl,cx,ry,cw,.48,fc=bg)
            T(sl,val,cx+.04,ry+.05,cw-.08,.38,sz=10.5,bold=bd,
              color=GD if bd else BLACK,
              align=PP_ALIGN.LEFT if j==0 else PP_ALIGN.CENTER)

    for bx,bw2,title,body,bg,bd in [
        (.3,3.9,"📉  Logistic Regression",
         ["Tuyến tính — Không học được tương tác phi tuyến.",
          "AUC 0.54 — Gần ngưỡng đoán ngẫu nhiên (0.50)."],
         RGBColor(0xFE,0xF9,0xC3),RGBColor(0x92,0x40,0x00)),
        (4.45,3.9,"🌳  Random Forest",
         ["Dễ overfit với dữ liệu nhiễu cao.",
          "AUC 0.56 — Tốt hơn một chút, nhưng kém CatBoost."],
         RGBColor(0xFE,0xF3,0xC7),RGBColor(0x78,0x35,0x00)),
        (8.6,4.68,"🚀  CatBoost — Tốt nhất",
         ["Oblivious Trees → Kiểm soát độ phức tạp.",
          "Gradient boosting tuần tự → Học từ lỗi của cây trước.",
          "AUC 0.59 — Tốt nhất, sát Holdout thực tế nhất."],
         GREEN,GD),
    ]:
        card(sl,bx,3.56+(.6 if bw2<4 else 0),bw2,3.51,title,body,bg=bg,bd=bd,bsz=11)

# ══════════════════════════════════════════════════════════
# SLIDE 12 — Holdout validation
# ══════════════════════════════════════════════════════════
def s12():
    sl = new()
    nav(sl,3,12)
    hdr(sl,"Kiểm định Tập Holdout Niêm phong — \"Đề thi Thật\"",
        sub="5.028 mẫu từ 02/2025–08/2026 · Chưa từng nhìn trong toàn bộ quá trình nghiên cứu")

    for i,(label,val,bg,bd) in enumerate([
        ("AUC Holdout","0.6046",LBLUE,NAVY),
        ("F1-Score","0.4022",GREEN,GD),
        ("Mẫu Holdout","5.028",AMBER,AD),
    ]):
        bx=.5+i*4.15
        R(sl,bx,1.22,3.85,1.65,fc=bg); R(sl,bx,1.22,3.85,1.65,lc=bd,lw=2)
        T(sl,val,bx+.1,1.28,3.65,.88,sz=38,bold=True,color=bd,align=PP_ALIGN.CENTER)
        T(sl,label,bx+.1,2.12,3.65,.62,sz=12,color=BLACK,align=PP_ALIGN.CENTER)

    R(sl,.3,3.00,W-.6,.62,fc=NAVY)
    T(sl,"🎯  Phát hiện then chốt: AUC Holdout (0.60) ≈ AUC Cách 3 (0.59)  —  Sai lệch chỉ 0.015!",
      .4,3.04,4.5,.30,sz=12,bold=True,color=GOLD)
    T(sl,"AUC Cách 1 (0.86) sai lệch 0.26 so với Holdout  ←  Không đáng tin cậy ngoài phòng lab!",
      .4,3.34,W-.8,.24,sz=11,bold=True,color=WHITE)

    comp=[
        ("Cách 1: Random K-Fold","0.8582","0.6046","−0.25","❌ Sai lệch cực lớn",RED),
        ("Cách 1b: Grouped K-Fold","0.7454","0.6046","−0.14","⚠️ Vẫn quá lạc quan",AMBER),
        ("Cách 2: Walk-Forward","0.5875","0.6046","+0.017","✅ Tốt",GREEN),
        ("Cách 3: WF + Purge","0.5895","0.6046","+0.015","🥇 Chính xác nhất!",LBLUE),
    ]
    ch=["Phương pháp","AUC Dev","AUC Holdout","Sai lệch","Độ tin cậy"]
    cw=[3.0,1.5,1.7,1.3,4.7]; cx=[.3]
    for w in cw[:-1]: cx.append(cx[-1]+w+.04)
    T(sl,"📐  Đối chiếu chi tiết:",.3,3.70,6.,.30,sz=11.5,bold=True,color=NAVY)
    for j,(h,x,w) in enumerate(zip(ch,cx,cw)):
        R(sl,x,4.04,w,.34,fc=NAVY)
        T(sl,h,x+.03,4.06,w-.06,.30,sz=9.5,bold=True,color=WHITE,align=PP_ALIGN.CENTER)
    for ri,(p,dv,hd,di,trust,bg) in enumerate(comp):
        ry=4.40+ri*.46
        for j,(val,x,w) in enumerate(zip([p,dv,hd,di,trust],cx,cw)):
            R(sl,x,ry,w,.40,fc=bg)
            T(sl,val,x+.04,ry+.04,w-.08,.32,sz=10,bold=(ri==3),
              color=ND if ri==3 else BLACK,
              align=PP_ALIGN.LEFT if j in[0,4] else PP_ALIGN.CENTER)

    R(sl,.3,6.48,W-.6,.48,fc=NAVYBG); R(sl,.3,6.48,W-.6,.48,lc=NAVY,lw=1)
    T(sl,"📌  Bài học: Bất kỳ báo cáo ML trên chuỗi thời gian nào chỉ dùng Random K-Fold đều KHÔNG đáng tin cậy ngoài phòng thí nghiệm!",
      .4,6.52,W-.8,.40,sz=11,color=ND)

# ══════════════════════════════════════════════════════════
# SLIDE 13 — Financial + equity curves
# ══════════════════════════════════════════════════════════
def s13():
    sl = new()
    nav(sl,3,13)
    hdr(sl,"Hiệu quả Tầng Meta-Labeling & Đường cong Vốn")

    fin=[
        ("Số lệnh","20,007","10,004","−50%",AMBER),
        ("Win Rate","31.71%","34.99%","+3.28%",GREEN),
        ("Profit Factor","1.29","1.40","+8.5%",GREEN),
        ("Max Drawdown","236 R","142 R","−40% ✅",LBLUE),
    ]
    th=["Chỉ số","Trước ML","Sau ML (Top 50%)","Thay đổi"]
    tw=[2.8,2.2,2.8,2.4]; tx=[.3]
    for w in tw[:-1]: tx.append(tx[-1]+w+.04)
    for j,(h,x,w) in enumerate(zip(th,tx,tw)):
        R(sl,x,1.22,w,.36,fc=NAVY)
        T(sl,h,x+.03,1.24,w-.06,.32,sz=10.5,bold=True,color=WHITE,align=PP_ALIGN.CENTER)
    for ri,(m,b,a,ch,bg) in enumerate(fin):
        ry=1.60+ri*.50
        for j,(val,x,w) in enumerate(zip([m,b,a,ch],tx,tw)):
            R(sl,x,ry,w,.44,fc=bg)
            T(sl,val,x+.04,ry+.04,w-.08,.36,sz=11,bold=(j==3),
              color=GD if j==3 else BLACK,
              align=PP_ALIGN.CENTER if j!=0 else PP_ALIGN.LEFT)

    R(sl,.3,3.64,W-.6,.42,fc=NAVY)
    T(sl,"📈  Đường cong vốn — Tập Holdout niêm phong (02/2025–08/2026):",.4,3.68,7.,.30,sz=11,bold=True,color=WHITE)
    T(sl,"Xanh = Baseline (Tầng 1 không lọc)  |  Cam = Top 50% (Sau lọc CatBoost)",.4,3.70+.30,7.,.25,sz=9.5,italic=True,color=GOLD)

    P(sl, STAGE4/"equity-curve-holdout-top50.png", .3,4.14,7.70,3.32)

    # Insight right
    R(sl,8.15,4.14,5.0,3.32,fc=LBLUE); R(sl,8.15,4.14,5.0,3.32,lc=NAVY,lw=1)
    T(sl,"🔍  Nhận xét:"  ,8.25,4.20,4.8,.34,sz=11.5,bold=True,color=NAVY)
    ML(sl,[
        "Kết quả Holdout thấp hơn Dev → Bình thường! Không có rò rỉ.",
        "",
        "Dev Cách 1 (7000R) là ảo giác hoàn toàn.",
        "",
        "Holdout mới là con số khách quan nhất.",
        "",
        "Top 50% không hẳn tốt hơn Baseline trên Holdout → Đây là khó khăn của dữ liệu thật.",
    ],8.25,4.60,4.8,2.70,sz=10.5,color=BLACK,sb=3)

# ══════════════════════════════════════════════════════════
# SLIDE 14 — Confusion matrix
# ══════════════════════════════════════════════════════════
def s14():
    sl = new()
    nav(sl,4,14)
    hdr(sl,"Ma trận Nhầm lẫn — Phân tích Sai số Nghiệp vụ")

    P(sl, FIG/"04_confusion_matrix_holdout.png", .3,1.22,6.5,5.85)

    card(sl,7.0,1.22,5.98,2.64,"💸  FP — Báo nhầm (NGUY HIỂM nhất)",
         ["1,196 lệnh: Mô hình nói 'Thắng' → Thực tế 'Thua'.",
          "→ Mất tiền trực tiếp!",
          "→ Cần giảm tối đa loại sai lầm này.",
          "→ Xảy ra khi thị trường đột ngột đảo chiều."],
         bg=RED,bd=RD)

    card(sl,7.0,3.96,5.98,2.64,"⚠️  FN — Bỏ sót (Thiệt, không mất vốn)",
         ["754 lệnh: Mô hình nói 'Thua' → Thực tế 'Thắng'.",
          "→ Bỏ lỡ cơ hội — Nhưng KHÔNG mất vốn!",
          "→ Hệ thống bảo thủ: Thà bỏ lỡ còn hơn thua lỗ.",
          "→ Đây là trade-off Precision vs Recall trong ML."],
         bg=AMBER,bd=AD)

    R(sl,7.0,6.70,5.98,.37,fc=NAVY)
    T(sl,"Nguyên tắc: Giảm FP > Tăng TP — Bảo toàn vốn ưu tiên số 1",
      7.1,6.73,5.78,.30,sz=11,bold=True,color=WHITE,align=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════════════
# SLIDE 15 — Phát hiện kỹ thuật: thread_count
# ══════════════════════════════════════════════════════════
def s15():
    sl = new()
    nav(sl,4,15)
    hdr(sl,"Phát hiện Kỹ thuật — Tính Tái lập & Đa luồng CPU")

    R(sl,.3,1.22,W-.6,.58,fc=RED); R(sl,.3,1.22,W-.6,.58,lc=RD,lw=2)
    T(sl,"⚠️  CatBoost nhạy cảm với số luồng CPU (thread_count) — Ảnh hưởng đến tính tái lập!",
      .4,1.26,W-.8,.50,sz=12.5,bold=True,color=RD)

    for bx,bw2,title,body,bg,bd in [
        (.3,5.9,"✅  thread_count = 1  (Khuyến nghị)",
         ["Hai lần chạy độc lập → Dự đoán TRÙNG KHỚP 100%.",
          "max|Δp| = 0  → Tính tất định hoàn toàn.",
          "→ Chuẩn bắt buộc trong nghiên cứu khoa học."],
         GREEN,GD),
        (6.55,6.53,"❌  thread_count = -1  (Đa luồng tự do)",
         ["Thứ tự tính toán thay đổi → Sai số dấu phẩy động.",
          "max|Δp| = 0.34  → Lợi nhuận lệch từ −36R đến +30R!",
          "→ Không tái lập → Không đáng tin cậy!"],
         RED,RD),
    ]:
        card(sl,bx,1.92,bw2,2.30,title,body,bg=bg,bd=bd)

    R(sl,.3,4.32,W-.6,.04,fc=GRAY)
    T(sl,"📌  Nguyên nhân: Floating-point non-determinism — Tổng hợp số thực dấu phẩy động theo thứ tự khác nhau cho kết quả khác nhau.",
      .3,4.42,W-.6,.30,sz=11,italic=True,color=BLACK)

    P(sl, STAGE4/"equity-curve-chunk2-5-top50.png", .3,4.80,W-.6,2.60)
    T(sl,"Đường cong vốn tập Dev (2020–2025) — Cách 1 (màu cam vọt cao) là kết quả ảo do rò rỉ dữ liệu, KHÔNG thể đạt được ngoài thực tế!",
      .3,7.46,W-.6,.18,sz=9,italic=True,color=RD)

# ══════════════════════════════════════════════════════════
# SLIDE 16 — Tổng kết số liệu
# ══════════════════════════════════════════════════════════
def s16():
    sl = new()
    nav(sl,4,16)
    hdr(sl,"Tổng kết Toàn bộ Kết quả — Bức tranh Hoàn chỉnh")

    R(sl,.3,1.22,W-.6,1.02,fc=NAVY)
    T(sl,"🏆  3 Câu hỏi Nghiên cứu — 3 Câu trả lời:",.4,1.26,W-.8,.32,sz=12,bold=True,color=GOLD)
    qa=[
        ("CH1: Chia dữ liệu ảnh hưởng bao nhiêu?",    "→ Cực lớn: AUC từ 0.59 (thực) lên 0.86 (ảo) — Chênh lệch 47%!"),
        ("CH2: Phương pháp nào đáng tin cậy?",         "→ Purged WF (0.59) ≈ Holdout thật (0.60). Sai lệch chỉ 1.5%."),
        ("CH3: Meta-Labeling AI có hiệu quả?",          "→ Có: Max Drawdown −40%, Win Rate +3%, Profit Factor +8.5%."),
    ]
    for ri,(q,a) in enumerate(qa):
        ry=1.58+ri*.22
        T(sl,q,.4,ry,5.0,.20,sz=10,bold=True,color=WHITE)
        T(sl,a,5.5,ry,7.2,.20,sz=10.5,bold=True,color=GOLD)

    quads=[
        (.3,2.34,5.9,2.20,"📊  Kết quả Phân loại (Purged WF)",
         ["CatBoost: AUC=0.5895, F1=0.395",
          "Logistic Reg: AUC=0.5412",
          "Random Forest: AUC=0.5630",
          "Holdout: AUC=0.6046  ← Gần CatBoost nhất"],
         NAVYBG,ND),
        (6.55,2.34,6.53,2.20,"💰  Cải thiện Tài chính (Top 50%)",
         ["Số lệnh: −50%  (Loại bỏ lệnh xấu)",
          "Win Rate: +3.28%",
          "Profit Factor: +8.5%",
          "Max Drawdown: −40%  ← Quan trọng nhất!"],
         GREEN,GD),
        (.3,4.64,5.9,2.20,"⚠️  Cảnh báo Phương pháp",
         ["Random K-Fold: AUC=0.86  ← ĐỪNG dùng!",
          "Grouped K-Fold: AUC=0.74  ← Còn sai",
          "Walk-Forward: AUC=0.59  ← Tốt",
          "Purged WF: AUC=0.59  ← CHUẨN TIN CẬY"],
         RED,RD),
        (6.55,4.64,6.53,2.20,"🎓  Đóng góp Học thuật",
         ["Bằng chứng định lượng về ảnh hưởng cách chia.",
          "Pipeline 2 tầng áp dụng mọi chuỗi thời gian.",
          "Phát hiện: thread_count=1 bắt buộc.",
          "Purged WF = tiêu chuẩn tin cậy đã chứng minh."],
         LBLUE,NAVY),
    ]
    for args in quads:
        l,t,w,h,title,items,bg,bd = args
        card(sl,l,t,w,h,title,items,bg=bg,bd=bd,bsz=10.5)

    R(sl,.3,6.94,W-.6,.52,fc=NAVY)
    T(sl,"📌  Kết luận phương pháp luận: Trong ML chuỗi thời gian, cách chia dữ liệu quan trọng hơn thuật toán. "
        "Điểm số đẹp trên Random K-Fold KHÔNG có nghĩa gì ngoài phòng thí nghiệm!",
      .4,6.97,W-.8,.44,sz=10.5,bold=True,color=WHITE)

# ══════════════════════════════════════════════════════════
# SLIDE 17 — Hạn chế & Hướng phát triển
# ══════════════════════════════════════════════════════════
def s17():
    sl = new()
    nav(sl,4,17)
    hdr(sl,"Hạn chế của Đề tài & Hướng Phát triển Tiếp theo")

    card(sl,.3,1.22,5.9,5.85,"⚠️  Hạn chế hiện tại",
         ["Dữ liệu: Chỉ 1 cặp (BTCUSD), 1 khung thời gian (M15).",
          "→ Cần kiểm nghiệm trên nhiều chuỗi thời gian khác.",
          "",
          "Mô hình: Ngưỡng p̂ ≥ 0.5 cố định cho mọi điều kiện.",
          "→ Thị trường biến động mạnh cần ngưỡng linh hoạt hơn.",
          "",
          "Nhãn: Triple-Barrier phụ thuộc tham số k, m thủ công.",
          "→ Có thể tự động hoá bằng phương pháp tối ưu hoá.",
          "",
          "Chưa thử nghiệm kiến trúc Deep Learning.",
          ""],
         bg=RED,bd=RD)

    card(sl,6.65,1.22,6.53,2.72,"🚀  Hướng phát triển kỹ thuật",
         ["Thử Temporal Fusion Transformer, LSTM cho chuỗi thời gian.",
          "Ngưỡng xác suất động theo chế độ thị trường (Regime-switching).",
          "AutoML để tự động tìm siêu tham số tối ưu.",
          "Áp dụng pipeline cho bài toán phi tài chính (y tế, IoT, năng lượng)."],
         bg=GREEN,bd=GD)

    card(sl,6.65,4.04,6.53,3.03,"🔬  Hướng phát triển phương pháp luận",
         ["Nghiên cứu thêm Combinatorial Purged Cross-Validation (CPCV).",
          "So sánh với Expanding Window và Walk-Forward tự thích nghi.",
          "Kiểm tra tính robust khi thay đổi embargo period (10, 30, 100 nến).",
          "Mở rộng sang bài toán multi-class (Thắng lớn / Thắng nhỏ / Thua)."],
         bg=NAVYBG,bd=ND)

# ══════════════════════════════════════════════════════════
# SLIDE 18 — Kết luận & Cảm ơn
# ══════════════════════════════════════════════════════════
def s18():
    sl = new()
    nav(sl,4,18)
    hdr(sl,"Kết luận & Bài học Kinh nghiệm")

    card(sl,.3,1.22,5.9,3.45,"✅  Kết luận 1 — Hệ thống AI 2 Tầng",
         ["Pipeline Meta-Labeling 2 tầng hoạt động hiệu quả.",
          "CatBoost lọc tín hiệu: Max Drawdown −40%, Win Rate +3%.",
          "Kiến trúc áp dụng được cho mọi bài toán phân loại chuỗi thời gian.",
          "Đảm bảo tái lập 100% qua thread_count=1, random_seed=42."],
         bg=GREEN,bd=GD)

    card(sl,6.55,1.22,6.53,3.45,"✅  Kết luận 2 — Phương pháp luận",
         ["Phương pháp chia dữ liệu ảnh hưởng AUC lên đến 47%.",
          "Purged WF (0.59) ≈ Holdout thực tế (0.60) — Sai lệch 1.5%.",
          "Random K-Fold (0.86) là điểm ảo — Không tin cậy!",
          "Đây là cảnh báo quan trọng cho nghiên cứu ML chuỗi thời gian."],
         bg=NAVYBG,bd=ND)

    # Lesson box (big, impactful)
    R(sl,.3,4.77,W-.6,1.10,fc=NAVY)
    T(sl,"💡  Bài học Quan trọng nhất:",
      .4,4.81,W-.8,.34,sz=12.5,bold=True,color=GOLD)
    T(sl,"\"Kết quả đẹp trong phòng lab ≠ Kết quả đúng trong thực tế.\n"
        "Tính liêm chính trong phân chia dữ liệu quan trọng hơn điểm số AUC cao trên giấy.\"",
      .4,5.16,W-.8,.66,sz=13,bold=True,italic=True,color=WHITE,align=PP_ALIGN.CENTER)

    # Real-world reminder
    R(sl,.3,5.97,W-.6,1.10,fc=AMBER)
    T(sl,"🌍  Ứng dụng của kiến thức này:",
      .4,6.01,W-.8,.34,sz=12,bold=True,color=AD)
    T(sl,"Y tế · Năng lượng · IoT · An ninh mạng · Khí tượng · Logistics — Bất kỳ nơi nào có chuỗi dữ liệu theo thời gian!",
      .4,6.36,W-.8,.34,sz=12,bold=True,color=BLACK,align=PP_ALIGN.CENTER)

    R(sl,.3,7.14,W-.6,.32,fc=NAVY)
    T(sl,"🙏  Xin trân trọng cảm ơn Thầy TS. Nguyễn Đình Hiển và các bạn đã lắng nghe!  —  Nhóm 11  |  CS106  |  UIT",
      .4,7.15,W-.8,.28,sz=11,bold=True,color=WHITE,align=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════════════
# BUILD
# ══════════════════════════════════════════════════════════
print("Building 18 slides (v3 — gọn, có hình nến, rõ ứng dụng)...")
s01(); s02(); s03(); s04(); s05(); s06()
s07(); s08(); s09(); s10(); s11(); s12()
s13(); s14(); s15(); s16(); s17(); s18()

out = ROOT / "slides" / "Do_An_MetaLabeling_BTCUSD_18Slide.pptx"
prs.save(str(out))
print(f"Saved: {out}")
print(f"Total slides: {len(prs.slides)}")
