# draw_java_5blocks_with_icons_vector.py
# pip install reportlab pygments
#
# Vector-equivalent version of the Pillow renderer: everything is drawn as PDF vector objects
# (text, lines, polygons, circles). No raster image embedding.

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
import os, re, math

W = 1150
OUTER_MARGIN = 18
INNER_MARGIN = 14

FRAME_W = 4
DASH_W = 2
DASH = 12
GAP = 7

BG = (255, 255, 255)
BLACK = (0, 0, 0)

C_DEFAULT = (0, 0, 0)
C_KEYWORD = (123, 31, 162)
C_TYPE = (0, 121, 107)
C_STRING = (46, 125, 50)
C_NUMBER = (216, 27, 96)
C_COMMENT = (120, 120, 120)
C_ANNOT = (255, 143, 0)
C_OPERATOR = (55, 71, 79)

TITLE_COLOR = (0, 0, 0)

OK_FILL = (0, 170, 90)
BAD_FILL = (220, 40, 60)

def rgb255(t):
    r, g, b = t
    return colors.Color(r / 255.0, g / 255.0, b / 255.0)

def try_register_font(font_name, path):
    try:
        if os.path.exists(path):
            pdfmetrics.registerFont(TTFont(font_name, path))
            return True
    except Exception:
        return False
    return False

FONT_TITLE_NAME = "DejaVuSans-Bold"
FONT_MONO_NAME = "DejaVuSansMono"
FONT_BADGE_NAME = "DejaVuSans-Bold"
FONT_MARK_NAME = "DejaVuSans-Bold"

_dejavu_bold_ok = try_register_font(
    "DejaVuSans-Bold",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
) or try_register_font(
    "DejaVuSans-Bold",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)

_dejavu_mono_ok = try_register_font(
    "DejaVuSansMono",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
) or try_register_font(
    "DejaVuSansMono",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
)

if not _dejavu_bold_ok:
    FONT_TITLE_NAME = "Helvetica-Bold"
    FONT_BADGE_NAME = "Helvetica-Bold"
    FONT_MARK_NAME = "Helvetica-Bold"

if not _dejavu_mono_ok:
    FONT_MONO_NAME = "Courier"

FONT_TITLE_SIZE = 22
FONT_MONO_SIZE = 17
FONT_BADGE_SIZE = 14
FONT_MARK_SIZE = 30

def ascent(font_name, font_size):
    try:
        a = pdfmetrics.getAscent(font_name)
        return (a / 1000.0) * font_size
    except Exception:
        return 0.8 * font_size

def text_width(s, font_name, font_size):
    return pdfmetrics.stringWidth(s, font_name, font_size)

def draw_text_top_left(c, H, x, y_top, text, font_name, font_size, fill_rgb):
    c.setFont(font_name, font_size)
    c.setFillColor(rgb255(fill_rgb))
    baseline = H - y_top - ascent(font_name, font_size)
    c.drawString(x, baseline, text)

def dashed_rect(c, H, x0, y0, x1, y1, dash=DASH, gap=GAP, width=DASH_W, outline=BLACK):
    c.setLineWidth(width)
    c.setStrokeColor(rgb255(outline))
    c.setFillColor(colors.transparent)

    c.setDash([dash, gap], 0)
    c.line(x0, H - y0, x1, H - y0)

    c.setDash([dash, gap], 0)
    c.line(x0, H - y1, x1, H - y1)

    c.setDash([dash, gap], 0)
    c.line(x0, H - y0, x0, H - y1)

    c.setDash([dash, gap], 0)
    c.line(x1, H - y0, x1, H - y1)

    c.setDash()

def draw_number_badge(c, H, x1, y0, number):
    cx, cy = x1 - 38, y0 + 38
    c.setLineWidth(1.5)
    c.setStrokeColor(rgb255((170, 190, 220)))
    c.setFillColor(rgb255((66, 133, 244)))
    c.circle(cx, H - cy, 24, stroke=1, fill=1)

    label = str(number)
    label_w = text_width(label, FONT_BADGE_NAME, FONT_BADGE_SIZE + 8)
    draw_text_top_left(
        c,
        H,
        cx - label_w / 2,
        cy - 12,
        label,
        FONT_BADGE_NAME,
        FONT_BADGE_SIZE + 8,
        (255, 255, 255),
    )

def draw_java_badge_bottom_right(c, H, x1, y1):
    cx, cy = x1 - 38, y1 - 38
    r = 24
    pts = []
    for i in range(6):
        ang = math.radians(30 + i * 60)
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))

    p = c.beginPath()
    x_start, y_start = pts[0]
    p.moveTo(x_start, H - y_start)
    for x, y in pts[1:]:
        p.lineTo(x, H - y)
    p.close()

    c.setStrokeColor(rgb255(BLACK))
    c.setLineWidth(1)
    c.setFillColor(rgb255((235, 235, 235)))
    c.drawPath(p, stroke=1, fill=1)

    draw_text_top_left(c, H, cx - 18, cy - 9, "JAVA", FONT_BADGE_NAME, FONT_BADGE_SIZE, (0, 70, 200))

def draw_status_icon(c, H, cx, cy, ok=True):
    fill = OK_FILL if ok else BAD_FILL
    mark = "✓" if ok else "✗"

    c.setLineWidth(3)
    c.setStrokeColor(colors.white)
    c.setFillColor(rgb255(fill))
    c.circle(cx, H - cy, 26, stroke=1, fill=1)

    draw_text_top_left(c, H, cx - 10, cy - 16, mark, FONT_MARK_NAME, FONT_MARK_SIZE, (255, 255, 255))

def pygments_available():
    try:
        import pygments  # noqa
        return True
    except Exception:
        return False

def tokenize_with_pygments(code: str):
    from pygments import lex
    from pygments.lexers import JavaLexer
    from pygments.token import Token

    def color_for(tok):
        if tok in Token.Comment:
            return C_COMMENT
        if tok in Token.String:
            return C_STRING
        if tok in Token.Number:
            return C_NUMBER
        if tok in Token.Keyword:
            return C_KEYWORD
        if tok in Token.Name.Class or tok in Token.Name.Namespace:
            return C_TYPE
        if tok in Token.Name.Decorator:
            return C_ANNOT
        if tok in Token.Operator or tok in Token.Punctuation:
            return C_OPERATOR
        return C_DEFAULT

    lines = [[]]
    for ttype, text in lex(code, JavaLexer()):
        color = color_for(ttype)
        parts = text.split("\n")
        for i, part in enumerate(parts):
            if part:
                lines[-1].append((part, color))
            if i != len(parts) - 1:
                lines.append([])
    return lines

JAVA_KEYWORDS = {
    "abstract","assert","boolean","break","byte","case","catch","char","class","const",
    "continue","default","do","double","else","enum","extends","final","finally","float",
    "for","goto","if","implements","import","instanceof","int","interface","long","native",
    "new","package","private","protected","public","return","short","static","strictfp",
    "super","switch","synchronized","this","throw","throws","transient","try","void",
    "volatile","while","var","record","sealed","permits","non-sealed","yield"
}
JAVA_LITERALS = {"true","false","null"}

def tokenize_fallback_java(code: str):
    lines_out = []
    in_block_comment = False

    for raw_line in code.splitlines():
        i = 0
        line_tokens = []
        line = raw_line

        while i < len(line):
            if in_block_comment:
                end = line.find("*/", i)
                if end == -1:
                    line_tokens.append((line[i:], C_COMMENT))
                    i = len(line)
                else:
                    line_tokens.append((line[i:end + 2], C_COMMENT))
                    i = end + 2
                    in_block_comment = False
                continue

            if line.startswith("//", i):
                line_tokens.append((line[i:], C_COMMENT))
                break

            if line.startswith("/*", i):
                end = line.find("*/", i + 2)
                if end == -1:
                    line_tokens.append((line[i:], C_COMMENT))
                    in_block_comment = True
                    break
                line_tokens.append((line[i:end + 2], C_COMMENT))
                i = end + 2
                continue

            ch = line[i]

            if ch.isspace():
                j = i + 1
                while j < len(line) and line[j].isspace():
                    j += 1
                line_tokens.append((line[i:j], C_DEFAULT))
                i = j
                continue

            if ch == '"':
                j = i + 1
                while j < len(line):
                    if line[j] == "\\" and j + 1 < len(line):
                        j += 2
                        continue
                    if line[j] == '"':
                        j += 1
                        break
                    j += 1
                line_tokens.append((line[i:j], C_STRING))
                i = j
                continue

            if ch == "'":
                j = i + 1
                while j < len(line):
                    if line[j] == "\\" and j + 1 < len(line):
                        j += 2
                        continue
                    if line[j] == "'":
                        j += 1
                        break
                    j += 1
                line_tokens.append((line[i:j], C_STRING))
                i = j
                continue

            if ch == "@":
                j = i + 1
                while j < len(line) and (line[j].isalnum() or line[j] in {"_", "."}):
                    j += 1
                line_tokens.append((line[i:j], C_ANNOT))
                i = j
                continue

            if ch.isdigit():
                j = i + 1
                while j < len(line) and re.match(r"[0-9a-fA-FxX_\.]", line[j]):
                    j += 1
                line_tokens.append((line[i:j], C_NUMBER))
                i = j
                continue

            if ch.isalpha() or ch in {"_", "$"}:
                j = i + 1
                while j < len(line) and (line[j].isalnum() or line[j] in {"_", "$"}):
                    j += 1
                word = line[i:j]
                if word in JAVA_KEYWORDS:
                    line_tokens.append((word, C_KEYWORD))
                elif word in JAVA_LITERALS:
                    line_tokens.append((word, C_NUMBER))
                elif word and word[0].isupper():
                    line_tokens.append((word, C_TYPE))
                else:
                    line_tokens.append((word, C_DEFAULT))
                i = j
                continue

            line_tokens.append((ch, C_OPERATOR))
            i += 1

        lines_out.append(line_tokens if line_tokens else [("", C_DEFAULT)])
    return lines_out

def tokenize_java(code: str):
    if pygments_available():
        try:
            return tokenize_with_pygments(code)
        except Exception:
            return tokenize_fallback_java(code)
    return tokenize_fallback_java(code)

def _split_to_fit(s, font_name, font_size, max_w):
    lo, hi = 1, len(s)
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if text_width(s[:mid], font_name, font_size) <= max_w:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best

def measure_tokens_line_wrapped(tokens_line, max_w, font_name, font_size):
    cur_x = 0.0
    lines_used = 1

    for text, _ in tokens_line:
        if not text:
            continue

        idx = 0
        while idx < len(text):
            remain = text[idx:]
            w = text_width(remain, font_name, font_size)

            if cur_x + w <= max_w:
                cur_x += w
                break

            fit = _split_to_fit(remain, font_name, font_size, max_w - cur_x)
            if fit <= 0:
                lines_used += 1
                cur_x = 0.0
                continue

            cur_x += text_width(remain[:fit], font_name, font_size)
            idx += fit
            lines_used += 1
            cur_x = 0.0

    return lines_used

def estimate_block_height(code, area_w, font_name, font_size, line_h, pad_top, pad_bottom, min_h):
    token_lines = tokenize_java(code)
    visual_lines = 0
    for tl in token_lines:
        visual_lines += measure_tokens_line_wrapped(tl, area_w, font_name, font_size)

    needed = int(pad_top + visual_lines * line_h + pad_bottom + 6)
    return max(min_h, needed)

def draw_tokens_wrapped(c, H, tokens_line, x, y, max_w, font_name, font_size, line_h):
    cur_x = x
    cur_y = y

    for text, color in tokens_line:
        if not text:
            continue

        idx = 0
        while idx < len(text):
            remain = text[idx:]
            w = text_width(remain, font_name, font_size)

            if cur_x + w <= x + max_w:
                draw_text_top_left(c, H, cur_x, cur_y, remain, font_name, font_size, color)
                cur_x += w
                break

            fit = _split_to_fit(remain, font_name, font_size, x + max_w - cur_x)
            piece = remain[:fit]
            draw_text_top_left(c, H, cur_x, cur_y, piece, font_name, font_size, color)

            idx += fit
            cur_x = x
            cur_y += line_h

    return cur_y

def draw_code_block(
    c,
    H,
    box,
    title,
    code,
    ok=True,
    badge_number=None,
    show_status_icon=True,
    show_java_badge_bottom_right=False,
):
    x0, y0, x1, y1 = box
    dashed_rect(c, H, x0, y0, x1, y1)

    draw_text_top_left(c, H, x0 + 12, y0 + 10, title, FONT_TITLE_NAME, FONT_TITLE_SIZE, TITLE_COLOR)
    if badge_number is not None:
        draw_number_badge(c, H, x1, y0, badge_number)

    icon_space = 78
    pad_left = 20
    pad_top = 58
    pad_right = 20 + icon_space
    pad_bottom = 18

    area_x = x0 + pad_left
    area_y = y0 + pad_top
    area_w = (x1 - x0) - pad_left - pad_right
    line_h = 24

    token_lines = tokenize_java(code)
    cur_y = area_y
    for tl in token_lines:
        cur_y = draw_tokens_wrapped(c, H, tl, area_x, cur_y, area_w, FONT_MONO_NAME, FONT_MONO_SIZE, line_h)
        cur_y += line_h
        if cur_y > y1 - pad_bottom:
            break

    if show_status_icon:
        draw_status_icon(c, H, x1 - 38, y1 - 38, ok=ok)
    if show_java_badge_bottom_right:
        draw_java_badge_bottom_right(c, H, x1, y1)

CODE_BLOCKS = [
    {
        "title": "Baseline:",
        "ok": False,
        "code": """public void readFrom(final InputStream in) throws IOException {
    buffer.clear();
    int read;
    while ((read = in.read(buffer.array(), buffer.position(), buffer.remaining())) != -1) {
        buffer.position(buffer.position() + read);
    }
    buffer.flip();
}""",
    },
    {
        "title": "Structure-Constrained:",
        "ok": False,
        "code": """public void readFrom(final InputStream in) throws IOException {
    buffer.clear();
    int read;
    while ((read = in.read(buffer.array(), buffer.position(), buffer.remaining())) != -1) {
        buffer.position(buffer.position() + read);
    }
    buffer.flip();
}""",
    },
    {
        "title": "Robust-Handling:",
        "ok": False,
        "code": """public void readFrom(final InputStream in) throws IOException {
    if (in == null) {
        throw new IllegalArgumentException("InputStream cannot be null");
    }
    try {
        byte[] buffer = new byte[in.available()];
        int bytesRead = in.read(buffer);
        if (bytesRead != buffer.length) {
            throw new IOException("Failed to read the expected number of bytes");
        }
        this.buffer = buffer;
        this.position = 0;
    } finally {
        in.close();
    }
}""",
    },
    {
        "title": "Reasoning-Guided:",
        "ok": True,
        "code": """public void readFrom(final InputStream in) throws IOException {
    if (in == null) {
        throw new IllegalArgumentException("InputStream cannot be null");
    }
    try {
        byte[] buffer = new byte[in.available()];
        int bytesRead = in.read(buffer);
        if (bytesRead != buffer.length) {
            throw new IOException("Failed to read the expected number of bytes");
        }
        this.buffer = buffer;
    } finally {
        in.close();
    }
}""",
    },
    {
        "title": "Edge-Coverage:",
        "ok": False,
        "code": """public void readFrom(final InputStream in) throws IOException {
    if (in == null) {
        throw new IllegalArgumentException("InputStream cannot be null");
    }
    try {
        this.buffer = new byte[0];
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte[] buffer = new byte[1024];
        int bytesRead;
        while ((bytesRead = in.read(buffer)) != -1) {
            baos.write(buffer, 0, bytesRead);
        }
        this.buffer = baos.toByteArray();
        this.readPointer = 0;
    } catch (IOException e) {
        throw new IOException("Error reading from InputStream", e);
    }
}""",
    },
    {
        "title": "Official-Implementation:",
        "ok": True,
        "code": """public void readFrom(final InputStream in) throws IOException {
    pointer = 0;
    size = 0;
    int n;
    do {
        n = in.read(buffer, size, buffer.length - size);
        if (n > 0) {
            size += n;
        }
        resizeIfNeeded();
    } while (n >= 0);
}""",
    },
]

def main(out_path="example.pdf"):
    gap = 12

    icon_space = 78
    pad_left = 20
    pad_top = 58
    pad_right = 20 + icon_space
    pad_bottom = 18
    line_h = 24
    min_h = 170

    x0 = OUTER_MARGIN + INNER_MARGIN
    x1 = W - OUTER_MARGIN - INNER_MARGIN
    area_w = (x1 - x0) - pad_left - pad_right

    block_heights = []
    for blk in CODE_BLOCKS:
        h = estimate_block_height(
            blk["code"],
            area_w,
            FONT_MONO_NAME,
            FONT_MONO_SIZE,
            line_h,
            pad_top,
            pad_bottom,
            min_h,
        )
        block_heights.append(h-30)

    total_h = OUTER_MARGIN * 2 + INNER_MARGIN * 2 + sum(block_heights) + gap * (len(block_heights) - 1)
    H = max(total_h, 980)

    c = canvas.Canvas(out_path, pagesize=(W, H))
    c.setFillColor(rgb255(BG))
    c.rect(0, 0, W, H, stroke=0, fill=1)

    c.setStrokeColor(rgb255(BLACK))
    c.setLineWidth(FRAME_W)
    c.setFillColor(colors.transparent)
    c.rect(OUTER_MARGIN, OUTER_MARGIN, W - OUTER_MARGIN * 2, H - OUTER_MARGIN * 2, stroke=1, fill=0)

    y = OUTER_MARGIN + INNER_MARGIN
    total_blocks = len(CODE_BLOCKS)
    for idx, (blk, bh) in enumerate(zip(CODE_BLOCKS, block_heights)):
        box = (x0, y, x1, y + bh)
        draw_code_block(
            c,
            H,
            box,
            blk["title"],
            blk["code"],
            ok=blk["ok"],
            badge_number=(idx + 1) if idx < 5 else None,
            show_status_icon=idx != (total_blocks - 1),
            show_java_badge_bottom_right=idx == (total_blocks - 1),
        )
        y += bh + gap

    c.showPage()
    c.save()
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
