import os, glob, io, sys, re

base = "D:/src/llama.cpp_code2book"
arts = sorted(glob.glob(os.path.join(base, "article_*.md")))

SRC_RE = re.compile(r'源码路径：.*?[-－]\s*`([^`]+)`')

def extract_first_paragraph(paralist):
    if not paralist:
        return "代码实现"
    text = " ".join(paralist).strip()
    # Find first sentence ending
    for endp in ('。', '；', '.', '？', '！'):
        idx = text.find(endp)
        if idx != -1:
            return text[:idx+1].strip()
    return text[:30].strip() if len(text) > 30 else text

for ap in arts:
    with open(ap, encoding="utf-8") as f:
        lines = f.readlines()
    new_lines = []
    in_code = False
    last_paragraph = []  # current paragraph lines (non-blank, non-heading)
    pending_paragraph = None  # paragraph before a 源码路径 line
    expect_code_block = False
    inserted_heading = False
    i = 0
    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()
        # Detect code block fences
        if stripped.startswith("```"):
            if not in_code:
                in_code = True
                if expect_code_block and inserted_heading:
                    new_lines.append(ln)
                    expect_code_block = False
                    inserted_heading = False
                else:
                    new_lines.append(ln)
            else:
                in_code = False
                new_lines.append(ln)
            i += 1
            continue
        if in_code:
            new_lines.append(ln)
            i += 1
            continue
        # Not in code block
        if stripped == "":
            if last_paragraph:
                pending_paragraph = last_paragraph.copy()
                last_paragraph = []
            new_lines.append(ln)
            i += 1
            continue
        if stripped.startswith("### ") or stripped.startswith("## ") or stripped.startswith("# "):
            if last_paragraph:
                pending_paragraph = last_paragraph.copy()
                last_paragraph = []
            new_lines.append(ln)
            i += 1
            continue
        # Regular line
        if stripped.startswith("源码路径："):  # note full-width colon
            m = SRC_RE.search(stripped)
            symbol = m.group(1).strip() if m else ""
            if symbol == "":
                new_lines.append(ln)
                i += 1
                continue
            # Use paragraph before this line
            desc_para = pending_paragraph if pending_paragraph is not None else last_paragraph
            description = extract_first_paragraph(desc_para) if desc_para else "代码实现"
            heading = f"### {symbol} —— {description}\n"
            new_lines.append(heading)
            new_lines.append(ln)
            expect_code_block = True
            inserted_heading = True
            pending_paragraph = None
            last_paragraph = []
            i += 1
            continue
        # Otherwise accumulate paragraph
        last_paragraph.append(stripped)
        new_lines.append(ln)
        i += 1
    # Write back
    with open(ap, "w", encoding="utf-8", newline="") as f:
        f.writelines(new_lines)
    print(f"updated {os.path.basename(ap)}")
print("done")