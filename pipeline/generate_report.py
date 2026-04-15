"""
Generate a formatted PDF report from the final Q&A dataset.
Sorted by label → sublabel → source, with clean headings and readable layout.
Run from project root: python generate_report.py
"""
import csv
from pathlib import Path
from collections import defaultdict
from fpdf import FPDF

INPUT_CSV = Path("output/buffett_qa_detailed.csv")
OUTPUT_PDF = Path("output/buffett_qa_dataset_report.pdf")

SOURCE_NAMES = {
    "partnership_letters.pdf": "Partnership Letters (1957-1970)",
    "cunningham_essays.pdf": "Cunningham Essays (1977-2012)",
    "florida_speech.pdf": "Florida Speech (1998)",
    "ivey_2008.pdf": "Ivey Discussion (2008)",
    "notredame_lectures.pdf": "Notre Dame Lectures (1991)",
    "shareholder_letters_2000_2012.pdf": "Shareholder Letters (2000-2012)",
}

# Blue palette
BLUE_DARK  = (26,  82, 118)   # headings
BLUE_MED   = (41, 128, 185)   # accents / table headers
BLUE_LIGHT = (214, 234, 248)  # Q background
GRAY_LIGHT = (245, 245, 245)  # A background / table rows
GRAY_TEXT  = (100, 100, 100)
BODY_TEXT  = (30,  30,  30)
WHITE      = (255, 255, 255)


def sanitize(text: str) -> str:
    """Replace characters outside latin-1 with safe ASCII equivalents."""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "\u2013": "-",    # en dash
        "\u2014": "--",   # em dash
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2026": "...",  # ellipsis
        "\u00a0": " ",    # non-breaking space
        "\u2022": "*",    # bullet
        "\u00e9": "e",    # e acute
        "\u00e8": "e",
        "\u00e0": "a",
        "\u00fc": "u",
        "\u00f6": "o",
        "\u00e4": "a",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Final fallback: drop anything still outside latin-1
    return text.encode("latin-1", errors="replace").decode("latin-1")


class Report(FPDF):
    def header(self):
        if self.page_no() > 2:  # skip title + TOC pages
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*GRAY_TEXT)
            self.cell(0, 5, "Warren Buffett  |  Q&A Training Dataset", align="L")
            self.cell(0, 5, f"Page {self.page_no()}", align="R",
                      new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(200, 200, 200)
            self.line(10, 14, 200, 14)
            self.ln(4)
            self.set_text_color(*BODY_TEXT)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*GRAY_TEXT)
        self.cell(0, 5, "Generated via LLM Pipeline with Quality Scoring", align="C")


def section_rule(pdf, color=BLUE_MED):
    """Draw a thin colored horizontal rule."""
    pdf.set_draw_color(*color)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.set_draw_color(0, 0, 0)


def accent_bar(pdf, y, height, color=BLUE_MED):
    """Draw a vertical accent bar just outside the text area."""
    pdf.set_fill_color(*color)
    pdf.rect(7, y, 2, height, style="F")
    pdf.set_fill_color(*GRAY_LIGHT)


def load_data(csv_path):
    """Load CSV and organize by label -> sublabel -> source."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    organized = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        label    = row.get("Label",    "Unknown")
        sublabel = row.get("Sublabel", "") or "General"
        source   = row.get("Source",   "Unknown")
        organized[label][sublabel][source].append(row)

    return organized, len(rows)


def build_title_page(pdf, total_count, num_labels):
    pdf.add_page()
    pdf.ln(55)

    # Top rule
    pdf.set_draw_color(*BLUE_MED)
    pdf.set_line_width(1)
    pdf.line(30, pdf.get_y(), 180, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 32)
    pdf.set_text_color(*BLUE_DARK)
    pdf.cell(0, 16, "Warren Buffett", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*BLUE_MED)
    pdf.cell(0, 11, "Q&A Training Dataset", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.set_draw_color(*BLUE_MED)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*GRAY_TEXT)
    lines = [
        f"{total_count:,} Question-Answer Pairs",
        f"{num_labels} Thematic Labels  |  6 Primary Sources",
        "Sourced from Buffett Letters, Speeches & Lectures",
        "Filtered and Scored via LLM Quality Pipeline",
    ]
    for line in lines:
        pdf.cell(0, 8, sanitize(line), align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)
    pdf.set_draw_color(*BLUE_MED)
    pdf.set_line_width(1)
    pdf.line(30, pdf.get_y(), 180, pdf.get_y())
    pdf.set_line_width(0.2)


def build_summary_page(pdf, organized):
    pdf.add_page()
    pdf.set_text_color(*BLUE_DARK)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "Dataset Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    section_rule(pdf)
    pdf.ln(6)

    # Table header
    pdf.set_fill_color(*BLUE_MED)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(68, 9, "  Label",          border=0, fill=True)
    pdf.cell(22, 9, "Pairs",            border=0, fill=True, align="C")
    pdf.cell(25, 9, "Sublabels",        border=0, fill=True, align="C")
    pdf.cell(65, 9, "Top Source",       border=0, fill=True,
             new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*BODY_TEXT)
    shade = False
    label_totals = []

    for label in sorted(organized.keys()):
        sublabels = organized[label]
        total_pairs  = sum(len(p) for sub in sublabels.values() for p in sub.values())
        num_sublabels = len(sublabels)

        source_counts = defaultdict(int)
        for sub in sublabels.values():
            for src, pairs in sub.items():
                source_counts[src] += len(pairs)
        top_src  = max(source_counts, key=source_counts.get) if source_counts else "N/A"
        top_name = sanitize(SOURCE_NAMES.get(top_src, top_src))[:38]

        fill_color = (235, 245, 255) if shade else GRAY_LIGHT
        pdf.set_fill_color(*fill_color)
        pdf.cell(68, 8, sanitize(f"  {label}"),   border=0, fill=True)
        pdf.cell(22, 8, str(total_pairs),           border=0, fill=True, align="C")
        pdf.cell(25, 8, str(num_sublabels),         border=0, fill=True, align="C")
        pdf.cell(65, 8, top_name,                   border=0, fill=True,
                 new_x="LMARGIN", new_y="NEXT")
        shade = not shade
        label_totals.append((label, total_pairs))

    pdf.ln(8)

    # Source legend
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*BLUE_DARK)
    pdf.cell(0, 8, "Sources Referenced", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*BODY_TEXT)
    for filename, pretty in SOURCE_NAMES.items():
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(5, 6, "*", new_x="RIGHT", new_y="TOP")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 6, sanitize(f" {pretty}  ({filename})"),
                 new_x="LMARGIN", new_y="NEXT")


def build_qa_pages(pdf, organized):
    for label in sorted(organized.keys()):
        pdf.add_page()

        # ── Label heading ─────────────────────────────────────────────────────
        pdf.set_fill_color(*BLUE_MED)
        pdf.rect(10, pdf.get_y(), 190, 14, style="F")
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(*WHITE)
        pdf.cell(0, 14, sanitize(f"  {label}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*BODY_TEXT)

        total_in_label = sum(
            len(p) for sub in organized[label].values() for p in sub.values()
        )
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*GRAY_TEXT)
        pdf.cell(0, 6,
                 sanitize(f"{total_in_label} pairs  |  "
                           f"{len(organized[label])} sublabels"),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        pdf.set_text_color(*BODY_TEXT)

        pair_num = 0  # running counter within label

        for sublabel in sorted(organized[label].keys()):
            sources = organized[label][sublabel]

            # ── Sublabel heading ──────────────────────────────────────────────
            if pdf.get_y() > 240:
                pdf.add_page()

            sublabel_total = sum(len(p) for p in sources.values())
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(*BLUE_DARK)
            pdf.cell(0, 9,
                     sanitize(f"{sublabel}   ({sublabel_total} pairs)"),
                     new_x="LMARGIN", new_y="NEXT")
            section_rule(pdf, BLUE_MED)
            pdf.ln(3)

            for source in sorted(sources.keys()):
                pairs       = sources[source]
                source_name = sanitize(SOURCE_NAMES.get(source, source))

                # Source label
                pdf.set_font("Helvetica", "BI", 8)
                pdf.set_text_color(*GRAY_TEXT)
                pdf.cell(0, 5, f"  Source: {source_name}",
                         new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)
                pdf.set_text_color(*BODY_TEXT)

                for row in pairs:
                    pair_num += 1
                    question = sanitize(row.get("Questions", ""))
                    answer   = sanitize(row.get("Answers",   ""))
                    quality  = sanitize(row.get("Quality",   ""))

                    # Estimate block height to avoid mid-pair page breaks
                    est_lines_q = max(1, len(question) // 85) + 1
                    est_lines_a = max(1, len(answer)   // 85) + 1
                    est_height  = (est_lines_q + est_lines_a) * 5 + 18
                    if pdf.get_y() + est_height > 268:
                        pdf.add_page()

                    block_top = pdf.get_y()

                    # Indent Q/A text slightly from the left margin
                    # so the accent bar (drawn at x=7) never clips the text.
                    pdf.set_left_margin(14)

                    # Question block
                    pdf.set_fill_color(*BLUE_LIGHT)
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.set_text_color(*BLUE_DARK)
                    pdf.set_x(14)
                    pdf.multi_cell(
                        186, 5.5,
                        sanitize(f"Q{pair_num}:  {question}"),
                        fill=True,
                        new_x="LMARGIN", new_y="NEXT",
                    )
                    pdf.ln(1)

                    # Answer block
                    pdf.set_fill_color(*GRAY_LIGHT)
                    pdf.set_font("Helvetica", "", 9)
                    pdf.set_text_color(*BODY_TEXT)
                    pdf.set_x(14)
                    pdf.multi_cell(
                        186, 5.5,
                        sanitize(f"A:  {answer}"),
                        fill=True,
                        new_x="LMARGIN", new_y="NEXT",
                    )

                    # Quality score
                    if quality:
                        pdf.set_font("Helvetica", "I", 7)
                        pdf.set_text_color(160, 160, 160)
                        pdf.cell(0, 4, f"Quality score: {quality}",
                                 align="R", new_x="LMARGIN", new_y="NEXT")

                    # Restore default left margin
                    pdf.set_left_margin(10)

                    # Left accent bar covering the whole Q+A block
                    block_height = pdf.get_y() - block_top
                    accent_bar(pdf, block_top, block_height)

                    pdf.set_text_color(*BODY_TEXT)
                    pdf.ln(5)

            pdf.ln(3)


def build_pdf(organized, total_count):
    pdf = Report()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(10, 18, 10)

    build_title_page(pdf, total_count, len(organized))
    build_summary_page(pdf, organized)
    build_qa_pages(pdf, organized)

    return pdf


if __name__ == "__main__":
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run assembly notebook first.")
        exit(1)

    print(f"Loading data from {INPUT_CSV}...")
    organized, total = load_data(INPUT_CSV)
    print(f"  {total} pairs across {len(organized)} labels")

    print("Generating PDF...")
    pdf = build_pdf(organized, total)
    pdf.output(str(OUTPUT_PDF))
    print(f"  Saved to {OUTPUT_PDF}")
