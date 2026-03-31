import sys

out_file = "hw1.txt"
with open(out_file, "w", encoding="utf-8") as f:
    try:
        import fitz # PyMuPDF
        doc = fitz.open(sys.argv[1])
        for page in doc:
            f.write(page.get_text() + "\n")
    except ImportError:
        import pypdf
        reader = pypdf.PdfReader(sys.argv[1])
        for page in reader.pages:
            f.write(page.extract_text() + "\n")
