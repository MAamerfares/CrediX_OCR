"""Flexible Test Runner for Egyptian Document OCR.

Allows testing:
1. A single document (National ID, Bank Statement, Salary Certificate, or iScore)
2. Any combination of documents without requiring all 5
3. Automatic discovery of test documents in a directory

Usage examples:
  python test_runner.py --id_front "path/to/front.jpg" --id_back "path/to/back.jpg"
  python test_runner.py --bank "path/to/statement.pdf"
  python test_runner.py --salary "path/to/salary.pdf"
  python test_runner.py --iscore "path/to/iscore.pdf"
  python test_runner.py --dir "path/to/folder_with_documents"
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Windows CMD can default to a code page that cannot print Arabic or ✓.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_environment():
    """Verify Tesseract and environment settings."""
    tess_cmd = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    print("\n" + "="*60)
    print("ENVIRONMENT CHECK:")
    if Path(tess_cmd).is_file():
        print(f"  [✓] Tesseract executable found: {tess_cmd}")
    else:
        print(f"  [!] Warning: Tesseract not found at default: {tess_cmd}")
        print(r"      In CMD: set TESSERACT_CMD=C:\path\to\tesseract.exe")

    try:
        import pytesseract
        import utils.ocr  # Configure the project Tesseract data directory.
        languages = set(pytesseract.get_languages(config=""))
        if {"ara", "eng"}.issubset(languages):
            print("  [✓] Arabic and English Tesseract data available")
        else:
            print(f"  [!] Missing Tesseract language data: {sorted({'ara', 'eng'} - languages)}")
    except Exception as exc:
        print(f"  [!] Tesseract language check failed: {exc}")
    
    try:
        from utils.ollama_client import is_ollama_running
        if is_ollama_running():
            print("  [✓] Ollama is active on http://127.0.0.1:11434")
        else:
            print("  [-] Ollama is NOT running (Pipeline will use regex/OCR fallbacks cleanly)")
    except Exception:
        print("  [-] Ollama status: unchecked")
    print("="*60 + "\n")

def test_national_id(front_path: str, back_path: str):
    from extractors.national_id import extract
    print(f"\n>>> Processing Egyptian National ID:")
    print(f"    Front: {front_path}")
    print(f"    Back:  {back_path}")
    res = extract(str(front_path), str(back_path))
    return res

def test_bank_statement(path: str, ollama_model: str):
    from extractors.bank_statement import extract
    print(f"\n>>> Processing Bank Statement: {path}")
    res = extract(str(path), ollama_model=ollama_model)
    return res

def test_salary_cert(path: str, ollama_model: str):
    from extractors.salary_cert import extract
    print(f"\n>>> Processing Salary Certificate: {path}")
    res = extract(str(path), ollama_model=ollama_model)
    return res

def test_iscore(path: str, ollama_model: str):
    from extractors.iscore import extract
    print(f"\n>>> Processing iScore Report: {path}")
    res = extract(str(path), ollama_model=ollama_model)
    return res

def auto_detect_files(directory: Path):
    """Auto-detect documents inside a folder based on filenames."""
    files = list(directory.glob("*.*"))
    detected = {}
    for f in files:
        name = f.stem.lower()
        if "front" in name or "id_f" in name or "بطاقة_وجه" in name:
            detected["id_front"] = str(f)
        elif "back" in name or "id_b" in name or "بطاقة_ظهر" in name:
            detected["id_back"] = str(f)
        elif "bank" in name or "statement" in name or "حساب" in name:
            detected["bank"] = str(f)
        elif "salary" in name or "cert" in name or "مرتب" in name or "مفردات" in name:
            detected["salary"] = str(f)
        elif "iscore" in name or "score" in name or "اي_سكور" in name:
            detected["iscore"] = str(f)
    return detected

def main():
    parser = argparse.ArgumentParser(description="Test individual or combined Egyptian document extractors")
    parser.add_argument("--id_front", help="Path to National ID front image")
    parser.add_argument("--id_back", help="Path to National ID back image")
    parser.add_argument("--bank", help="Path to Bank Statement (PDF or image)")
    parser.add_argument("--salary", help="Path to Salary Certificate (PDF or image)")
    parser.add_argument("--iscore", help="Path to iScore Report (PDF or image)")
    parser.add_argument("--dir", help="Path to a folder containing test documents (auto-detects files)")
    parser.add_argument("--ollama_model", default="qwen2.5:7b", help="Ollama model to use for LLM fallbacks")
    parser.add_argument("--output", default="test_output.json", help="Path to save the JSON output")
    args = parser.parse_args()

    check_environment()

    # If --dir was passed, auto-detect
    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"Error: Directory not found: {dir_path}")
            sys.exit(1)
        detected = auto_detect_files(dir_path)
        print(f"Auto-detected files in {dir_path}:")
        for k, v in detected.items():
            print(f"  {k}: {v}")
        if not args.id_front and "id_front" in detected:
            args.id_front = detected["id_front"]
        if not args.id_back and "id_back" in detected:
            args.id_back = detected["id_back"]
        if not args.bank and "bank" in detected:
            args.bank = detected["bank"]
        if not args.salary and "salary" in detected:
            args.salary = detected["salary"]
        if not args.iscore and "iscore" in detected:
            args.iscore = detected["iscore"]

    if not any([args.id_front, args.bank, args.salary, args.iscore]):
        print("No test files specified! Run with --help to see available arguments.")
        print("\nExamples:")
        print("  python test_runner.py --id_front front.jpg --id_back back.jpg")
        print("  python test_runner.py --bank statement.pdf")
        print("  python test_runner.py --dir C:\\path\\to\\test_folder")
        sys.exit(1)

    results = {}
    nid_res = {}
    bank_res = {}
    sal_res = {}
    isc_res = {}

    if args.id_front and args.id_back:
        nid_res = test_national_id(args.id_front, args.id_back)
        results["national_id"] = nid_res
    elif args.id_front or args.id_back:
        print("[!] Warning: Both --id_front and --id_back are required to test National ID.")

    if args.bank:
        bank_res = test_bank_statement(args.bank, args.ollama_model)
        results["bank_statement"] = bank_res

    if args.salary:
        sal_res = test_salary_cert(args.salary, args.ollama_model)
        results["salary_certificate"] = sal_res

    if args.iscore:
        isc_res = test_iscore(args.iscore, args.ollama_model)
        results["iscore_report"] = isc_res

    # If multiple or complete, assemble through main pipeline assembler
    from main import assemble
    from confidence import apply_validation_scores
    assembled = apply_validation_scores(assemble(nid_res, sal_res, bank_res, isc_res))

    out_file = Path(args.output).resolve()
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(assembled, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print(f"EXTRACTION COMPLETE! Saved to: {out_file}")
    print("="*60)
    print("\nSummary of extracted values:")
    if nid_res:
        print("  [National ID]:")
        for k in ["full_name", "national_id", "date_of_birth", "governorate", "address", "gender", "profession_on_id"]:
            print(f"    - {k}: {nid_res.get(k)}")
    if bank_res:
        print("  [Bank Statement]:")
        for k in ["bank_name", "account_number", "avg_monthly_net_inflow", "avg_monthly_balance", "income_regularity_score"]:
            print(f"    - {k}: {bank_res.get(k)}")
    if sal_res:
        print("  [Salary Certificate]:")
        for k in ["employer_name", "job_title", "declared_net_salary", "declared_gross_salary", "employment_date"]:
            print(f"    - {k}: {sal_res.get(k)}")
    if isc_res:
        print("  [iScore Report]:")
        for k in ["credit_score", "score_tier", "score_date", "total_outstanding_balance"]:
            print(f"    - {k}: {isc_res.get(k)}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
