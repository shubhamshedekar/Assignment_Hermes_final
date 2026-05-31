import streamlit as st
import pandas as pd
import numpy as np
import fitz
import json
import re
import logging
from difflib import SequenceMatcher
from PIL import Image
from paddleocr import PaddleOCR
from groq import Groq

# -----------------------------
# LOGGING SETUP
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Trade Document Validator", layout="wide")

if "GROQ_API_KEY" not in st.secrets:
    st.error("❌ GROQ_API_KEY not found in secrets. Please add it to .streamlit/secrets.toml")
    st.stop()

client = Groq(api_key=st.secrets["GROQ_API_KEY"])
ocr = PaddleOCR(use_angle_cls=True, lang="en")

MAX_RETRIES = 3


# -----------------------------
# PDF -> IMAGE (ALL PAGES)
# -----------------------------
def convert_pdf_to_images(pdf_file):
    images = []
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        logger.info(f"PDF has {len(doc)} page(s)")
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        return images
    except Exception as e:
        logger.error(f"PDF conversion error: {e}")
        st.error(f"PDF conversion error: {e}")
        return []


# -----------------------------
# IMAGE -> TEXT (ALL PAGES)
# -----------------------------
def img_to_text(images):
    all_text = []
    try:
        for idx, img in enumerate(images):
            img_np = np.array(img)
            result = ocr.ocr(img_np)
            for block in result:
                for line in block:
                    all_text.append(line[1][0])
            logger.info(f"Page {idx+1} OCR complete: {len(all_text)} lines so far")
        return " ".join(all_text)
    except Exception as e:
        logger.error(f"OCR error: {e}")
        st.error(f"OCR error: {e}")
        return None


# -----------------------------
# GROQ EXTRACTION (WITH RETRY)
# -----------------------------
def key_extraction(ocr_text):
    prompt = f"""
    You are an expert document information extraction system.

    Extract the following fields from the document:
    - Shipper, Consignee, Product_Description, Quantity, Gross_Weight, Net_Weight, Packages, Invoice_Value

    Rules:
    1. Return ONLY valid JSON with exactly these keys.
    2. If a value is not found, return an empty string "".
    3. Do not add explanations, markdown, or extra text.
    4. Extract the most relevant value even if labels vary slightly.
    5. For weights and quantities, include the unit (e.g. "500 KG").

    Document OCR Text:
    {ocr_text}
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Extraction succeeded on attempt {attempt}")
            return result
        except Exception as e:
            logger.warning(f"Extraction attempt {attempt} failed: {e}")
            if attempt == MAX_RETRIES:
                st.error(f"Extraction failed after {MAX_RETRIES} attempts: {e}")
                return None


# -----------------------------
# RULE-BASED CONFIDENCE SCORING
# -----------------------------

def normalize_string(s):
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def extract_number(s):
    s = str(s).replace(",", "")
    match = re.search(r"[\d]+(?:\.\d+)?", s)
    return float(match.group()) if match else None


def string_similarity(a, b):
    return SequenceMatcher(None, normalize_string(a), normalize_string(b)).ratio()


def numeric_confidence(a, b, tolerance=0.02):
    na, nb = extract_number(a), extract_number(b)
    if na is None or nb is None:
        if na is None and nb is None:
            return 1.0    
        return 0.0      
    if na == 0 and nb == 0:
        return 1.0
    diff = abs(na - nb) / max(abs(na), abs(nb))
    if diff <= tolerance:
        return 1.0
    return max(0.0, 1.0 - diff)


FIELD_TYPES = {
    "Shipper": "text",
    "Consignee": "text",
    "Product_Description": "text",
    "Quantity": "numeric",
    "Gross_Weight": "numeric",
    "Net_Weight": "numeric",
    "Packages": "numeric",
    "Invoice_Value": "numeric",
}

MATCH_THRESHOLD = 0.80   # Confidence >= 80% → MATCH


def rule_based_validate(invoice_json, packing_json):
    results = {}
    for field, ftype in FIELD_TYPES.items():
        inv_val = str(invoice_json.get(field, "")).strip()
        pak_val = str(packing_json.get(field, "")).strip()

        if ftype == "numeric":
            score = numeric_confidence(inv_val, pak_val)
        else:
            score = string_similarity(inv_val, pak_val)

        status = "MATCH" if score >= MATCH_THRESHOLD else "MISMATCH"
        results[field] = {
            "invoice_value": inv_val,
            "packing_list_value": pak_val,
            "confidence": round(score * 100, 1),
            "rule_status": status,
            "field_type": ftype,
        }
        logger.info(f"Rule-based [{field}]: {status} (confidence={score:.2%})")
    return results


# -----------------------------
# LLM SEMANTIC VALIDATION
# -----------------------------
def llm_validate(invoice_json, packing_json):
    prompt = f"""
    You are an expert Trade Document Validation Assistant.

    Compare Invoice and Packing List JSON and return structured validation.

    COMPARISON RULES:
    1. Shipper / Consignee: ignore punctuation, case. Core name must match.
    2. Product_Description: MATCH if same product, even with different prefixes.
    3. Quantity / Gross_Weight / Net_Weight / Packages / Invoice_Value: numeric comparison only.

    OUTPUT FORMAT (valid JSON only, no extra text):
    {{
      "overall_status": "PASS or FAIL",
      "matched_fields": [],
      "mismatched_fields": [],
      "field_validation": [
        {{
          "field": "",
          "invoice_value": "",
          "packing_list_value": "",
          "status": "MATCH or MISMATCH",
          "reason": ""
        }}
      ],
      "summary": "One-line summary"
    }}

    Invoice JSON: {invoice_json}
    Packing List JSON: {packing_json}
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            logger.info(f"LLM validation succeeded on attempt {attempt}")
            return result
        except Exception as e:
            logger.warning(f"LLM validation attempt {attempt} failed: {e}")
            if attempt == MAX_RETRIES:
                st.error(f"LLM validation failed after {MAX_RETRIES} attempts: {e}")
                return None


# -----------------------------
# HYBRID MERGE
# -----------------------------
def hybrid_validate(invoice_json, packing_json):
    rule_results = rule_based_validate(invoice_json, packing_json)
    llm_results = llm_validate(invoice_json, packing_json)

    if not llm_results:
        return None, rule_results

    # Build a lookup from LLM field_validation
    llm_lookup = {}
    for item in llm_results.get("field_validation", []):
        llm_lookup[item["field"]] = item

    hybrid_fields = []
    for field, rule_data in rule_results.items():
        ftype = rule_data["field_type"]
        llm_item = llm_lookup.get(field, {})
        llm_status = llm_item.get("status", "UNKNOWN")
        rule_status = rule_data["rule_status"]

        # Choose authoritative source
        if ftype == "numeric":
            final_status = rule_status
            authority = "Rule-based"
        else:
            final_status = llm_status if llm_status != "UNKNOWN" else rule_status
            authority = "LLM"

        # Agreement flag
        if llm_status == "UNKNOWN":
            agreement = "LLM N/A"
        elif rule_status == llm_status:
            agreement = "✅ Both Agree"
        else:
            agreement = "⚠️ Disagree — Review"
            if ftype == "text":
                rule_data["confidence"] = round(rule_data["confidence"] * 0.5, 1)

        hybrid_fields.append({
            "field": field,
            "invoice_value": rule_data["invoice_value"],
            "packing_list_value": rule_data["packing_list_value"],
            "confidence": rule_data["confidence"],
            "final_status": final_status,
            "rule_status": rule_status,
            "llm_status": llm_status,
            "agreement": agreement,
            "authority": authority,
            "reason": llm_item.get("reason", "Numeric comparison"),
        })


    overall_confidence = round(
        sum(f["confidence"] for f in hybrid_fields) / len(hybrid_fields), 1
    )

    all_match = all(f["final_status"] == "MATCH" for f in hybrid_fields)
    overall_status = "PASS" if all_match and overall_confidence >= 80 else "FAIL"

    return {
        "overall_status": overall_status,
        "overall_confidence": overall_confidence,
        "summary": llm_results.get("summary", ""),
        "matched_fields": [f["field"] for f in hybrid_fields if f["final_status"] == "MATCH"],
        "mismatched_fields": [f["field"] for f in hybrid_fields if f["final_status"] == "MISMATCH"],
        "field_validation": hybrid_fields,
    }, rule_results


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("📄 Trade Document Validator")
st.caption("Invoice vs Packing List — Hybrid AI + Rule-Based Validation with Confidence Scoring")

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Validation Strategy:**
    - 🔢 **Numeric fields** (Quantity, Weights, Packages, Invoice Value):  
      Deterministic rule-based comparison with tolerance ±2%
    - 📝 **Text fields** (Shipper, Consignee, Product Description):  
      LLM semantic comparison (handles abbreviations, punctuation)
    - 🤝 **Hybrid merge:** Both engines run; disagreements are flagged for review

    **Confidence Score:**
    - Per-field score (0–100%)
    - Overall document confidence = mean of all fields
    - PASS requires all fields MATCH + confidence ≥ 80%
    """)
    st.divider()
    st.markdown("**Match Threshold:** 80%")
    st.markdown("**Numeric Tolerance:** ±2%")
    st.markdown("**LLM Model:** llama-3.3-70b-versatile")

invoice_file = st.file_uploader("📤 Upload Invoice PDF", type=["pdf"])
packing_file = st.file_uploader("📤 Upload Packing List PDF", type=["pdf"])

if st.button("🚀 Run Validation", type="primary"):

    if not invoice_file or not packing_file:
        st.error("Please upload both files before running validation.")
        st.stop()

    # Step 1: PDF → Images
    with st.spinner("📄 Converting PDFs to images..."):
        invoice_imgs = convert_pdf_to_images(invoice_file)
        packing_imgs = convert_pdf_to_images(packing_file)

    if not invoice_imgs or not packing_imgs:
        st.error("Failed to convert one or both PDFs. Please check the files.")
        st.stop()

    st.info(f"Invoice: {len(invoice_imgs)} page(s) | Packing List: {len(packing_imgs)} page(s)")

    # Step 2: OCR
    with st.spinner("🔍 Running OCR on all pages..."):
        invoice_text = img_to_text(invoice_imgs)
        packing_text = img_to_text(packing_imgs)

    if not invoice_text or not packing_text:
        st.error("OCR failed on one or both documents.")
        st.stop()

    # Step 3: Extract structured data
    with st.spinner("🧠 Extracting structured fields..."):
        invoice_json = key_extraction(invoice_text)
        packing_json = key_extraction(packing_text)

    if not invoice_json or not packing_json:
        st.error("Field extraction failed.")
        st.stop()

    with st.expander("📦 Invoice Extracted Fields"):
        st.json(invoice_json)
    with st.expander("📦 Packing List Extracted Fields"):
        st.json(packing_json)

    # Step 4: Hybrid Validation
    with st.spinner("⚖️ Running hybrid validation (rules + LLM)..."):
        report, rule_details = hybrid_validate(invoice_json, packing_json)

    if not report:
        st.error("Validation failed. Please try again.")
        st.stop()

    st.success("✅ Validation Complete")

    # -----------------------------
    # SUMMARY METRICS
    # -----------------------------
    st.subheader("📊 Summary")
    st.write(report.get("summary", ""))

    col1, col2, col3, col4 = st.columns(4)
    status = report.get("overall_status", "N/A")
    confidence = report.get("overall_confidence", 0)
    matched = len(report.get("matched_fields", []))
    mismatched = len(report.get("mismatched_fields", []))

    col1.metric("Overall Status", status, delta="✅ PASS" if status == "PASS" else "❌ FAIL")
    col2.metric("Overall Confidence", f"{confidence}%")
    col3.metric("Matched Fields", matched)
    col4.metric("Mismatched Fields", mismatched)

    # Confidence bar
    st.markdown("**Overall Document Confidence**")
    st.progress(int(confidence) / 100)

    # -----------------------------
    # FIELD-WISE RESULTS TABLE
    # -----------------------------
    st.subheader("🔍 Field-wise Validation")

    table_data = []
    for item in report.get("field_validation", []):
        table_data.append({
            "Field": item["field"],
            "Invoice Value": item["invoice_value"],
            "Packing List Value": item["packing_list_value"],
            "Confidence %": item["confidence"],
            "Final Status": item["final_status"],
            "Rule": item["rule_status"],
            "LLM": item["llm_status"],
            "Agreement": item["agreement"],
            "Authority": item["authority"],
        })

    df = pd.DataFrame(table_data)

    def color_status(val):
        if val == "MATCH":
            return "background-color: #d4edda; color: #155724"
        elif val == "MISMATCH":
            return "background-color: #f8d7da; color: #721c24"
        return ""

    def color_agreement(val):
        if "Disagree" in str(val):
            return "background-color: #fff3cd; color: #856404"
        return ""

    styled = df.style.applymap(color_status, subset=["Final Status", "Rule", "LLM"]) \
                     .applymap(color_agreement, subset=["Agreement"]) \
                     .format({"Confidence %": "{:.1f}%"})
    st.dataframe(styled, use_container_width=True)

    # -----------------------------
    # DETAIL VIEW PER FIELD
    # -----------------------------
    st.subheader("📋 Field Detail")
    for item in report.get("field_validation", []):
        status_icon = "🟢" if item["final_status"] == "MATCH" else "🔴"
        with st.expander(f"{status_icon} {item['field']} — {item['confidence']}% confidence"):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Invoice:** `{item['invoice_value']}`")
            c2.markdown(f"**Packing List:** `{item['packing_list_value']}`")
            st.progress(int(item["confidence"]) / 100)
            st.write(f"**Reason:** {item['reason']}")
            st.write(f"**Authority:** {item['authority']} | **Agreement:** {item['agreement']}")

    # -----------------------------
    # DOWNLOAD REPORT
    # -----------------------------
    st.subheader("⬇️ Download Report")
    full_report = {
        "validation_report": report,
        "rule_based_details": rule_details,
        "extracted_fields": {
            "invoice": invoice_json,
            "packing_list": packing_json,
        }
    }
    st.download_button(
        label="📥 Download Full JSON Report",
        data=json.dumps(full_report, indent=4),
        file_name="validation_report.json",
        mime="application/json"
    )
