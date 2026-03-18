"""
Document Review System with Hugging Face LLM
- Models cached (no reload on rerun)
- Deduplication at page + chunk + query level
- Smart company/doc-type extractor (bypasses weak LLM)
- Dark UI theme
"""

import os
import re
import tempfile
import torch
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline
)
import streamlit as st


# ==================== CONFIG ====================
class Config:
    MODE = 'local'
    LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LOCAL_LLM_MODEL = "google/flan-t5-base"
    HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    API_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    MAX_DOCUMENTS = 10
    RETRIEVAL_K = 3


# ==================== PROMPTS ====================
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an expert document analyst. Use ONLY the context below to answer.\n"
        "Be specific and extract exact values (names, dates, numbers, lists).\n"
        "If the answer is a list (like skills), list every item found.\n"
        "Do NOT guess. If not in context, say: Not found in document.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Precise answer:"
    )
)

SUMMARY_PROMPT_TEXT = (
    "Summarize this document in 5 key points:\n\n"
    "{text}\n\n"
    "Summary:"
)


# ==================== SMART EXTRACTORS ====================

DOC_TYPE_KEYWORDS = {
    "admit card": [
        "admit card", "admission card", "hall ticket", "roll no",
        "roll number", "examination", "exam date", "center code",
        "reporting time", "invigilator"
    ],
    "resume / CV": [
        "skills", "experience", "education", "projects",
        "gpa", "b.tech", "internship", "objective"
    ],
    "cover letter": [
        "dear sir", "dear ma", "i am writing", "hiring team",
        "apply for", "sincerely", "regards"
    ],
    "certificate": [
        "certificate", "awarded", "completed",
        "certify", "this is to certify"
    ],
    "invoice": [
        "invoice", "total amount", "gst", "billing", "payment due"
    ],
    "marksheet / result": [
        "result", "marks obtained", "pass", "fail",
        "grade card", "marksheet"
    ],
}


def detect_doc_type(text):
    text_lower = text.lower()
    scores = {}
    for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
        scores[doc_type] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "document"


def extract_company_name(chunks):
    """
    Find the company this document is FOR.
    Priority: text patterns like 'apply...at X', 'position at X', 'with X'
    before falling back to address-block scanning.
    Address lines (city, state, road etc.) are strictly excluded.
    """
    all_text = " ".join([c.page_content for c in chunks[:3]])

    # These are city/state/address words — never a company name
    ADDRESS_WORDS = {
        "road","street","nagar","colony","pune","mumbai","delhi","bangalore",
        "hyderabad","india","maharastra","telangana","rajasthan","jaipur",
        "mp","baner","pashan","gachibowli","adhartal","jbp","suhagi",
        "divyasree","orion","madhura","sector","phase","plot","flat",
        "block","lane","avenue","park","near","opposite","behind","above",
        "floor","building","tower","complex","centre","center","plaza",
        "highway","side","new delhi","noida","gurgaon","gurugram","chandigarh",
        "lucknow","ahmedabad","surat","vadodara","kolkata","chennai","kochi",
    }

    def is_address(text):
        tl = text.lower()
        word_hits = sum(1 for w in ADDRESS_WORDS if w in tl)
        return word_hits >= 1 or bool(re.search(r"\d", tl))

    # Strategy 1 — "applying for <role> at <Company>"
    m = re.search(
        r"applying\s+(?:for\s+[\w\s]{0,50}?\s+)?at\s+([A-Z][A-Za-z0-9 &.,-]{2,50}?)(?:\.|,|\n|$)",
        all_text
    )
    if m:
        candidate = m.group(1).strip().rstrip(".,")
        if not is_address(candidate):
            return candidate

    # Strategy 2 — "position/role/job at <Company>"
    m = re.search(
        r"(?:position|role|job|opportunity)\s+(?:with|at)\s+([A-Z][A-Za-z0-9 &.,-]{2,50}?)(?:\.|,|\n|\s{2}|$)",
        all_text
    )
    if m:
        candidate = m.group(1).strip().rstrip(".,")
        if not is_address(candidate):
            return candidate

    # Strategy 3 — "writing this application to/with <Company>"
    m = re.search(
        r"writing\s+(?:this\s+)?(?:application|letter)\s+to\s+"
        r"(?:the\s+)?(?:[\w\s]{0,30}?\s+)?(?:position\s+(?:with|at)\s+)?"
        r"([A-Z][A-Za-z0-9 &]{2,40})(?:\.|,|\s|$)",
        all_text
    )
    if m:
        candidate = m.group(1).strip().rstrip(".,")
        if not is_address(candidate):
            return candidate

    # Strategy 4 — address block scan: lines just BEFORE "Dear Sir/Ma'am"
    lines = [l.strip() for l in all_text.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        if re.search(r"dear\s+(sir|ma|hiring|team)", line.lower()):
            candidates = lines[max(0, i - 6):i]
            for candidate in reversed(candidates):
                if is_address(candidate):
                    continue
                if re.search(r"@|\+91|phone|email|the hiring team", candidate.lower()):
                    continue
                # Must look like a proper noun (starts with capital, no all-caps address noise)
                if re.match(r"^[A-Z][A-Za-z0-9 &.,-]{2,55}$", candidate):
                    return candidate.strip()

    return None


def extract_person_name(chunks):
    """
    Find the candidate's name.
    A name is: 2-4 words, each Title-Cased, appears in first 8 lines,
    NOT a job title, location, email, phone, or section header.
    """
    JOB_TITLE_WORDS = {
        "engineer","analyst","developer","manager","consultant","officer",
        "executive","director","specialist","associate","assistant","intern",
        "data","software","business","marketing","finance","hr","sales",
        "fresher","trainee","lead","head","senior","junior","product",
    }
    SKIP_PATTERNS = re.compile(
        r"@|\+|\d{4,}|dear|http|resume|curriculum|cv |skills|education|"
        r"experience|objective|summary|profile|projects|certif|reference",
        re.IGNORECASE
    )

    for chunk in chunks[:3]:
        lines = [l.strip() for l in chunk.page_content.splitlines() if l.strip()]
        for line in lines[:8]:
            if SKIP_PATTERNS.search(line):
                continue
            # Must be 2-4 words, each starting with capital letter
            words = line.split()
            if not (2 <= len(words) <= 4):
                continue
            all_title = all(re.match(r"^[A-Z][a-zA-Z'-]+$", w) for w in words)
            if not all_title:
                continue
            # Must NOT be a job title
            lower_words = {w.lower() for w in words}
            if lower_words & JOB_TITLE_WORDS:
                continue
            return line

    return None


def extract_skills(chunks):
    """
    Scan ALL chunks for a Skills section header, then collect
    comma-separated or bullet skill items until the next section.
    """
    all_text = " ".join([c.page_content for c in chunks])
    lines = all_text.splitlines()

    SECTION_HEADERS = re.compile(
        r"^(education|experience|projects?|certifications?|objective|"
        r"summary|work history|internship|achievements?|interests?|"
        r"references?|languages?|hobbies|declaration)\s*[:\-]?\s*$",
        re.IGNORECASE
    )
    SENTENCE_WORDS = re.compile(
        r"\b(i am|i have|i believe|please|dear|regards|thank you|writing|"
        r"applying|looking forward|opportunity|contribute|appreciate|"
        r"enclosed|attached|sincerely|yours|truly)\b",
        re.IGNORECASE
    )

    technical = []
    interpersonal = []
    mode = None  # "technical" or "interpersonal"

    for line in lines:
        line = line.strip()
        if not line:
            continue
        ll = line.lower()

        # Detect "Technical Skills" header
        if re.match(r"^technical\s+skills?\s*[:\-]?\s*$", ll):
            mode = "technical"
            continue
        # Detect "Interpersonal Skills" / "Soft Skills" header
        if re.match(r"^(interpersonal|soft)\s+skills?\s*[:\-]?\s*$", ll):
            mode = "interpersonal"
            continue
        # Detect plain "Skills" header
        if re.match(r"^skills?\s*[:\-]?\s*$", ll):
            mode = "technical"
            continue

        # Also handle inline format: "Technical Skills: Python, MySQL, ..."
        inline = re.match(
            r"^(technical\s+skills?|interpersonal\s+skills?|skills?)\s*[:\-]\s*(.+)$",
            line, re.IGNORECASE
        )
        if inline:
            kind = "interpersonal" if "interpersonal" in inline.group(1).lower() else "technical"
            items = [s.strip() for s in re.split(r"[,;]", inline.group(2)) if s.strip()]
            items = [i for i in items if not SENTENCE_WORDS.search(i) and len(i) < 50]
            if kind == "interpersonal":
                interpersonal.extend(items)
            else:
                technical.extend(items)
            mode = kind
            continue

        if mode is None:
            continue

        # Stop at next section header
        if SECTION_HEADERS.match(ll):
            mode = None
            continue

        # Stop if this looks like a sentence (cover letter paragraph)
        if SENTENCE_WORDS.search(ll):
            mode = None
            continue

        # Collect skill items
        items = [s.strip() for s in re.split(r"[,;•\-]", line) if s.strip()]
        items = [i for i in items if 1 < len(i) < 50 and not SENTENCE_WORDS.search(i)]
        if items:
            if mode == "interpersonal":
                interpersonal.extend(items)
            else:
                technical.extend(items)

    parts = []
    if technical:
        parts.append("Technical Skills:\n• " + "\n• ".join(technical))
    if interpersonal:
        parts.append("Interpersonal Skills:\n• " + "\n• ".join(interpersonal))
    if parts:
        return "\n\n".join(parts)
    return None


def extract_education(chunks):
    all_text = " ".join([c.page_content for c in chunks])
    lines = all_text.splitlines()
    edu_lines = []
    capture = False
    STOP = re.compile(
        r"^(experience|skills?|projects?|certif|objective|summary|"
        r"work history|internship|achievements?|references?)\s*[:\-]?\s*$",
        re.IGNORECASE
    )
    for line in lines:
        line = line.strip()
        if not line:
            continue
        ll = line.lower()
        if re.match(r"^education\s*[:\-]?\s*$", ll):
            capture = True
            continue
        if capture:
            if STOP.match(ll):
                break
            edu_lines.append(line)
        if len(edu_lines) > 12:
            break
    return "\n".join(edu_lines) if edu_lines else None


def extract_experience(chunks):
    all_text = " ".join([c.page_content for c in chunks])
    lines = all_text.splitlines()
    exp_lines = []
    capture = False
    STOP = re.compile(
        r"^(education|skills?|projects?|certif|objective|summary|"
        r"references?|achievements?|interests?)\s*[:\-]?\s*$",
        re.IGNORECASE
    )
    for line in lines:
        line = line.strip()
        if not line:
            continue
        ll = line.lower()
        if re.match(r"^(experience|work history|internship|employment)\s*[:\-]?\s*$", ll):
            capture = True
            continue
        if capture:
            if STOP.match(ll):
                break
            exp_lines.append(line)
        if len(exp_lines) > 18:
            break
    return "\n".join(exp_lines) if exp_lines else None


# ==================== CACHED MODELS ====================

@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=Config.LOCAL_EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


@st.cache_resource(show_spinner="Loading LLM (first time only)...")
def get_local_llm(model_name):
    try:
        if 't5' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
            )
            pipe = pipeline(
                "text2text-generation", model=model, tokenizer=tokenizer,
                max_length=512, do_sample=False
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
            )
            pipe = pipeline(
                "text-generation", model=model, tokenizer=tokenizer,
                max_new_tokens=512, do_sample=False
            )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.warning(f"Could not load {model_name}, falling back to flan-t5-small. Error: {e}")
        return get_local_llm("google/flan-t5-small")


@st.cache_resource(show_spinner="Connecting to Hugging Face API...")
def get_api_llm(model_name, hf_token):
    if not hf_token:
        raise ValueError("HF_TOKEN required for API mode.")
    return HuggingFaceEndpoint(
        repo_id=model_name,
        huggingfacehub_api_token=hf_token,
        temperature=0.1,
        max_new_tokens=512,
        top_p=0.95
    )


# ==================== DOCUMENT REVIEWER ====================

class HuggingFaceDocumentReviewer:
    def __init__(self, mode='local', model_name="google/flan-t5-small", hf_token=""):
        self.mode = mode
        self.vectorstore = None
        self.documents = []
        self.processed_chunks = []
        self.embeddings = get_embeddings()
        self.llm = get_local_llm(model_name) if mode == 'local' else get_api_llm(model_name, hf_token)

    def load_pdf(self, pdf_file):
        documents = []
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(pdf_file.getvalue())
                tmp_path = tmp.name

            reader = PdfReader(tmp_path)
            total = len(reader.pages)
            # Track truly duplicate pages using hash of full content
            seen_hashes = set()

            for i, page in enumerate(reader.pages):
                text = None
                try:
                    text = page.extract_text()
                except Exception:
                    pass

                # Skip blank pages only
                if not text or not text.strip():
                    continue

                # Clean extraction noise
                text = re.sub(r"\x00", " ", text)
                text = re.sub(r"[ \t]{3,}", " ", text)
                text = re.sub(r"\n{5,}", "\n\n", text)
                text = text.strip()

                # Deduplicate only exact duplicates (same hash)
                # Use full text hash — not a prefix — so novel chapters don't get dropped
                page_hash = hash(text)
                if page_hash in seen_hashes:
                    continue
                seen_hashes.add(page_hash)

                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": pdf_file.name,
                        "page": i + 1,
                        "total_pages": total
                    }
                ))

        except Exception as e:
            st.error(f"Error loading {pdf_file.name}: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        if not documents:
            st.warning(
                f"No text found in {pdf_file.name}. "
                "If this is a scanned/image PDF, text cannot be extracted. "
                "Open the PDF and try selecting text — if you can't, it's image-based."
            )
        else:
            st.success(f"Loaded {len(documents)} pages from {pdf_file.name}")
        return documents

    def process_documents(self, uploaded_files):
        all_docs = []
        failed = []

        for f in uploaded_files:
            docs = self.load_pdf(f)
            if docs:
                all_docs.extend(docs)
            else:
                failed.append(f.name)

        if failed:
            st.warning(f"Could not extract text from: {', '.join(failed)}. "
                       "These may be scanned/image PDFs.")

        if not all_docs:
            st.error("No text could be extracted from any document.")
            return False

        # Detect document type to set smarter chunk size
        sample_text = " ".join([d.page_content for d in all_docs[:3]])
        is_technical = bool(re.search(
            r"(theorem|equation|regression|algorithm|formula|"
            r"hypothesis|coefficient|variable|function|matrix|"
            r"derivative|integral|probability|statistics)",
            sample_text, re.IGNORECASE
        ))

        # Larger chunks for technical/academic docs to preserve context
        chunk_size = 800 if is_technical else Config.CHUNK_SIZE
        chunk_overlap = 150 if is_technical else Config.CHUNK_OVERLAP

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        raw_chunks = splitter.split_documents(all_docs)

        # Deduplicate
        seen = set()
        unique_chunks = []
        for chunk in raw_chunks:
            key = chunk.page_content.strip().lower()[:300]
            if key not in seen:
                seen.add(key)
                unique_chunks.append(chunk)
        self.processed_chunks = unique_chunks

        doc_type_label = "technical/academic" if is_technical else "standard"
        with st.spinner(f"Building search index ({doc_type_label} mode, {len(unique_chunks)} chunks)..."):
            self.vectorstore = FAISS.from_documents(self.processed_chunks, self.embeddings)

        self.documents = all_docs
        return True

    def summarize_documents(self):
        if not self.processed_chunks:
            return "No documents to summarize."
        try:
            text = " ".join([c.page_content for c in self.processed_chunks[:5]])
            prompt = SUMMARY_PROMPT_TEXT.format(text=text)
            return self.llm.invoke(prompt)
        except Exception as e:
            return f"Error generating summary: {e}"

    def ask_question(self, question):
        if not self.vectorstore:
            return {"answer": "Please upload and process documents first.", "sources": []}

        q_lower = question.lower()

        # Smart: company extraction
        company_triggers = [
            "which company", "what company", "company is this",
            "written to", "addressed to", "applied to", "applying to",
            "which organisation", "what organisation", "which organization",
            "this cv is for", "this resume is for", "this letter is for",
            "who is this for", "for which company", "target company",
            "which firm", "what firm", "for which"
        ]
        if any(t in q_lower for t in company_triggers):
            company = extract_company_name(self.processed_chunks)
            if company:
                return {
                    "answer": "This document is addressed to: **" + company + "**",
                    "sources": []
                }
            else:
                return {
                    "answer": "Could not detect the company. Try: 'Who is the hiring team addressed to?'",
                    "sources": []
                }

        # Smart: person/candidate name
        name_triggers = [
            "whose cv", "whose resume", "candidate name", "person name",
            "who is this cv", "who is this resume", "applicant name",
            "what is the name", "name of the person", "name of candidate"
        ]
        if any(t in q_lower for t in name_triggers):
            name = extract_person_name(self.processed_chunks)
            if name:
                return {"answer": "The candidate name is: **" + name + "**", "sources": []}
            else:
                return {"answer": "Could not detect the person name.", "sources": []}

        # Smart: skills extraction
        skills_triggers = [
            "what skills", "list skills", "technical skills", "skills does",
            "skills of", "what are the skills", "interpersonal skills",
            "skills mentioned", "all skills", "skill set"
        ]
        if any(t in q_lower for t in skills_triggers):
            skills = extract_skills(self.processed_chunks)
            if skills:
                return {"answer": "Skills found in document:\n\n" + skills, "sources": []}
            else:
                return {"answer": "No skills section found in the document.", "sources": []}

        # Smart: education extraction
        edu_triggers = [
            "education", "qualification", "degree", "university", "college",
            "gpa", "graduation", "b.tech", "studied", "academic"
        ]
        if any(t in q_lower for t in edu_triggers):
            edu = extract_education(self.processed_chunks)
            if edu:
                return {"answer": "Education details:\n\n" + edu, "sources": []}

        # Smart: experience extraction
        exp_triggers = [
            "experience", "work history", "internship", "job", "worked at",
            "employment", "projects", "previous role"
        ]
        if any(t in q_lower for t in exp_triggers):
            exp = extract_experience(self.processed_chunks)
            if exp:
                return {"answer": "Experience / Work History:\n\n" + exp, "sources": []}

        # Smart: document type detection
        doc_type_triggers = [
            "what is this", "what document", "type of document",
            "what kind", "whose admit", "admit card",
            "is this a", "what is the document"
        ]
        if any(t in q_lower for t in doc_type_triggers):
            all_text = " ".join([c.page_content for c in self.processed_chunks[:3]])
            detected = detect_doc_type(all_text)
            name_answer = ""
            for chunk in self.processed_chunks[:3]:
                for line in chunk.page_content.splitlines():
                    if any(kw in line.lower() for kw in ["name:", "candidate", "student name", "applicant"]):
                        name_answer = line.strip()
                        break
                if name_answer:
                    break
            answer = "This appears to be a " + detected + "."
            if name_answer:
                answer += " " + name_answer
            return {"answer": answer, "sources": []}

        # Normal RAG flow
        raw_docs = self.vectorstore.similarity_search(question, k=Config.RETRIEVAL_K * 3)
        seen = set()
        relevant_docs = []
        for doc in raw_docs:
            key = doc.page_content.strip().lower()[:300]
            if key not in seen:
                seen.add(key)
                relevant_docs.append(doc)
            if len(relevant_docs) >= Config.RETRIEVAL_K:
                break

        context = "\n\n".join([
            "[Source: " + d.metadata.get('source', '?') + ", Page " +
            str(d.metadata.get('page', '?')) + "]\n" + d.page_content
            for d in relevant_docs
        ])

        prompt = QA_PROMPT.format(context=context, question=question)

        try:
            answer = self.llm.invoke(prompt)
        except Exception as e:
            answer = "Error: " + str(e)

        return {
            "answer": answer,
            "sources": [
                {
                    "file": d.metadata.get('source', '?'),
                    "page": d.metadata.get('page', '?'),
                    "text": d.page_content[:200] + "..."
                }
                for d in relevant_docs
            ]
        }


# ==================== STREAMLIT UI ====================

def create_streamlit_app():
    st.set_page_config(
        page_title="Document Review System",
        page_icon="Review system",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .stApp { background-color: #1a1a2e; color: #e0e0e0; }
    .main-header {
        text-align: center;
        padding: 1.2rem 2rem;
        background: linear-gradient(135deg, #0f3460 0%, #533483 100%);
        color: #ffffff;
        border-radius: 12px;
        margin-bottom: 2rem;
        font-size: 1.6rem;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }
    .success-box {
        padding: 1.2rem 1.5rem;
        background-color: #0d3b2e;
        border-left: 5px solid #00c896;
        border-radius: 8px;
        margin: 1rem 0;
        color: #d4f5ec;
        font-size: 1rem;
        line-height: 1.7;
    }
    .info-box {
        padding: 0.9rem 1.2rem;
        background-color: #0f2a4a;
        border-left: 5px solid #3b9eff;
        border-radius: 8px;
        margin: 1rem 0;
        color: #cce4ff;
        font-size: 0.95rem;
    }
    section[data-testid="stSidebar"] { background-color: #16213e; }
    section[data-testid="stSidebar"] * { color: #d0d8f0 !important; }
    .stTabs [data-baseweb="tab"] { color: #a0aec0; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #63b3ed !important; border-bottom: 3px solid #63b3ed; }
    div[data-testid="metric-container"] {
        background-color: #0f2a4a; border-radius: 10px;
        padding: 0.8rem; border: 1px solid #1e3a5f;
    }
    div[data-testid="metric-container"] label { color: #90cdf4 !important; }
    div[data-testid="metric-container"] div   { color: #ffffff !important; }
    .stButton > button {
        background: linear-gradient(135deg, #0f3460, #533483);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 0.5rem 1.2rem;
    }
    .stButton > button:hover { opacity: 0.85; }
    .stTextInput > div > div > input {
        background-color: #0f2a4a; color: #e2e8f0;
        border: 1px solid #2d4a7a; border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div class='main-header'> Document Review System</div>",
        unsafe_allow_html=True
    )

    for key, val in [('reviewer', None), ('documents_processed', False), ('question', '')]:
        if key not in st.session_state:
            st.session_state[key] = val

    with st.sidebar:
        st.header("⚙️ Configuration")
        mode = st.radio("Mode", ["local", "api"])

        hf_token = ""
        if mode == "api":
            hf_token = st.text_input("Hugging Face API Token", type="password")
            if not hf_token:
                st.warning("Enter your HF token.")

        local_models = {
            "google/flan-t5-small": "Fastest 300MB",
            "google/flan-t5-base":  "Balanced 990MB",
            "google/flan-t5-large": "Better 2.5GB",
            "microsoft/phi-2":      "Best local 2.7GB"
        }
        api_models = {
            "mistralai/Mistral-7B-Instruct-v0.2": "Best free",
            "google/flan-t5-xxl":                 "Large T5",
            "meta-llama/Llama-2-7b-chat-hf":      "Llama 2"
        }
        model_options = local_models if mode == "local" else api_models
        selected_model = st.selectbox(
            "Model", list(model_options.keys()),
            format_func=lambda x: x + " - " + model_options[x]
        )
        st.caption("Model loads once and stays cached.")

        st.divider()
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files", type=['pdf'], accept_multiple_files=True
        )

        if uploaded_files:
            if len(uploaded_files) > Config.MAX_DOCUMENTS:
                st.warning("Max " + str(Config.MAX_DOCUMENTS) + " files allowed.")
            elif st.button("Process Documents", type="primary"):
                try:
                    reviewer = HuggingFaceDocumentReviewer(
                        mode=mode, model_name=selected_model, hf_token=hf_token
                    )
                    if reviewer.process_documents(uploaded_files):
                        st.session_state.reviewer = reviewer
                        st.session_state.documents_processed = True
                        st.success(str(len(uploaded_files)) + " document(s) ready!")
                    else:
                        st.error("Processing failed.")
                except Exception as e:
                    st.error("Error: " + str(e))

    if st.session_state.documents_processed and st.session_state.reviewer:
        reviewer = st.session_state.reviewer
        tab1, tab2, tab3 = st.tabs(["Summary", "Ask Questions", "Info"])

        with tab1:
            st.header("Document Summary")
            st.markdown(
                "<div class='info-box'>Generate a summary of all uploaded documents.</div>",
                unsafe_allow_html=True
            )
            if st.button("Generate Summary", use_container_width=True):
                with st.spinner("Summarising..."):
                    summary = reviewer.summarize_documents()
                st.markdown(
                    "<div class='success-box'>" + summary + "</div>",
                    unsafe_allow_html=True
                )

        with tab2:
            st.header("Ask Questions")
            st.markdown(
                "<div class='info-box'>Ask anything about your documents.</div>",
                unsafe_allow_html=True
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Main topics"):
                    st.session_state.question = "What are the main topics discussed?"
            with col2:
                if st.button("Key points"):
                    st.session_state.question = "Summarize the key points"

            question = st.text_input(
                "Your question:",
                value=st.session_state.question,
                placeholder="e.g., Which company is this CV for?"
            )

            if st.button("Ask", type="primary", use_container_width=True) and question:
                with st.spinner("Finding answer..."):
                    result = reviewer.ask_question(question)
                st.markdown("### Answer")
                st.markdown(
                    "<div class='success-box'>" + result['answer'] + "</div>",
                    unsafe_allow_html=True
                )
                if result['sources']:
                    with st.expander("Sources"):
                        for i, s in enumerate(result['sources'], 1):
                            st.markdown("**" + str(i) + ". " + s['file'] + "** - Page " + str(s['page']))
                            st.caption(s['text'])
                            st.divider()

        with tab3:
            st.header("Document Info")
            c1, c2, c3 = st.columns(3)
            c1.metric("Pages", len(reviewer.documents))
            c2.metric("Chunks", len(reviewer.processed_chunks))
            c3.metric("Mode", "Local" if reviewer.mode == 'local' else "API")
            if reviewer.documents:
                st.markdown("### Files processed")
                sources = set(d.metadata.get('source', '?') for d in reviewer.documents)
                for src in sources:
                    pages = sum(1 for d in reviewer.documents if d.metadata.get('source') == src)
                    st.markdown("- **" + src + "** - " + str(pages) + " pages")

    else:
        st.markdown("""
        <div style='text-align:center; padding:3rem; color:#a0aec0;'>
            <h2 style='color:#e2e8f0;'>Welcome!</h2>
            <p style='font-size:1.1rem;'>Upload PDF documents in the sidebar to get started.</p>
            <br>
        
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    create_streamlit_app()