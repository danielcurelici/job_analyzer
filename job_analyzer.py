from openai import OpenAI
import streamlit as st
import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, computed_field, field_validator
import instructor
from groq import Groq
from dotenv import load_dotenv
import json
from pydantic import ValidationError
import traceback


SENIORITY_MAPPING = {
    # INTERN
    "intern": "Intern",
    "internship": "Intern",
    "trainee": "Intern",
    "stagiar": "Intern",

    # JUNIOR
    "junior": "Junior",
    "jr": "Junior",
    "entry": "Junior",
    "entry-level": "Junior",
    "entry level": "Junior",
    "beginner": "Junior",
    "associate": "Junior",

    # MID
    "mid": "Mid",
    "mid-level": "Mid",
    "mid level": "Mid",
    "intermediate": "Mid",
    "regular": "Mid",

    # SENIOR
    "senior": "Senior",
    "sr": "Senior",
    "senior-level": "Senior",
    "expert": "Senior",
    "specialist": "Senior",

    # LEAD
    "lead": "Lead",
    "team lead": "Lead",
    "tech lead": "Lead",
    "principal": "Lead",

    # ARCHITECT
    "architect": "Architect",
    "solution architect": "Architect",
    "software architect": "Architect",
}

# ==============================================================================
# 1. SETUP & SECURITATE
# ==============================================================================
st.set_page_config(page_title="GenAI Headhunter", page_icon="🕵️", layout="wide")

# Încărcăm variabilele din fișierul .env
load_dotenv()

# Încercăm să luăm cheia din OS (local) sau din Streamlit Secrets (cloud)
groq_api_key = os.getenv("GROQ_API_KEY")
op_api_key = os.getenv("OPENROUTER_API_KEY")

# Fallback pentru Streamlit Cloud deployment
if not groq_api_key and "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]

if not op_api_key and "OPENROUTER_API_KEY" in st.secrets:
    op_api_key = st.secrets["OPENROUTER_API_KEY"]


# Validare critică: Dacă nu avem cheie, oprim aplicația aici.
if not groq_api_key or not op_api_key:
    st.error("⛔ EROARE CRITICĂ: Lipsește `GROQ_API_KEY` sau `OPENROUTER_API_KEY`.")
    st.info("Te rog creează un fișier `.env` în folderul proiectului și adaugă: GROQ_API_KEY=cheia_ta_aici și OPENROUTER_API_KEY=cheia_ta_aici")
    st.stop()

# Configurare Client Groq Global (pentru a nu-l reinițializa constant)
groq_client = instructor.from_groq(Groq(api_key=groq_api_key), mode=instructor.Mode.JSON)
op_client = instructor.from_openai(OpenAI(api_key=op_api_key, base_url="https://openrouter.ai/api/v1"), mode=instructor.Mode.JSON)


# Sidebar Informativ (Fără input de date sensibile)
with st.sidebar:
    st.header("🕵️ GenAI Headhunter")
    st.success("✅ API Key încărcat securizat")
    st.markdown("---")
    st.write("Acest tool demonstrează:")
    st.write("• Web Scraping (BS4)")
    st.write("• Secure Env Variables")
    st.write("• Structured Data (Pydantic)")


# ==============================================================================
# 2. DATA MODELS (PYDANTIC SCHEMAS)
# ==============================================================================
class Location(BaseModel):
    city: Optional[str] = Field(None, description="Orașul")
    country: Optional[str] = Field(None, description="Țara")

class Red_flags(BaseModel):
    severity: Optional[Literal["low", "medium", "high"]] = Field(None, description="Nivelul de severitate al semnalului (ex: low, medium, high)")    
    category: Optional[Literal["toxicity", "vagueness", "unrealistic", "stress"]] = Field(None, description="Categoria semnalului de alarmă, Poate inseamnă că anunțul este neclar, ambiguu sau generic. Sau inseamnă că cerințele sau oferta sunt nerealiste sau disproporționate.")
  

class RawExtraction(BaseModel):
    role_title: str = Field(..., description="Titlul jobului standardizat")
    company_name: str = Field(..., description="Numele companiei")
    
    seniority: str = Field(..., description="Nivelul de experiență dedus")
    @field_validator("seniority", mode="before")
    @classmethod
    def normalize_seniority(cls, v):
        if not isinstance(v, str):
            return "Mid"

        key = v.strip().lower()
        return SENIORITY_MAPPING.get(key, "Mid")
    
    match_score: int = Field(..., ge=0, le=100, description="Scor 0-100: Calitatea descrierii jobului")
    tech_stack: List[str] = Field(..., description="Listă cu tehnologii specifice (ex: Python, AWS, React)")
    red_flags: List[Red_flags] = Field(..., description="Lista de semnale de alarmă (toxicitate, stres, vaguitate)")
    summary: str = Field(..., description="Un rezumat scurt al rolului (max 2 fraze) în limba română")
    is_remote: bool = Field(False, description="True dacă jobul este remote sau hibrid")
    SalaryRange: Optional[str] = Field(
        default=None, 
        description='Interval salarial dacă este menționat (ex: "1000-5000 EUR")',
        json_schema_extra={"min_sal": 500, "max_sal": 5000, "currency": "EUR"},)
    location: Optional[Location] = Field(None, description="Locația fizică a jobului dacă este specificată (ex: București, Cluj, etc.)")  


    @computed_field
    @property
    def is_hybrid(self) -> bool:
        return self.is_remote and self.location is not None 

class FieldValidation(BaseModel):
    field: str
    status: Literal["ok", "warning", "error"]
    issues: List[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    fields: List[FieldValidation]
    overall_status: Literal["consistent", "minor_issues", "inconsistent"]
    confidence: int = Field(..., ge=0, le=100)

class StrategicAdvice(BaseModel):
    # 1️⃣ Potrivire cu piața (RO)
    market_fit_summary: str = Field(..., description="Evaluare neutră a anunțului: cât de bine se aliniază cu piața IT din România și unde poate fi îmbunătățit.")
    market_improvements_for_hr: List[str] = Field(default_factory=list, description="Sugestii concrete pentru HR ca să facă anunțul mai competitiv/clar (ex: claritate rol, cerințe, beneficii, limbaj).")
    
    # 2️⃣ Întrebări pentru formular de pre-screening (către candidat)
    pre_screening_form_knockout_questions: List[str] = Field(default_factory=list, description="Întrebări eliminatorii/scurte pentru formular (eligibility, disponibilitate, cerințe must-have, salariu, remote/hybrid).")
    pre_screening_form_technical_questions: List[str] = Field(default_factory=list, description="Întrebări tehnice pentru formular (răspuns scurt/multiple-choice) bazate pe cerințele din anunț.")
    pre_screening_form_behavioral_questions: List[str] = Field(default_factory=list, description="Întrebări comportamentale pentru formular (răspuns scurt) relevante pentru rol.")
    
    # 3️⃣ Negociere salariu (către HR)
    salary_negotiation_tips_for_hr: List[str] = Field(default_factory=list, description="Recomandări pentru HR: cum să poziționeze oferta, ce să clarifice, ce compromisuri sunt uzuale în România.")

# ==============================================================================
# 3. UTILS - SCRAPER (Colectare Date)
# ==============================================================================

def scrape_clean_job_text(url: str, max_chars: int = 3000) -> str:
    """
    Descarcă pagina și returnează un text curat, optimizat pentru contextul LLM.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return f"Error: Status code {response.status_code}"
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Eliminăm elementele inutile care consumă tokeni
        for junk in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            junk.decompose()
            
        # Extragem textul și eliminăm spațiile multiple
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)
        
        return text[:max_chars] 
        
    except Exception as e:
        return f"Scraping Error: {str(e)}"

# ==============================================================================
# 4. AI SERVICE LAYER (Logica LLM)
# ==============================================================================

def extract_job_with_ai(text: str) -> RawExtraction:
    """
    Trimite textul curățat către Groq și returnează obiectul structurat.
    """
    try:
        return groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_model=RawExtraction,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system", 
                "content": (
                    "Ești un Recruiter Expert în IT din România care lucrează într-o firmă de headhunting. "
                    "Identifică tehnologiile și potențialele probleme (red flags). "
                    "Răspunde strict în formatul cerut."
                )
            },
            {
                "role": "user", 
                "content": f"Analizează acest job description:\n\n{text}"
            }
        ],
        max_retries=2,
        temperature=0,
    )
    except ValidationError as ve:
        st.error("ValidationError (RawExtraction)")
        st.json(ve.errors())
        raise

    except Exception:
        st.error("Eroare neașteptată (Extractor)")
        st.code(traceback.format_exc())
        raise
def validate_extraction_with_ai(original_text: str, extraction: RawExtraction) -> ValidationReport:
    """
    Agent 3: The Validator
    Verifică consistența dintre textul original și output-ul Extractorului.
    """

    try:
        return groq_client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct-0905",
        response_model=ValidationReport,
        response_format={"type": "json_object"},
        temperature=0,
        max_retries=2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ești 'The Validator' într-un pipeline AI cu 3 agenți: Extractor, Validator, Counselor.\n\n"
                    "Rolul tău este să verifici CONSISTENȚA dintre textul original al jobului și JSON-ul extras.\n"
                    "Pentru fiecare câmp relevant (role_title, company_name, seniority, tech_stack, "
                    "is_remote, location, SalaryRange, summary, red_flags):\n"
                    "- Marchează status = 'ok' dacă este consistent.\n"
                    "- Marchează status = 'warning' dacă este parțial ambiguu.\n"
                    "- Marchează status = 'error' dacă este greșit sau inventat.\n\n"
                    "Adaugă issues DOAR dacă există probleme reale.\n"
                    "Nu inventa informații.\n"
                    "Returnează STRICT un ValidationReport valid JSON."
                )
            },
            {
                "role": "user",
                "content": (
                    "TEXT ORIGINAL:\n"
                    f"{original_text}\n\n"
                    "JSON EXTRAS:\n"
                    f"{extraction.model_dump_json(indent=2)}"
                )
            }
        ]
        
    )
    except ValidationError as ve:
        st.error("ValidationError (ValidationReport)")
        st.json(ve.errors())
        raise

    except Exception:
        st.error("Eroare neașteptată (Validator)")
        st.code(traceback.format_exc())
        raise

def strategic_advice_with_ai(extraction: RawExtraction) -> StrategicAdvice:
    try:
        return groq_client.chat.completions.create(
            model="groq/compound",  
            response_model=StrategicAdvice,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ești un Recruiter Expert în IT din România care lucrează într-o firmă de headhunting. "
                        "Analizezi anunțurile postate de HR cu scopul de a le îmbunătăți. "
                        "Returnează STRICT JSON valid care respectă schema StrategicAdvice. "
                        "Fără text în plus, fără markdown."
                        "Structură în 3 categorii: "
                        "1) potrivire a anunțului cu piața din România (și îmbunătățiri pentru HR), "
                        "2) întrebări pentru un formular de pre-screening completat de candidat (knockout + tehnic + comportamental), "
                        "3) recomandări pentru negociere salarială adresate HR."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Generează analiza pe baza acestui JSON:\n\n"
                        f"{extraction.model_dump_json(indent=2)}"
                    ),
                },
            ],
            max_retries=2,
            temperature=0.7
        )

    except ValidationError as ve:
        st.error("ValidationError (StrategicAdvice)")
        st.json(ve.errors())  # aici vezi exact câmpul care lipsește / e greșit
        raise

    except Exception as e:
        st.error("Eroare neașteptată: Counselor")
        st.code(traceback.format_exc())
        raise
    
# ==============================================================================
# 5. UI - APLICAȚIA STREAMLIT
# ==============================================================================

st.title("🕵️ GenAI Headhunter Assistant")
st.markdown("Transformă orice Job Description într-o analiză structurată folosind AI.")

# Tab-uri
tab1, tab2 = st.tabs(["🚀 Analiză Job", "📊 Market Scan (Batch)"])

# --- TAB 1: ANALIZA UNUI SINGUR LINK ---
with tab1:
    st.subheader("Analizează un Job URL")
    url_input = st.text_input("Introdu URL-ul:", placeholder="https://...")
    
    if st.button("Analizează Job", key="btn_single"):
        if not url_input:
            st.warning("Te rugăm introdu un URL.")
        else:
            with st.spinner("🕷️ Scraping & 🤖 AI Analysis..."):
                raw_text = scrape_clean_job_text(url_input)
            
            if "Error" in raw_text:
                st.error(raw_text)
            else:
                try:
                    data = extract_job_with_ai(raw_text)
                    # st.json(data)  # PRINT
                    validation = validate_extraction_with_ai(raw_text, data)
                    #st.json(validation)  # PRINT
                    
                    if validation.overall_status == "inconsistent":
                        st.warning("⚠️ Validator a detectat inconsistențe. Se încearcă re-extragerea automată...")

                        # Retry extraction
                        data = extract_job_with_ai(raw_text)
                        validation = validate_extraction_with_ai(raw_text, data)

                        # Dacă tot e inconsistent după retry
                        if validation.overall_status == "inconsistent":
                            st.error("❌ Extraction rămâne inconsistentă după retry. Insight-urile pot fi afectate.")

                    elif validation.overall_status == "minor_issues":
                        st.info("ℹ️ Extraction are mici ambiguități.")


                    strategic_data = strategic_advice_with_ai(validation)
                    # st.json(strategic_data)  # PRINT

                    # -- DISPLAY --
                    st.divider()
                    col_h1, col_h2, col_h3 = st.columns([3, 2, 1])
                    with col_h1:
                        st.markdown(f"### {data.role_title}")
                        st.caption(f"Companie: **{data.company_name}** | Nivel: **{data.seniority}**")
                    with col_h2:
                        color = "normal" if validation.confidence > 70 else "inverse"
                        st.metric("Calitate AI", f"{validation.confidence}/100", delta_color=color)                    
                    with col_h3:
                        color = "normal" if data.match_score > 70 else "inverse"
                        st.metric("Calitate anunt", f"{data.match_score}/100", delta_color=color)


                
                    location_text = "N/A"

                    if data.location:
                        parts = []
                        if data.location.city:
                            parts.append(data.location.city)
                        if data.location.country:
                            parts.append(data.location.country)
                        if parts:
                            location_text = ", ".join(parts)

                    st.markdown("### 🧩 Overview")
                    c1, c2, c3 = st.columns([2, 1, 1])

                    with c1:
                        st.info(
                            f"""
                            **Mod lucru**  
                            - Remote: {'Da' if data.is_remote else 'Nu'}  
                            - Hybrid: {'Da' if data.is_hybrid else 'Nu'}  
                            - Locație: {location_text}
                            """
                        )

                    with c2:
                        st.success(f"**Tehnologii**: {len(data.tech_stack)}")
                        st.info(f"**Interval salarial**: {data.SalaryRange or 'N/A'}")

                    with c3:
                        rf_count = len(data.red_flags)

                        if rf_count == 0:
                            st.success("**Red Flags**: 0")
                        else:
                            content = [f"### Red Flags: {rf_count}"]

                            for rf in data.red_flags:
                                if not rf.category:
                                    continue
                                category_label = rf.category.replace("_", " ").title()
                                severity_label = (rf.severity or "N/A").title()
                                content.append(f"• **{category_label}** — severitate: **{severity_label}**")

                            st.error("\n\n".join(content))

                    st.markdown("### 🛠️ Tech Stack")

                    if data.tech_stack:
                        st.markdown(
                            " ".join(f"`{tech}`" for tech in data.tech_stack)
                        )
                    else:
                        st.caption("N/A")

                    st.markdown(f"**📝 Rezumat job:** {data.summary}")
                    st.markdown(f"**📝 Aliniere cu piata din Romania:** {strategic_data.market_fit_summary}")

                    st.divider()
                    st.markdown("## 🧾 Formular pre-screening (pentru candidat)")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("### 🚫 Knockout")
                        qs = strategic_data.pre_screening_form_knockout_questions
                        if qs:
                            st.info("\n\n".join(f"**{i+1}.** {q}" for i, q in enumerate(qs)))
                        else:
                            st.caption("N/A")

                    with col2:
                        st.markdown("### 🛠️ Tehnic")
                        qs = strategic_data.pre_screening_form_technical_questions
                        if qs:
                            st.info("\n\n".join(f"**{i+1}.** {q}" for i, q in enumerate(qs)))
                        else:
                            st.caption("N/A")

                    with col3:
                        st.markdown("### 🧠 Comportamental")
                        qs = strategic_data.pre_screening_form_behavioral_questions
                        if qs:
                            st.info("\n\n".join(f"**{i+1}.** {q}" for i, q in enumerate(qs)))
                        else:
                            st.caption("N/A")

                    st.divider()
                    st.markdown("## 🧩 Recomandări pentru HR")

                    col_left, col_right = st.columns(2)

                    with col_left:
                        st.markdown("### 📈 Îmbunătățiri anunț")

                        imps = strategic_data.market_improvements_for_hr
                        if imps:
                            st.info("\n\n".join(f"• {x}" for x in imps))
                        else:
                            st.caption("Nu există sugestii.")

                    with col_right:
                        st.markdown("### 💰 Negociere salarială (HR)")
                        tips = strategic_data.salary_negotiation_tips_for_hr
                        if tips:
                            st.success("\n\n".join(f"• {t}" for t in tips))
                        else:
                            st.caption("N/A")

                except Exception as e:
                    st.error(f"Eroare AI: {str(e)}")

# --- TAB 2: BATCH PROCESSING ---
with tab2:
    st.subheader("📊 Compară mai multe joburi")
    urls_text = st.text_area("Paste URL-uri (unul pe linie):", height=200)
    
    if st.button("Scanează Piața", key="btn_batch"):
        urls = [u.strip() for u in urls_text.split('\n') if u.strip()]
        
        if not urls:
            st.warning("Nu ai introdus link-uri.")
        else:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, link in enumerate(urls):
                status_text.text(f"Analizez {i+1}/{len(urls)}...")
                text = scrape_clean_job_text(link)
                
                if "Error" not in text:
                    try:
                        res = extract_job_with_ai(text)
                        results.append({
                            "Role": res.role_title,
                            "Company": res.company_name,
                            "Seniority": res.seniority,
                            "Tech": res.tech_stack,
                            "Score": res.match_score
                        })
                    except:
                        pass # Continuăm chiar dacă unul crapă
                
                progress_bar.progress((i + 1) / len(urls))
            
            status_text.text("Gata!")
            
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df)
                
                # Grafic simplu
                st.bar_chart(df['Seniority'].value_counts())

                print()