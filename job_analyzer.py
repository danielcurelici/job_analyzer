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
api_key = os.getenv("GROQ_API_KEY")

# Fallback pentru Streamlit Cloud deployment
if not api_key and "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]

# Validare critică: Dacă nu avem cheie, oprim aplicația aici.
if not api_key:
    st.error("⛔ EROARE CRITICĂ: Lipsește `GROQ_API_KEY`.")
    st.info("Te rog creează un fișier `.env` în folderul proiectului și adaugă: GROQ_API_KEY=cheia_ta_aici")
    st.stop()

# Configurare Client Groq Global (pentru a nu-l reinițializa constant)
client = instructor.from_groq(Groq(api_key=api_key), mode=instructor.Mode.TOOLS)

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
    severity: Literal["low", "medium", "high"] = Field(None, description="Nivelul de severitate al semnalului (ex: low, medium, high)")    
    category: Literal["toxicity", "vagueness", "unrealistic", "stress"] = Field(None, description="Categoria semnalului de alarmă, Poate inseamnă că anunțul este neclar, ambiguu sau generic. Sau inseamnă că cerințele sau oferta sunt nerealiste sau disproporționate.")
  

class JobAnalysis(BaseModel):
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
    SalaryRange: Optional[str] = Field(min_sal = 500, max_sal = 5000, currency = "EUR", description="Interval salarial dacă este menționat (ex: 1000-5000 EUR)")
    location: Optional[Location] = Field(None, description="Locația fizică a jobului dacă este specificată (ex: București, Cluj, etc.)")  


    @computed_field
    @property
    def is_hybrid(self) -> bool:
        return self.is_remote and self.location is not None 

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

def analyze_job_with_ai(text: str) -> JobAnalysis:
    """
    Trimite textul curățat către Groq și returnează obiectul structurat.
    """
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_model=JobAnalysis,
        messages=[
            {
                "role": "system", 
                "content": (
                    "Ești un Recruiter Expert în IT. Analizează textul jobului cu obiectivitate. "
                    "Identifică tehnologiile și potențialele probleme (red flags). "
                    "Răspunde strict în formatul cerut."
                )
            },
            {
                "role": "user", 
                "content": f"Analizează acest job description:\n\n{text}"
            }
        ],
        temperature=0.1,
    )

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
                    data = analyze_job_with_ai(raw_text)
                    st.json(data)  # PRINT
                    # -- DISPLAY --
                    st.divider()
                    col_h1, col_h2 = st.columns([3, 1])
                    with col_h1:
                        st.markdown(f"### {data.role_title}")
                        st.caption(f"Companie: **{data.company_name}** | Nivel: **{data.seniority}**")
                    with col_h2:
                        color = "normal" if data.match_score > 70 else "inverse"
                        st.metric("Quality Score", f"{data.match_score}/100", delta_color=color)

                    # Detalii
                    c1, c2, c3 = st.columns(3)

                    location_text = "N/A"

                    if data.location:
                        parts = []
                        if data.location.city:
                            parts.append(data.location.city)
                        if data.location.country:
                            parts.append(data.location.country)
                        if parts:
                            location_text = ", ".join(parts)

                    c1.info(
                    f"""
                    **Mod lucru:**  
                    - Remote: {'Da' if data.is_remote else 'Nu'}  
                    - Hybrid: {'Da, in locatia de mai jos' if data.is_hybrid else 'Nu'}  
                    - Locație: {location_text}
                    """
)
                    c2.success(f"**Tehnologii:** {len(data.tech_stack)}")
                    c3.error(f"**Red Flags:** {len(data.red_flags)}")
                    c4, c5, c6 = st.columns(3)
                    c4.info(f"**Interval salarial:** {data.SalaryRange or 'N/A'}")


                    st.markdown(f"**📝 Rezumat:** {data.summary}")
                    st.markdown("#### 🛠️ Tech Stack")
                    st.write(", ".join([f"`{tech}`" for tech in data.tech_stack]))

                    if data.red_flags:
                        lines = []
                        for rf in data.red_flags:
                            if not rf.category:
                                continue
                            category_label = rf.category.replace("_", " ").title()
                            severity_label = (rf.severity or "N/A").title()
                            lines.append(f"- **{category_label}** — severitate: **{severity_label}**")

                        if lines:
                            st.warning("\n".join(lines))
                        

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
                        res = analyze_job_with_ai(text)
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