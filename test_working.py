import os
import json
import streamlit as st
import requests
import pandas as pd
import numpy as np
import re
import time
from collections import Counter
from packaging.version import Version
from dotenv import load_dotenv

load_dotenv() 

# ---------------- App config + options ----------------
st.set_page_config(page_title="Vaccine Pipeline Platform", page_icon="💉", layout="wide")

DEFAULT_LLM_MODEL = "mistralai/mistral-7b-instruct:free"
MAX_TRIALS_FOR_SUMMARY = 12
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Try to force the legacy dataframe serializer (ignored if not supported in your version)
try:
    st.set_option("global.dataFrameSerialization", "legacy")
except Exception:
    pass

def _supports_width_string() -> bool:
    """True if st.dataframe supports width='stretch'/'content' (Streamlit >= 1.39)."""
    try:
        return Version(st.__version__) >= Version("1.39.0")
    except Exception:
        return False

# ---------------- Utilities ----------------
def _norm_txt(s: str) -> str:
    """Normalize a text string (lowercase, alnum, single spaces)."""
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def _request_json(url, params=None, session=None, retries=3, timeout=30, headers=None):
    """GET JSON with basic retry + 429 handling."""
    s = session or requests.Session()
    for attempt in range(retries):
        try:
            resp = s.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "1"))
                time.sleep(wait + 0.5)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(1.2 * (2 ** attempt))
                continue
            raise

# ---------------- LLM helpers (OpenRouter) ----------------
def _get_secret(name: str):
    try:
        return st.secrets.get(name)  # type: ignore[attr-defined]
    except Exception:
        return None


def _get_openrouter_key():
    key = _get_secret("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None, "Set OPENROUTER_API_KEY in Streamlit secrets or environment."
    return key, None


def _call_openrouter(messages, model=None, max_tokens=900, temperature=0.25):
    api_key, err = _get_openrouter_key()
    if not api_key:
        return None, err

    payload = {
        "model": model or _get_secret("OPENROUTER_MODEL") or os.getenv("OPENROUTER_MODEL") or DEFAULT_LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://vaccine-pipeline.streamlit.app"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "Vaccine Pipeline Platform"),
    }

    try:
        resp = requests.post(OPENROUTER_BASE_URL, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {})
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
        return content or "", None
    except requests.HTTPError as e:
        try:
            err_json = resp.json()
            detail = err_json.get("error", {}).get("message") or err_json
        except Exception:
            detail = str(e)
        return None, f"OpenRouter request failed: {detail}"
    except Exception as e:
        return None, f"OpenRouter request failed: {e}"


def _summarize_trials_with_llm(records: list, context_instructions: str):
    """Summarize a list of trials using OpenRouter-backed LLM."""

    trials = records[:MAX_TRIALS_FOR_SUMMARY]
    if not trials:
        return None, "Nothing to summarize."

    bullet_lines = []
    for idx, t in enumerate(trials, start=1):
        parts = [
            f"{idx}. NCT ID: {t.get('NCT ID', 'N/A')}",
            f"Title: {t.get('Title', 'N/A')}",
            f"Phase: {t.get('Phase', 'Unknown')}",
            f"Status: {t.get('Status', 'Unknown')}",
            f"Sponsor: {t.get('Sponsor', 'Unknown')}",
            f"Vaccines: {t.get('Vaccines', 'Not reported')}"
        ]
        bullet_lines.append("; ".join(parts))

    system_prompt = (
        "You are a regulatory insights analyst. Provide concise, factual, executive-ready takeaways "
        "about vaccine clinical trials. Highlight phase mix, sponsor signals, risk items, and BD actions."
    )
    user_prompt = f"""
Context: {context_instructions}

Clinical trial snapshots:
{chr(10).join(bullet_lines)}

Deliver:
- 3-5 bullet insights on the landscape
- A short risk/consideration note
- Optional BD/medical affairs action prompts
"""

    return _call_openrouter(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )


def _summarize_single_trial(nct_id: str, details: dict):
    """Summarize one trial with OpenRouter-backed LLM."""

    payload = {
        "nct_id": nct_id,
        "title": details.get("Title"),
        "phase": details.get("Phase"),
        "status": details.get("Status"),
        "sponsor": details.get("Sponsor"),
        "vaccines": details.get("Vaccines"),
        "diseases": details.get("Diseases"),
        "outcomes": details.get("Outcomes"),
    }

    system_prompt = (
        "You are preparing a due-diligence brief for a single vaccine trial. "
        "Stay factual and reference the NCT ID in the opening."
    )
    user_prompt = f"""
Structured data:
{json.dumps(payload, indent=2)}

Produce:
- 2 sentence topline (phase, status, sponsor, target disease) mentioning NCT ID.
- Bullet list for vaccine product, population, outcomes.
- Note any missing data or pending info.
"""

    return _call_openrouter(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

def _extract_vaccine_names(study) -> list:
    """Collect intervention names + otherNames + armGroup interventionNames."""
    proto = (study or {}).get("protocolSection", {}) or {}
    arms = proto.get("armsInterventionsModule", {}) or {}
    names = set()

    for intr in arms.get("interventions", []) or []:
        nm = intr.get("name")
        if nm:
            names.add(nm.strip())
        other = intr.get("otherNames")
        if isinstance(other, list):
            for on in other:
                if on:
                    names.add(on.strip())

    for grp in arms.get("armGroups", []) or []:
        for nm in grp.get("interventionNames", []) or []:
            if nm:
                names.add(nm.strip())

    return sorted(names)

def _mesh_terms_intervention(study) -> list:
    """Return MeSH terms for interventions (used to detect vaccines)."""
    derived = (study or {}).get("derivedSection", {}) or {}
    iv_browse = derived.get("interventionBrowseModule", {}) or {}
    leaves = iv_browse.get("browseLeaves", []) or []
    return [x.get("meshTerm", "") for x in leaves if x.get("meshTerm")]

def _mesh_terms_condition(study) -> list:
    """Return MeSH terms for conditions (helps disease normalization)."""
    derived = (study or {}).get("derivedSection", {}) or {}
    cond_browse = derived.get("conditionBrowseModule", {}) or {}
    leaves = cond_browse.get("browseLeaves", []) or []
    return [x.get("meshTerm", "") for x in leaves if x.get("meshTerm")]

def _primary_outcomes_from_protocol(study) -> list:
    """Primary outcomes from protocol (exists even when no results posted)."""
    proto = (study or {}).get("protocolSection", {}) or {}
    out_mod = proto.get("outcomesModule", {}) or {}
    out = []
    for o in out_mod.get("primaryOutcomes", []) or []:
        out.append({
            "Title": o.get("measure") or o.get("title") or "",
            "Description": o.get("description", "") or o.get("timeFrame", "") or ""
        })
    return out

def _results_outcomes(study) -> list:
    """Outcomes from posted results (if available)."""
    results = (study or {}).get("resultsSection", {}) or {}
    res_mod = results.get("outcomeMeasuresModule", {}) or {}
    out = []
    for o in res_mod.get("outcomeMeasures", []) or []:
        out.append({
            "Title": o.get("title") or "",
            "Description": o.get("description", "") or ""
        })
    return out

def _is_vaccine_study(study) -> bool:
    """
    Classify study as vaccine:
    1) Intervention MeSH terms include 'Vaccine' (most reliable)
    2) Biological intervention with name/otherNames mentioning 'vaccine'
    3) Outcomes text contains immunogenicity/antibody keywords
    """
    mesh = [m.lower() for m in _mesh_terms_intervention(study)]
    if any("vaccine" in m for m in mesh):
        return True

    proto = (study or {}).get("protocolSection", {}) or {}
    arms = proto.get("armsInterventionsModule", {}) or {}
    intrs = arms.get("interventions", []) or []
    has_bio = any((i.get("type") or "").lower() == "biological" for i in intrs)
    has_vaccine_word = False
    for i in intrs:
        other = i.get("otherNames", [])
        if not isinstance(other, list):
            other = []
        txt = " ".join([
            _norm_txt(i.get("name", "")),
            _norm_txt(" ".join(other))
        ])
        if "vaccine" in txt or " vax " in f" {txt} ":
            has_vaccine_word = True
            break
    if has_bio and has_vaccine_word:
        return True

    po = _primary_outcomes_from_protocol(study)
    titles = _norm_txt(" ".join([(x.get("Title") or "") for x in po]))
    if any(k in titles for k in ["immunogenicity", "antibody", "neutralizing", "seroconversion"]):
        return True

    return False

# ---------------- Safe table rendering ----------------
def _cell_to_str(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, (list, tuple, set)):
        return ", ".join("" if (x is None or (isinstance(x, float) and pd.isna(x))) else str(x) for x in v)
    if isinstance(v, dict):
        return "; ".join(f"{k}: {v[k]}" for k in sorted(v.keys()))
    return str(v)

def df_to_arrow_utf8_table(df: pd.DataFrame):
    """Convert any DataFrame to a PyArrow Table with all columns utf8 strings."""
    import pyarrow as pa
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    data = {c: pa.array([_cell_to_str(v) for v in df[c].tolist()], type=pa.string()) for c in df.columns}
    return pa.table(data)

def df_normalize_str(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to string dtype for st.table or legacy paths."""
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    for c in df.columns:
        df[c] = df[c].map(_cell_to_str).astype("string[python]")
    df.reset_index(drop=True, inplace=True)
    return df

def show_df(df: pd.DataFrame, height: int = 420):
    """Render an interactive table robustly across Streamlit/Arrow versions."""
    safe_df = df_normalize_str(df)

    # 1) Prefer explicit Arrow Table with utf8 strings
    try:
        table = df_to_arrow_utf8_table(safe_df)
    except Exception as e:
        table = None

    # 2) Try new API (width='stretch'); then old API; then numeric width; then fallback to pandas; then st.table
    try:
        if table is not None:
            if _supports_width_string():
                st.dataframe(table, width="stretch", height=height)
            else:
                st.dataframe(table, use_container_width=True, height=height)
        else:
            if _supports_width_string():
                st.dataframe(safe_df, width="stretch", height=height)
            else:
                st.dataframe(safe_df, use_container_width=True, height=height)
    except Exception:
        try:
            # Try numeric width if this build insists on an int
            if table is not None:
                st.dataframe(table, width=1200, height=height)
            else:
                st.dataframe(safe_df, width=1200, height=height)
        except Exception as e2:
            st.warning(f"Interactive table failed: {e2}. Showing static table instead.")
            st.table(safe_df)

# ---------------- Data fetchers (cached) ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_vaccine_trials(disease: str, max_pages: int = 10):
    """
    Fetch vaccine trials by disease.
    Strategy: search by disease, page through, classify vaccines client-side.
    Fallback: if none found, add query.term=vaccin*.
    """
    url = "https://clinicaltrials.gov/api/v2/studies"
    all_results = []
    page_token = None
    page_count = 0
    seen = set()
    session = requests.Session()
    headers = {"User-Agent": "VaccinePipeline/1.0"}

    try:
        while page_count < max_pages:
            params = {"query.cond": disease, "pageSize": 100}
            if page_token:
                params["pageToken"] = page_token

            data = _request_json(url, params=params, session=session, headers=headers)
            studies = data.get("studies", []) or []
            if not studies:
                break

            for s in studies:
                if not _is_vaccine_study(s):
                    continue

                proto = s.get("protocolSection", {}) or {}
                ident = proto.get("identificationModule", {}) or {}
                design = proto.get("designModule", {}) or {}
                status = proto.get("statusModule", {}) or {}
                sponsor = proto.get("sponsorCollaboratorsModule", {}) or {}

                nct_id = ident.get("nctId")
                if not nct_id or nct_id in seen:
                    continue
                seen.add(nct_id)

                title = ident.get("briefTitle") or ident.get("officialTitle") or "No title"
                phases = design.get("phases") or ["Not reported"]
                overall_status = status.get("overallStatus") or "Unknown"
                sponsor_name = sponsor.get("leadSponsor", {}).get("name", "Unknown")
                vaccines = _extract_vaccine_names(s)

                all_results.append({
                    "NCT ID": str(nct_id),
                    "Title": str(title),
                    "Phase": ", ".join(phases),
                    "Status": str(overall_status),
                    "Sponsor": str(sponsor_name),
                    "Vaccines": ", ".join(vaccines) if vaccines else "Not reported"
                })

            page_token = data.get("nextPageToken")
            page_count += 1
            if not page_token:
                break
            time.sleep(0.25)

        # Fallback: no vaccine trials detected -> broaden text search
        if not all_results:
            page_token = None
            page_count = 0
            while page_count < max_pages:
                params = {"query.cond": disease, "query.term": "vaccin*", "pageSize": 100}
                if page_token:
                    params["pageToken"] = page_token

                data = _request_json(url, params=params, session=session, headers=headers)
                studies = data.get("studies", []) or []
                if not studies:
                    break

                for s in studies:
                    if not _is_vaccine_study(s):
                        continue
                    proto = s.get("protocolSection", {}) or {}
                    ident = proto.get("identificationModule", {}) or {}
                    design = proto.get("designModule", {}) or {}
                    status = proto.get("statusModule", {}) or {}
                    sponsor = proto.get("sponsorCollaboratorsModule", {}) or {}
                    nct_id = ident.get("nctId")
                    if not nct_id or nct_id in seen:
                        continue
                    seen.add(nct_id)
                    vaccines = _extract_vaccine_names(s)
                    all_results.append({
                        "NCT ID": str(nct_id),
                        "Title": str(ident.get("briefTitle") or ident.get("officialTitle") or "No title"),
                        "Phase": ", ".join(design.get("phases") or ["Not reported"]),
                        "Status": str(status.get("overallStatus") or "Unknown"),
                        "Sponsor": str(sponsor.get("leadSponsor", {}).get("name", "Unknown")),
                        "Vaccines": ", ".join(vaccines) if vaccines else "Not reported"
                    })

                page_token = data.get("nextPageToken")
                page_count += 1
                if not page_token:
                    break
                time.sleep(0.25)

        return all_results
    except requests.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_trial_details_with_vaccines(nct_id: str):
    """Fetch detailed trial info with better vaccine/disease/outcome extraction."""
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    try:
        data = _request_json(url, params=None)
        proto = data.get("protocolSection", {}) or {}
        design = proto.get("designModule", {}) or {}
        status = proto.get("statusModule", {}) or {}
        sponsor = proto.get("sponsorCollaboratorsModule", {}) or {}
        conditions = proto.get("conditionsModule", {}) or {}

        vaccines = _extract_vaccine_names(data)

        # Conditions: protocol + MeSH
        disease_list = []
        disease_list.extend(conditions.get("conditions", []) or [])
        disease_list.extend(_mesh_terms_condition(data))
        seen_d = set()
        disease_list = [d for d in disease_list if d and not (d in seen_d or seen_d.add(d))]

        # Outcomes
        outcomes = _results_outcomes(data)
        if not outcomes:
            outcomes = _primary_outcomes_from_protocol(data)

        ident = proto.get("identificationModule", {}) or {}
        title = ident.get("briefTitle") or ident.get("officialTitle") or "No title"
        phases = design.get("phases") or ["Not reported"]
        overall_status = status.get("overallStatus", "Unknown")
        sponsor_name = sponsor.get("leadSponsor", {}).get("name", "Unknown")

        return {
            "Title": str(title),
            "Phase": ", ".join(phases),
            "Status": str(overall_status),
            "Sponsor": str(sponsor_name),
            "Vaccines": sorted(vaccines) if vaccines else ["Not reported"],
            "Diseases": disease_list,
            "Outcomes": outcomes
        }
    except requests.RequestException:
        return None

# ---------------- Main UI ----------------
st.title("💉 Vaccine Pipeline Platform")
st.markdown("Explore complete vaccine trial data from ClinicalTrials.gov. Search by disease condition or vaccine product name with competitor analysis.")

# Ensure state keys exist
for k in ["studies", "vaccine_trials", "competitor_trials", "target_vaccine", "target_diseases"]:
    st.session_state.setdefault(k, [] if "trials" in k or "studies" in k or "diseases" in k else "")

tab1, tab2 = st.tabs(["🔍 Search by Disease", "💊 Search by Vaccine Product"])

# ---------------- TAB 1: Search by Disease ----------------
with tab1:
    st.subheader("Search Vaccine Trials by Disease")
    st.caption("Fetches trials by disease and classifies vaccines using MeSH and heuristics.")

    disease = st.text_input("Enter Disease Name", value="RSV", key="disease_input")

    if st.button("🔍 Fetch All Trials", key="fetch_disease"):
        with st.spinner("Fetching all vaccine trials (this may take a moment)..."):
            studies = fetch_all_vaccine_trials(disease, max_pages=10)
            if not studies:
                st.warning(f"No vaccine studies found for '{disease}'. Try another disease or broader term.")
                st.session_state["studies"] = []
            else:
                st.session_state["studies"] = studies
                st.success(f"✅ Found {len(studies)} vaccine trials for {disease}.")

    studies = st.session_state.get("studies", [])

    if studies:
        df = pd.DataFrame(studies)

        # Sidebar filters (Disease Search) - single filter set here
        st.sidebar.header("🎛️ Filters (Disease Search)")
        phase_options = sorted({p.strip() for val in df["Phase"].dropna() for p in str(val).split(",")})
        status_options = sorted([s for s in df["Status"].dropna().unique()])

        selected_phases = st.sidebar.multiselect("Phase", options=phase_options, default=phase_options, key="phase_filter_disease")
        selected_status = st.sidebar.multiselect("Status", options=status_options, default=status_options, key="status_filter_disease")

        def _row_has_phase(ph_str: str, selected: list) -> bool:
            row_phases = [p.strip() for p in str(ph_str).split(",")]
            return any(p in row_phases or p in ph_str for p in selected) if selected else True

        df_filtered = df[df["Phase"].apply(lambda x: _row_has_phase(x, selected_phases))]
        if selected_status:
            df_filtered = df_filtered[df_filtered["Status"].isin(selected_status)]

        st.info(f"📊 Showing {len(df_filtered)} of {len(studies)} trials")
        show_df(df_filtered, height=420)

        if st.button("🧠 Summarize Displayed Trials", key="summarize_disease_trials"):
            with st.spinner("Generating AI executive summary..."):
                summary_txt, summary_err = _summarize_trials_with_llm(
                    df_filtered.to_dict("records"),
                    context_instructions=f"Disease search term: {disease}. Showing {len(df_filtered)} of {len(df)} vaccine trials."
                )
            if summary_txt:
                st.markdown("#### 🤖 AI Executive Summary")
                st.write(summary_txt)
            elif summary_err:
                st.warning(summary_err)

        selected_id = st.selectbox(
            "🔬 View Detailed Info",
            options=["Select a study..."] + [str(x) for x in df_filtered["NCT ID"].tolist()],
            key="select_disease_detail"
        )

        if selected_id != "Select a study...":
            with st.spinner("Loading details..."):
                details = fetch_trial_details_with_vaccines(selected_id)

            if details:
                st.markdown("---")
                st.subheader(f"📋 Study Details: {selected_id}")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Title:** {details['Title']}")
                    st.markdown(f"**Phase:** {details['Phase']}")
                with col2:
                    st.markdown(f"**Status:** {details['Status']}")
                    st.markdown(f"**Sponsor:** {details['Sponsor']}")

                if details.get("Diseases"):
                    st.markdown("**🦠 Diseases/Conditions:**")
                    st.write(", ".join(details["Diseases"]))

                st.markdown("**💉 Vaccine Products:**")
                for v in details["Vaccines"]:
                    st.markdown(f"- {v}")

                if details["Outcomes"]:
                    st.markdown("**📊 Primary Outcome Measures:**")
                    for o in details["Outcomes"]:
                        st.write(f"• {o['Title']}")
                        if o["Description"]:
                            st.caption(o["Description"])
                else:
                    st.info("No outcomes reported yet.")

                if st.button("🧠 Summarize This Study", key=f"summarize_detail_{selected_id}"):
                    with st.spinner("Creating AI summary..."):
                        trial_summary, detail_err = _summarize_single_trial(selected_id, details)
                    if trial_summary:
                        st.markdown("#### 🤖 AI Trial Brief")
                        st.write(trial_summary)
                    elif detail_err:
                        st.warning(detail_err)
    else:
        st.info("👆 Enter a disease name and click Fetch All Trials to begin.")

# ---------------- TAB 2: Search by Vaccine Product + Competitors ----------------
with tab2:
    st.subheader("Search Trials by Vaccine Product")
    st.caption("Find your vaccine’s trials + competitor vaccines targeting the same disease(s).")

    vaccine_name = st.text_input("Enter Vaccine Product Name", value="", key="vaccine_input")

    if st.button("💊 Search Vaccine & Competitors", key="fetch_vaccine"):
        if not vaccine_name.strip():
            st.warning("Please enter a vaccine name.")
        else:
            target_norm = _norm_txt(vaccine_name)
            search_url = "https://clinicaltrials.gov/api/v2/studies"

            try:
                with st.spinner(f"Step 1/2: Finding trials for '{vaccine_name}'..."):
                    params = {"query.term": vaccine_name, "pageSize": 100}
                    data = _request_json(search_url, params=params)
                    studies = data.get("studies", []) or []

                    vaccine_results = []
                    all_diseases = []

                    for s in studies:
                        names = _extract_vaccine_names(s)
                        names_norm = " || ".join(_norm_txt(n) for n in names)
                        if target_norm and (target_norm in names_norm or any(_norm_txt(n) == target_norm for n in names)):
                            if not _is_vaccine_study(s):
                                continue
                            proto = s.get("protocolSection", {}) or {}
                            ident = proto.get("identificationModule", {}) or {}
                            design = proto.get("designModule", {}) or {}
                            status = proto.get("statusModule", {}) or {}
                            sponsor = proto.get("sponsorCollaboratorsModule", {}) or {}
                            nct_id = ident.get("nctId")
                            if nct_id:
                                vaccine_results.append({
                                    "NCT ID": str(nct_id),
                                    "Title": str(ident.get("briefTitle") or ident.get("officialTitle") or "No title"),
                                    "Phase": ", ".join(design.get("phases") or ["Not reported"]),
                                    "Status": str(status.get("overallStatus") or "Unknown"),
                                    "Sponsor": str(sponsor.get("leadSponsor", {}).get("name", "Unknown")),
                                    "Vaccines": ", ".join(names) if names else "Not reported"
                                })
                                ds = []
                                ds.extend(proto.get("conditionsModule", {}).get("conditions", []) or [])
                                ds.extend(_mesh_terms_condition(s))
                                all_diseases.extend(ds)

                    st.session_state["target_vaccine"] = vaccine_name
                    st.session_state["vaccine_trials"] = vaccine_results

                    disease_counts = Counter([d for d in all_diseases if d])
                    top_diseases = [d for d, _ in disease_counts.most_common(2)]
                    st.session_state["target_diseases"] = top_diseases

                # Step 2: competitors
                competitor_trials = []
                top_diseases = st.session_state.get("target_diseases", [])
                if top_diseases:
                    with st.spinner(f"Step 2/2: Finding competitor vaccines for {', '.join(top_diseases)}..."):
                        seen = set()
                        for d in top_diseases:
                            trials = fetch_all_vaccine_trials(d, max_pages=5)
                            for t in trials:
                                vacc_norm = _norm_txt(t.get("Vaccines", ""))
                                if target_norm and (target_norm in vacc_norm):
                                    continue
                                nct = t.get("NCT ID")
                                if nct and nct not in seen:
                                    seen.add(nct)
                                    competitor_trials.append(t)

                    st.session_state["competitor_trials"] = competitor_trials
                    st.success(f"✅ Found {len(vaccine_results)} trials for '{vaccine_name}' and {len(competitor_trials)} competitor trials!")
                else:
                    st.session_state["competitor_trials"] = []
                    st.success(f"✅ Found {len(st.session_state['vaccine_trials'])} trials for '{vaccine_name}' (no clear disease context detected)")
            except requests.RequestException as e:
                st.error(f"Search failed: {e}")
                st.session_state["vaccine_trials"] = []
                st.session_state["competitor_trials"] = []

    # Show vaccine trials (WITH its own filters)
    vaccine_trials = st.session_state.get("vaccine_trials", [])
    competitor_trials = st.session_state.get("competitor_trials", [])
    target_vaccine = st.session_state.get("target_vaccine", "")
    target_diseases = st.session_state.get("target_diseases", [])

    if vaccine_trials:
        st.markdown("---")
        st.subheader(f"🎯 Your Vaccine: {target_vaccine}")
        if target_diseases:
            st.caption(f"Primary Disease(s): {', '.join(target_diseases)}")

        df_vaccine = pd.DataFrame(vaccine_trials)

        # Sidebar filters for YOUR vaccine trials (Tab 2)
        st.sidebar.header("🎛️ Vaccine Filters")
        phase_options_v = sorted({p.strip() for val in df_vaccine["Phase"].dropna() for p in str(val).split(",")})
        status_options_v = sorted([s for s in df_vaccine["Status"].dropna().unique()])

        selected_phases_v = st.sidebar.multiselect(
            "Phase (Your Vaccine)", options=phase_options_v, default=phase_options_v, key="phase_filter_vaccine"
        )
        selected_status_v = st.sidebar.multiselect(
            "Status (Your Vaccine)", options=status_options_v, default=status_options_v, key="status_filter_vaccine"
        )

        def _row_has_phase_v(ph_str: str, selected: list) -> bool:
            row_phases = [p.strip() for p in str(ph_str).split(",")]
            return any(p in row_phases or p in ph_str for p in selected) if selected else True

        df_vaccine_filtered = df_vaccine[df_vaccine["Phase"].apply(lambda x: _row_has_phase_v(x, selected_phases_v))]
        if selected_status_v:
            df_vaccine_filtered = df_vaccine_filtered[df_vaccine_filtered["Status"].isin(selected_status_v)]

        st.info(f"📊 Showing {len(df_vaccine_filtered)} of {len(df_vaccine)} trials")
        show_df(df_vaccine_filtered, height=320)

        if st.button("🧠 Summarize Your Vaccine Trials", key="summarize_vaccine_trials"):
            with st.spinner("Generating AI summary for your vaccine..."):
                vacc_summary, vacc_err = _summarize_trials_with_llm(
                    df_vaccine_filtered.to_dict("records"),
                    context_instructions=f"Target vaccine: {target_vaccine}. Top diseases: {', '.join(target_diseases) if target_diseases else 'Unknown'}."
                )
            if vacc_summary:
                st.markdown("#### 🤖 AI Summary — Your Vaccine")
                st.write(vacc_summary)
            elif vacc_err:
                st.warning(vacc_err)

        selected_vaccine_id = st.selectbox(
            "🔬 View Detailed Info",
            options=["Select a study..."] + [str(x) for x in df_vaccine_filtered["NCT ID"].tolist()],
            key="select_vaccine_detail"
        )

        if selected_vaccine_id != "Select a study...":
            with st.spinner("Loading details..."):
                details_v = fetch_trial_details_with_vaccines(selected_vaccine_id)

            if details_v:
                st.markdown("---")
                st.subheader(f"📋 Study Details: {selected_vaccine_id}")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Title:** {details_v['Title']}")
                    st.markdown(f"**Phase:** {details_v['Phase']}")
                with col2:
                    st.markdown(f"**Status:** {details_v['Status']}")
                    st.markdown(f"**Sponsor:** {details_v['Sponsor']}")

                if details_v.get("Diseases"):
                    st.markdown("**🦠 Diseases/Conditions:**")
                    st.write(", ".join(details_v["Diseases"]))

                st.markdown("**💉 Vaccine Products:**")
                for v in details_v["Vaccines"]:
                    st.markdown(f"- {v}")

                if details_v["Outcomes"]:
                    st.markdown("**📊 Primary Outcome Measures:**")
                    for o in details_v["Outcomes"]:
                        st.write(f"• {o['Title']}")
                        if o["Description"]:
                            st.caption(o["Description"])

                if st.button("🧠 Summarize This Study", key=f"summarize_vaccine_detail_{selected_vaccine_id}"):
                    with st.spinner("Creating AI trial brief..."):
                        trial_summary_v, detail_err_v = _summarize_single_trial(selected_vaccine_id, details_v)
                    if trial_summary_v:
                        st.markdown("#### 🤖 AI Trial Brief")
                        st.write(trial_summary_v)
                    elif detail_err_v:
                        st.warning(detail_err_v)

    # Show competitor trials (existing filters kept as-is)
    if competitor_trials:
        st.markdown("---")
        st.subheader(f"🔄 Competitor Vaccines for {', '.join(target_diseases) if target_diseases else 'Same Disease'}")
        st.caption("All other vaccines targeting the same disease(s)")

        df_competitor = pd.DataFrame(competitor_trials)

        st.sidebar.header("🎛️ Competitor Filters")
        phase_options_c = sorted({p.strip() for val in df_competitor["Phase"].dropna() for p in str(val).split(",")})
        status_options_c = sorted([s for s in df_competitor["Status"].dropna().unique()])

        selected_phases_c = st.sidebar.multiselect(
            "Phase (Competitors)", options=phase_options_c, default=phase_options_c, key="phase_filter_comp"
        )
        selected_status_c = st.sidebar.multiselect(
            "Status (Competitors)", options=status_options_c, default=status_options_c, key="status_filter_comp"
        )

        def _row_has_phase_c(ph_str: str, selected: list) -> bool:
            row_phases = [p.strip() for p in str(ph_str).split(",")]
            return any(p in row_phases or p in ph_str for p in selected) if selected else True

        df_competitor_filtered = df_competitor[df_competitor["Phase"].apply(lambda x: _row_has_phase_c(x, selected_phases_c))]
        if selected_status_c:
            df_competitor_filtered = df_competitor_filtered[df_competitor_filtered["Status"].isin(selected_status_c)]

        st.info(f"📊 Showing {len(df_competitor_filtered)} of {len(competitor_trials)} competitor trials")
        show_df(df_competitor_filtered, height=420)

        if st.button("🧠 Summarize Competitor Trials", key="summarize_comp_trials"):
            with st.spinner("Creating AI competitor synopsis..."):
                comp_summary, comp_err = _summarize_trials_with_llm(
                    df_competitor_filtered.to_dict("records"),
                    context_instructions=f"Competitor vaccines targeting diseases: {', '.join(target_diseases) if target_diseases else 'Unknown'}."
                )
            if comp_summary:
                st.markdown("#### 🤖 AI Summary — Competitors")
                st.write(comp_summary)
            elif comp_err:
                st.warning(comp_err)

        selected_comp_id = st.selectbox(
            "🔬 View Competitor Trial Details",
            options=["Select a study..."] + [str(x) for x in df_competitor_filtered["NCT ID"].tolist()],
            key="select_comp_detail"
        )

        if selected_comp_id != "Select a study...":
            with st.spinner("Loading details..."):
                details_c = fetch_trial_details_with_vaccines(selected_comp_id)

            if details_c:
                st.markdown("---")
                st.subheader(f"📋 Competitor Study: {selected_comp_id}")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Title:** {details_c['Title']}")
                    st.markdown(f"**Phase:** {details_c['Phase']}")
                with col2:
                    st.markdown(f"**Status:** {details_c['Status']}")
                    st.markdown(f"**Sponsor:** {details_c['Sponsor']}")

                if details_c.get("Diseases"):
                    st.markdown("**🦠 Diseases/Conditions:**")
                    st.write(", ".join(details_c["Diseases"]))

                st.markdown("**💉 Vaccine Products:**")
                for v in details_c["Vaccines"]:
                    st.markdown(f"- {v}")
                if details_c["Outcomes"]:
                    st.markdown("**📊 Primary Outcome Measures:**")
                    for o in details_c["Outcomes"]:
                        st.write(f"• {o['Title']}")
                        if o["Description"]:
                            st.caption(o["Description"])

                if st.button("🧠 Summarize This Study", key=f"summarize_comp_detail_{selected_comp_id}"):
                    with st.spinner("Creating AI trial brief..."):
                        trial_summary_c, detail_err_c = _summarize_single_trial(selected_comp_id, details_c)
                    if trial_summary_c:
                        st.markdown("#### 🤖 AI Trial Brief")
                        st.write(trial_summary_c)
                    elif detail_err_c:
                        st.warning(detail_err_c)

if not st.session_state.get("vaccine_trials") and not st.session_state.get("competitor_trials"):
    st.info("👆 Enter a vaccine product name and click Search Vaccine & Competitors to begin.")

# ---------------- Roadmap / Guidance ----------------
with st.expander("🔭 Platform Roadmap to Full RAG Insights", expanded=False):
    st.markdown(
        """
- **Enhanced API coverage:** add `query.fields` calls for enrollment, outcome text blobs, and sponsor org IDs; schedule refresh jobs per indication.
- **Targeted web retrieval:** maintain an allow-listed catalog (FDA, EMA, company IR, WHO) and crawl via Requests/BS4, respecting `robots.txt`.
- **Document parsing:** normalize PDFs (PyPDF + unstructured.io) and HTML (trafilatura) into chunked markdown.
- **Vector store:** embed chunks with `text-embedding-004` or `all-MiniLM`, persist in FAISS per vaccine + disease taxonomy.
- **Retrieval-augmented summaries:** chain query understanding → chunk retrieval → OpenRouter LLM summarizer producing compliance-safe briefs.
- **Executive views:** combine structured KPIs + narrative insights, export to ppt/pdf for field medical + BD teams.
"""
    )

# ---------------- Footer ----------------
st.markdown("---")
st.caption("💡 Vaccine Pipeline Platform | Data from ClinicalTrials.gov | Developed by Aman & Smriti")