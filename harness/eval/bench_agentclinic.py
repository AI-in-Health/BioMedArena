"""Clinical simulation loader — converts scenarios to diagnostic QA.

We convert clinical cases into diagnostic QA tasks:
present patient history + findings, ask for diagnosis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Representative diagnostic scenarios covering multiple medical specialties.
AGENTCLINIC_TASKS: list[dict[str, Any]] = [
    # --- Internal Medicine ---
    {
        "id": "ac_001", "category": "Clinical/Internal Medicine",
        "answer_type": "exactMatch",
        "question": "Patient: 58-year-old male.\nChief complaint: Progressive fatigue and shortness of breath for 3 weeks.\nHistory: Hypertension, Type 2 DM for 12 years. Smoker 30 pack-years, quit 2 years ago.\nPhysical exam: JVD present, bilateral lower extremity edema 2+, bibasilar crackles, S3 gallop heard.\nVitals: BP 148/92, HR 102, RR 22, SpO2 91% on room air.\nLabs: BNP 1840 pg/mL, Troponin I 0.02 (normal), Cr 1.6.\nECG: Left ventricular hypertrophy, no acute ST changes.\nWhat is the most likely diagnosis?",
        "answer": "congestive heart failure",
    },
    {
        "id": "ac_002", "category": "Clinical/Internal Medicine",
        "answer_type": "multipleChoice",
        "question": "Patient: 45-year-old female.\nChief complaint: Recurrent episodes of palpitations, heat intolerance, and 15 lb weight loss over 2 months.\nPhysical exam: Diffuse, non-tender thyroid enlargement; fine tremor of hands; lid lag; brisk reflexes.\nVitals: HR 110, BP 145/68 (widened pulse pressure).\nLabs: TSH <0.01, Free T4 4.2 (high), TSH receptor antibodies positive.\nWhat is the most appropriate initial treatment?\nA) Radioactive iodine ablation\nB) Methimazole\nC) Propylthiouracil\nD) Total thyroidectomy\nE) Propranolol monotherapy",
        "answer": "B",
    },
    {
        "id": "ac_003", "category": "Clinical/Internal Medicine",
        "answer_type": "exactMatch",
        "question": "Patient: 35-year-old male.\nChief complaint: Severe epigastric pain radiating to the back for 6 hours, worse after eating.\nHistory: Heavy alcohol use (6-8 drinks daily for 10 years). No prior episodes.\nPhysical exam: Tenderness in epigastrium, guarding, diminished bowel sounds.\nVitals: HR 118, BP 100/65, Temp 38.1°C.\nLabs: Lipase 1,240 U/L (normal <60), WBC 16.2, glucose 220.\nCT abdomen: Pancreatic edema with peripancreatic fluid collections.\nWhat is the Ranson score criterion count at admission (considering: age, WBC, glucose, LDH, AST)?",
        "answer": "3",
    },
    # --- Cardiology ---
    {
        "id": "ac_010", "category": "Clinical/Cardiology",
        "answer_type": "exactMatch",
        "question": "Patient: 68-year-old male.\nChief complaint: Sudden onset severe tearing chest pain radiating to the back.\nHistory: Long-standing poorly controlled HTN, Marfan habitus.\nPhysical exam: BP right arm 180/95, BP left arm 142/80 (>20 mmHg difference). New diastolic murmur.\nCT angiography: Intimal flap in ascending aorta extending to the arch.\nWhat is the diagnosis and Stanford classification?",
        "answer": "Stanford type A aortic dissection",
    },
    {
        "id": "ac_011", "category": "Clinical/Cardiology",
        "answer_type": "multipleChoice",
        "question": "Patient: 72-year-old female.\nChief complaint: Syncope during exertion.\nPhysical exam: Harsh crescendo-decrescendo systolic murmur at the right upper sternal border radiating to carotids. Pulsus parvus et tardus.\nEchocardiogram: Aortic valve area 0.7 cm², mean gradient 48 mmHg, EF 55%.\nWhat is the management?\nA) Medical management with diuretics and ACE inhibitors\nB) Surgical aortic valve replacement (SAVR)\nC) Transcatheter aortic valve replacement (TAVR)\nD) Balloon aortic valvuloplasty\nE) Watchful waiting with serial echocardiograms",
        "answer": "B",
    },
    # --- Neurology ---
    {
        "id": "ac_020", "category": "Clinical/Neurology",
        "answer_type": "exactMatch",
        "question": "Patient: 28-year-old female.\nChief complaint: Two episodes of visual blurring in the left eye (lasting days) over the past year, now presenting with right leg weakness and tingling for 1 week.\nPhysical exam: RAPD in left eye, decreased visual acuity OS, right leg weakness 4/5, hyperreflexia bilateral lower extremities, positive Babinski sign right.\nMRI brain: Multiple periventricular white matter lesions with at least 1 enhancing lesion.\nCSF: Oligoclonal bands present, elevated IgG index.\nWhat is the most likely diagnosis?",
        "answer": "multiple sclerosis",
    },
    {
        "id": "ac_021", "category": "Clinical/Neurology",
        "answer_type": "multipleChoice",
        "question": "Patient: 65-year-old male.\nChief complaint: Sudden onset right-sided weakness and inability to speak, onset 90 minutes ago.\nPhysical exam: Global aphasia, right hemiplegia, right facial droop. NIHSS score: 18.\nCT head: No hemorrhage.\nCT angiography: Left middle cerebral artery M1 occlusion.\nWhat is the recommended treatment?\nA) IV alteplase only\nB) IV alteplase followed by mechanical thrombectomy\nC) Mechanical thrombectomy only\nD) Aspirin 325mg and admit to stroke unit\nE) IV heparin infusion",
        "answer": "B",
    },
    # --- Pulmonology ---
    {
        "id": "ac_030", "category": "Clinical/Pulmonology",
        "answer_type": "exactMatch",
        "question": "Patient: 55-year-old male, lifetime non-smoker.\nChief complaint: Progressive dyspnea on exertion over 2 years, dry cough.\nPhysical exam: Bilateral fine inspiratory crackles (\"Velcro crackles\") at lung bases, clubbing of fingers.\nPFTs: FVC 58% predicted, FEV1 62% predicted, FEV1/FVC ratio 0.85, DLCO 42% predicted.\nHRCT: Bilateral peripheral basilar predominant reticular pattern with honeycombing and traction bronchiectasis. UIP pattern.\nWhat is the diagnosis?",
        "answer": "idiopathic pulmonary fibrosis",
    },
    # --- Gastroenterology ---
    {
        "id": "ac_040", "category": "Clinical/Gastroenterology",
        "answer_type": "multipleChoice",
        "question": "Patient: 52-year-old male with known cirrhosis (Child-Pugh B) secondary to hepatitis C.\nChief complaint: Vomiting bright red blood.\nPhysical exam: Tachycardic (HR 120), hypotensive (BP 82/50), distended abdomen with shifting dullness, spider angiomata.\nHgb: 7.2 g/dL (baseline 10.5).\nWhat is the most critical immediate intervention?\nA) Emergent upper endoscopy with variceal banding\nB) IV octreotide and IV ceftriaxone, then emergent endoscopy\nC) Transjugular intrahepatic portosystemic shunt (TIPS)\nD) Balloon tamponade with Sengstaken-Blakemore tube\nE) IV proton pump inhibitor bolus",
        "answer": "B",
    },
    # --- Infectious Disease ---
    {
        "id": "ac_050", "category": "Clinical/Infectious Disease",
        "answer_type": "exactMatch",
        "question": "Patient: 32-year-old male.\nChief complaint: Fever, night sweats, 20 lb weight loss over 3 months, non-productive cough.\nHistory: Emigrated from India 6 months ago. HIV negative.\nPhysical exam: Cachectic, cervical lymphadenopathy.\nCXR: Right upper lobe cavitary lesion with surrounding infiltrate.\nSputum: Acid-fast bacilli positive on smear.\nWhat is the standard initial treatment regimen (drug names)?",
        "answer": "RIPE: rifampin, isoniazid, pyrazinamide, ethambutol",
    },
    {
        "id": "ac_051", "category": "Clinical/Infectious Disease",
        "answer_type": "multipleChoice",
        "question": "Patient: 40-year-old female with HIV (CD4 count 85, viral load 250,000).\nPresents with: Fever, headache, neck stiffness for 5 days.\nCSF analysis: Opening pressure 32 cm H2O, WBC 15 (95% lymphocytes), protein 65, glucose 35.\nCSF India ink: Encapsulated yeast forms seen.\nWhat is the recommended initial treatment?\nA) Amphotericin B deoxycholate + flucytosine for 2 weeks, then fluconazole\nB) Fluconazole 400mg daily for 8 weeks\nC) Voriconazole IV for 6 weeks\nD) Caspofungin + fluconazole\nE) Itraconazole 200mg BID",
        "answer": "A",
    },
    # --- Endocrinology ---
    {
        "id": "ac_060", "category": "Clinical/Endocrinology",
        "answer_type": "exactMatch",
        "question": "Patient: 48-year-old female.\nChief complaint: Episodic headaches, palpitations, and diaphoresis. Found to have severe hypertension (240/130) during an episode.\nLabs: 24-hour urine metanephrines markedly elevated. Plasma free metanephrines elevated.\nCT abdomen: 4 cm right adrenal mass.\nWhat is the diagnosis?",
        "answer": "pheochromocytoma",
    },
    # --- Nephrology ---
    {
        "id": "ac_070", "category": "Clinical/Nephrology",
        "answer_type": "multipleChoice",
        "question": "Patient: 25-year-old male.\nChief complaint: Cola-colored urine and facial swelling 2 weeks after pharyngitis.\nLabs: Cr 2.1, low C3 complement (normal C4), elevated ASO titer. UA: RBC casts, proteinuria 2+.\nRenal biopsy: Diffuse proliferative pattern with \"lumpy-bumpy\" subepithelial deposits on electron microscopy.\nWhat is the diagnosis?\nA) IgA nephropathy\nB) Post-streptococcal glomerulonephritis\nC) Membranous nephropathy\nD) Minimal change disease\nE) Lupus nephritis",
        "answer": "B",
    },
    # --- Hematology/Oncology ---
    {
        "id": "ac_080", "category": "Clinical/Hematology",
        "answer_type": "exactMatch",
        "question": "Patient: 60-year-old male.\nChief complaint: Fatigue, early satiety, and left upper quadrant fullness.\nPhysical exam: Massive splenomegaly (palpable 10 cm below costal margin).\nCBC: WBC 145,000 with left shift (myelocytes, metamyelocytes, bands, mature neutrophils, basophilia), Hgb 10.2, Plt 450,000.\nPeripheral smear: Full spectrum of myeloid maturation.\nBone marrow: Hypercellular with myeloid hyperplasia.\nCytogenetics: t(9;22) Philadelphia chromosome positive.\nWhat is the diagnosis?",
        "answer": "chronic myeloid leukemia",
    },
    # --- Rheumatology ---
    {
        "id": "ac_090", "category": "Clinical/Rheumatology",
        "answer_type": "multipleChoice",
        "question": "Patient: 35-year-old female.\nChief complaint: Joint pain, fatigue, facial rash for 3 months.\nPhysical exam: Butterfly-shaped malar rash sparing nasolabial folds, oral ulcers, symmetric polyarthritis of MCP and PIP joints.\nLabs: ANA 1:640, anti-dsDNA positive, low C3 and C4, WBC 3.2, Plt 98, UA shows 2+ protein.\nHow many ACR criteria does this patient meet?\nA) 4\nB) 5\nC) 6\nD) 7\nE) 3",
        "answer": "C",
    },
    # --- Emergency Medicine ---
    {
        "id": "ac_100", "category": "Clinical/Emergency",
        "answer_type": "exactMatch",
        "question": "Patient: 22-year-old male brought in after a motor vehicle collision.\nPrimary survey: A - speaking (patent), B - RR 28 SpO2 88% decreased breath sounds on left, C - HR 128 BP 85/60, D - GCS 14 (E4V4M6).\nCXR: Left-sided white-out with mediastinal shift to the right.\nWhat is the immediate life-saving intervention?",
        "answer": "left chest tube thoracostomy",
    },
    {
        "id": "ac_101", "category": "Clinical/Emergency",
        "answer_type": "multipleChoice",
        "question": "Patient: 8-year-old child.\nBrought in after choking on a grape. Now: stridor, drooling, unable to speak, becoming cyanotic.\nAbdominal thrusts attempted but unsuccessful. SpO2 dropping to 72%.\nWhat is the next step?\nA) Repeat abdominal thrusts\nB) Direct laryngoscopy and Magill forceps removal\nC) Emergency cricothyrotomy\nD) Back blows and chest thrusts\nE) Bag-valve-mask ventilation",
        "answer": "B",
    },
]


def load_agentclinic_tasks(
    vendor_path: str = "vendors/AgentClinic",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load clinical simulation tasks.

    Tries vendor repo first, falls back to built-in representative scenarios.
    `limit` caps the number of returned tasks (applied after either path),
    matching the signature convention used by other bench_*_tasks loaders.
    """
    vendor = Path(vendor_path)

    # Try to load from vendor repo
    for candidate in [
        vendor / "data" / "medqa_cases.json",
        vendor / "scenarios" / "medqa.json",
        vendor / "agentclinic" / "data" / "cases.json",
    ]:
        if candidate.exists():
            try:
                raw = json.loads(candidate.read_text())
                tasks = []
                for i, case in enumerate(raw):
                    # Convert interactive case to diagnostic QA
                    patient_info = case.get("patient_info", case.get("history", ""))
                    diagnosis = case.get("diagnosis", case.get("correct_diagnosis", ""))
                    if not patient_info or not diagnosis:
                        continue
                    tasks.append({
                        "id": f"ac_{i:03d}",
                        "question": f"Based on the following patient presentation, what is the most likely diagnosis?\n\n{patient_info}",
                        "answer": diagnosis,
                        "answer_type": "exactMatch",
                        "category": f"Clinical/{case.get('specialty', 'General')}",
                    })
                if tasks:
                    return tasks[:limit] if limit else tasks
            except Exception:
                pass

    return AGENTCLINIC_TASKS[:limit] if limit else list(AGENTCLINIC_TASKS)
