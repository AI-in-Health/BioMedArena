"""Top 20 clinical risk calculators as plain Python functions.

Each function returns {"score": numeric, "category": str, "recommendation": str}.
"""

from __future__ import annotations

import math
from typing import Any


def cha2ds2_vasc(
    age: int,
    sex: str,
    chf: bool = False,
    hypertension: bool = False,
    stroke_tia_history: bool = False,
    vascular_disease: bool = False,
    diabetes: bool = False,
) -> dict[str, Any]:
    """CHA₂DS₂-VASc score for atrial fibrillation stroke risk."""
    score = 0
    if chf:
        score += 1
    if hypertension:
        score += 1
    if age >= 75:
        score += 2
    elif age >= 65:
        score += 1
    if diabetes:
        score += 1
    if stroke_tia_history:
        score += 2
    if vascular_disease:
        score += 1
    if sex.lower() in ("f", "female"):
        score += 1

    if score == 0:
        cat, rec = "Low", "No anticoagulation recommended."
    elif score == 1:
        cat, rec = "Low-Moderate", "Consider anticoagulation; discuss with patient."
    else:
        cat, rec = "Moderate-High", "Oral anticoagulation recommended."

    return {"score": score, "category": cat, "recommendation": rec}


def heart_score(
    history: int,
    ecg: int,
    age: int,
    risk_factors: int,
    troponin: int,
) -> dict[str, Any]:
    """HEART Score for major cardiac events (each component 0-2)."""
    score = history + ecg + age + risk_factors + troponin
    if score <= 3:
        cat, rec = "Low", "Consider early discharge; outpatient follow-up."
    elif score <= 6:
        cat, rec = "Moderate", "Admit for observation and further workup."
    else:
        cat, rec = "High", "Early invasive strategy recommended."
    return {"score": score, "category": cat, "recommendation": rec}


def wells_dvt(
    active_cancer: bool = False,
    paralysis_or_immobilization: bool = False,
    bedridden_gt3days: bool = False,
    localized_tenderness: bool = False,
    entire_leg_swollen: bool = False,
    calf_swelling_gt3cm: bool = False,
    pitting_edema: bool = False,
    collateral_superficial_veins: bool = False,
    previous_dvt: bool = False,
    alternative_diagnosis_likely: bool = False,
) -> dict[str, Any]:
    """Wells score for DVT probability."""
    score = sum([
        active_cancer,
        paralysis_or_immobilization,
        bedridden_gt3days,
        localized_tenderness,
        entire_leg_swollen,
        calf_swelling_gt3cm,
        pitting_edema,
        collateral_superficial_veins,
        previous_dvt,
    ]) - (2 if alternative_diagnosis_likely else 0)

    if score <= 0:
        cat, rec = "Low", "DVT unlikely; consider D-dimer."
    elif score <= 2:
        cat, rec = "Moderate", "Moderate probability; consider ultrasound."
    else:
        cat, rec = "High", "DVT likely; obtain ultrasound."
    return {"score": score, "category": cat, "recommendation": rec}


def wells_pe(
    clinical_signs_dvt: bool = False,
    pe_most_likely: bool = False,
    heart_rate_gt100: bool = False,
    immobilization_or_surgery: bool = False,
    previous_dvt_pe: bool = False,
    hemoptysis: bool = False,
    malignancy: bool = False,
) -> dict[str, Any]:
    """Wells score for PE probability."""
    score = 0.0
    if clinical_signs_dvt:
        score += 3
    if pe_most_likely:
        score += 3
    if heart_rate_gt100:
        score += 1.5
    if immobilization_or_surgery:
        score += 1.5
    if previous_dvt_pe:
        score += 1.5
    if hemoptysis:
        score += 1
    if malignancy:
        score += 1

    if score <= 4:
        cat, rec = "PE unlikely", "Consider D-dimer to exclude PE."
    else:
        cat, rec = "PE likely", "Consider CT pulmonary angiography."
    return {"score": score, "category": cat, "recommendation": rec}


def curb65(
    confusion: bool = False,
    bun_gt19: bool = False,
    respiratory_rate_ge30: bool = False,
    sbp_lt90_or_dbp_le60: bool = False,
    age_ge65: bool = False,
) -> dict[str, Any]:
    """CURB-65 score for community-acquired pneumonia severity."""
    score = sum([confusion, bun_gt19, respiratory_rate_ge30, sbp_lt90_or_dbp_le60, age_ge65])
    if score <= 1:
        cat, rec = "Low", "Consider outpatient treatment."
    elif score == 2:
        cat, rec = "Moderate", "Consider short inpatient stay or closely supervised outpatient."
    else:
        cat, rec = "Severe", "Hospitalize; consider ICU if score 4-5."
    return {"score": score, "category": cat, "recommendation": rec}


def qsofa(
    respiratory_rate_ge22: bool = False,
    altered_mentation: bool = False,
    sbp_le100: bool = False,
) -> dict[str, Any]:
    """qSOFA score for sepsis screening."""
    score = sum([respiratory_rate_ge22, altered_mentation, sbp_le100])
    if score < 2:
        cat, rec = "Low", "qSOFA negative; reassess clinically."
    else:
        cat, rec = "High", "qSOFA positive; assess for organ dysfunction (full SOFA)."
    return {"score": score, "category": cat, "recommendation": rec}


def meld(bilirubin: float, inr: float, creatinine: float, sodium: float = 137.0, on_dialysis: bool = False) -> dict[str, Any]:
    """MELD-Na score for liver disease severity."""
    # Bound values per UNOS
    bilirubin = max(bilirubin, 1.0)
    creatinine = min(max(creatinine, 1.0) if not on_dialysis else 4.0, 4.0)
    inr = max(inr, 1.0)
    sodium = min(max(sodium, 125.0), 137.0)

    meld_score = 10 * (
        0.957 * math.log(creatinine)
        + 0.378 * math.log(bilirubin)
        + 1.120 * math.log(inr)
        + 0.643
    )
    meld_score = round(meld_score)
    meld_na = meld_score + 1.32 * (137 - sodium) - 0.033 * meld_score * (137 - sodium)
    meld_na = round(min(max(meld_na, 6), 40))

    if meld_na < 10:
        cat, rec = "Low", "3-month mortality ~2%. Monitor."
    elif meld_na < 20:
        cat, rec = "Moderate", "3-month mortality ~6%. Close follow-up."
    elif meld_na < 30:
        cat, rec = "High", "3-month mortality ~20%. Consider transplant evaluation."
    else:
        cat, rec = "Very High", "3-month mortality >50%. Urgent transplant evaluation."
    return {"score": meld_na, "category": cat, "recommendation": rec}


def child_pugh(
    bilirubin: float,
    albumin: float,
    inr: float,
    ascites: str = "none",
    encephalopathy: str = "none",
) -> dict[str, Any]:
    """Child-Pugh score for chronic liver disease."""
    score = 0

    # Bilirubin
    if bilirubin < 2:
        score += 1
    elif bilirubin <= 3:
        score += 2
    else:
        score += 3

    # Albumin
    if albumin > 3.5:
        score += 1
    elif albumin >= 2.8:
        score += 2
    else:
        score += 3

    # INR
    if inr < 1.7:
        score += 1
    elif inr <= 2.3:
        score += 2
    else:
        score += 3

    # Ascites
    if ascites == "none":
        score += 1
    elif ascites in ("mild", "controlled"):
        score += 2
    else:
        score += 3

    # Encephalopathy
    if encephalopathy == "none":
        score += 1
    elif encephalopathy in ("grade1", "grade2", "1", "2"):
        score += 2
    else:
        score += 3

    if score <= 6:
        cat, rec = "Class A", "Well-compensated; 1-year survival ~100%."
    elif score <= 9:
        cat, rec = "Class B", "Significant functional compromise; 1-year survival ~80%."
    else:
        cat, rec = "Class C", "Decompensated; 1-year survival ~45%."
    return {"score": score, "category": cat, "recommendation": rec}


def ckd_epi_egfr(creatinine: float, age: int, sex: str, race: str = "other") -> dict[str, Any]:
    """CKD-EPI eGFR (2021 equation, race-free)."""
    # 2021 CKD-EPI (no race coefficient)
    is_female = sex.lower() in ("f", "female")
    if is_female:
        kappa = 0.7
        alpha = -0.241 if creatinine <= 0.7 else -1.200
        multiplier = 142 * (0.9938 ** age) * 1.012
    else:
        kappa = 0.9
        alpha = -0.302 if creatinine <= 0.9 else -1.200
        multiplier = 142 * (0.9938 ** age)

    egfr = multiplier * ((creatinine / kappa) ** alpha)
    egfr = round(egfr, 1)

    if egfr >= 90:
        cat, rec = "G1 (Normal)", "Monitor; address risk factors."
    elif egfr >= 60:
        cat, rec = "G2 (Mildly decreased)", "Monitor kidney function annually."
    elif egfr >= 45:
        cat, rec = "G3a (Mild-moderate)", "Refer to nephrology; monitor every 6 months."
    elif egfr >= 30:
        cat, rec = "G3b (Moderate-severe)", "Nephrology management; monitor every 3 months."
    elif egfr >= 15:
        cat, rec = "G4 (Severely decreased)", "Prepare for renal replacement therapy."
    else:
        cat, rec = "G5 (Kidney failure)", "Dialysis or transplant indicated."
    return {"score": egfr, "category": cat, "recommendation": rec}


def apache_ii(
    temperature: float,
    map_mmhg: float,
    heart_rate: int,
    respiratory_rate: int,
    pao2_or_aado2: float,
    fio2_ge50: bool,
    arterial_ph: float,
    sodium: float,
    potassium: float,
    creatinine: float,
    hematocrit: float,
    wbc: float,
    gcs: int,
    age: int,
    chronic_health_points: int = 0,
) -> dict[str, Any]:
    """APACHE II score (simplified — uses midpoint ranges for APS)."""
    # This is a simplified implementation; full APACHE II has a detailed APS table.
    # We compute a rough APS score.
    aps = 0

    # Temperature
    t = temperature
    if t >= 41 or t <= 29.9:
        aps += 4
    elif t >= 39 or t <= 31.9:
        aps += 3
    elif t >= 38.5 or t <= 33.9:
        aps += 1
    elif t < 36 or t >= 38.5:
        aps += 1

    # MAP
    if map_mmhg >= 160 or map_mmhg <= 49:
        aps += 4
    elif map_mmhg >= 130 or map_mmhg <= 69:
        aps += 2

    # Heart rate
    if heart_rate >= 180 or heart_rate <= 39:
        aps += 4
    elif heart_rate >= 140 or heart_rate <= 54:
        aps += 3
    elif heart_rate >= 110 or heart_rate <= 69:
        aps += 2

    # GCS
    aps += (15 - gcs)

    # Age points
    if age >= 75:
        age_pts = 6
    elif age >= 65:
        age_pts = 5
    elif age >= 55:
        age_pts = 3
    elif age >= 45:
        age_pts = 2
    else:
        age_pts = 0

    score = aps + age_pts + chronic_health_points
    score = max(0, min(score, 71))

    if score <= 9:
        cat, rec = "Low", "Estimated mortality <10%."
    elif score <= 19:
        cat, rec = "Moderate", "Estimated mortality 10-25%."
    elif score <= 29:
        cat, rec = "High", "Estimated mortality 25-50%."
    else:
        cat, rec = "Very High", "Estimated mortality >50%."
    return {"score": score, "category": cat, "recommendation": rec}


def glasgow_coma_scale(eye: int, verbal: int, motor: int) -> dict[str, Any]:
    """Glasgow Coma Scale (3-15)."""
    score = eye + verbal + motor
    if score <= 8:
        cat, rec = "Severe", "Intubation likely required; ICU care."
    elif score <= 12:
        cat, rec = "Moderate", "Close monitoring; consider ICU."
    else:
        cat, rec = "Mild", "Observation; neurological monitoring."
    return {"score": score, "category": cat, "recommendation": rec}


def bmi(weight_kg: float, height_m: float) -> dict[str, Any]:
    """Body Mass Index."""
    val = round(weight_kg / (height_m ** 2), 1)
    if val < 18.5:
        cat, rec = "Underweight", "Nutritional assessment recommended."
    elif val < 25:
        cat, rec = "Normal", "Maintain healthy lifestyle."
    elif val < 30:
        cat, rec = "Overweight", "Lifestyle modifications recommended."
    elif val < 35:
        cat, rec = "Obese Class I", "Weight management program recommended."
    elif val < 40:
        cat, rec = "Obese Class II", "Intensive intervention recommended."
    else:
        cat, rec = "Obese Class III", "Consider bariatric surgery evaluation."
    return {"score": val, "category": cat, "recommendation": rec}


def bsa(weight_kg: float, height_cm: float) -> dict[str, Any]:
    """Body Surface Area (Du Bois formula)."""
    val = round(0.007184 * (weight_kg ** 0.425) * (height_cm ** 0.725), 2)
    return {"score": val, "category": f"{val} m²", "recommendation": "Used for drug dosing calculations."}


def framingham_10yr_cvd(
    age: int,
    sex: str,
    total_cholesterol: float,
    hdl: float,
    sbp: float,
    bp_treated: bool = False,
    smoker: bool = False,
    diabetes: bool = False,
) -> dict[str, Any]:
    """Framingham 10-year CVD risk (simplified)."""
    # Simplified point-based system
    is_female = sex.lower() in ("f", "female")
    pts = 0

    # Age
    if is_female:
        if age >= 70:
            pts += 12
        elif age >= 60:
            pts += 8
        elif age >= 50:
            pts += 4
        elif age >= 40:
            pts += 0
    else:
        if age >= 70:
            pts += 13
        elif age >= 60:
            pts += 9
        elif age >= 50:
            pts += 6
        elif age >= 40:
            pts += 0

    # Cholesterol
    if total_cholesterol >= 280:
        pts += 3
    elif total_cholesterol >= 240:
        pts += 2
    elif total_cholesterol >= 200:
        pts += 1

    # HDL
    if hdl < 35:
        pts += 2
    elif hdl < 45:
        pts += 1
    elif hdl >= 60:
        pts -= 1

    # SBP
    if bp_treated:
        if sbp >= 160:
            pts += 4
        elif sbp >= 140:
            pts += 3
        elif sbp >= 130:
            pts += 2
        elif sbp >= 120:
            pts += 1
    else:
        if sbp >= 160:
            pts += 3
        elif sbp >= 140:
            pts += 2
        elif sbp >= 130:
            pts += 1

    if smoker:
        pts += 2
    if diabetes:
        pts += 3 if is_female else 2

    # Rough risk mapping
    risk_pct = min(max(pts * 1.5, 1.0), 50.0)

    if risk_pct < 5:
        cat, rec = "Low", "Lifestyle modifications; reassess in 5 years."
    elif risk_pct < 10:
        cat, rec = "Borderline", "Lifestyle modifications; consider risk-enhancing factors."
    elif risk_pct < 20:
        cat, rec = "Intermediate", "Statin therapy may be beneficial; discuss with patient."
    else:
        cat, rec = "High", "Statin therapy recommended; aggressive risk factor management."
    return {"score": round(risk_pct, 1), "category": cat, "recommendation": rec}


def ascvd_pooled_cohort(
    age: int,
    sex: str,
    total_cholesterol: float,
    hdl: float,
    sbp: float,
    bp_treated: bool = False,
    smoker: bool = False,
    diabetes: bool = False,
) -> dict[str, Any]:
    """ASCVD Pooled Cohort Equations 10-year risk (simplified)."""
    # Simplified — uses Framingham-like approximation
    return framingham_10yr_cvd(age, sex, total_cholesterol, hdl, sbp, bp_treated, smoker, diabetes)


def has_bled(
    hypertension: bool = False,
    renal_disease: bool = False,
    liver_disease: bool = False,
    stroke_history: bool = False,
    prior_bleeding: bool = False,
    labile_inr: bool = False,
    age_gt65: bool = False,
    antiplatelet_or_nsaid: bool = False,
    alcohol: bool = False,
) -> dict[str, Any]:
    """HAS-BLED bleeding risk score."""
    score = sum([
        hypertension,
        renal_disease,
        liver_disease,
        stroke_history,
        prior_bleeding,
        labile_inr,
        age_gt65,
        antiplatelet_or_nsaid,
        alcohol,
    ])
    if score <= 2:
        cat, rec = "Low", "Low bleeding risk; anticoagulation generally safe."
    else:
        cat, rec = "High", "High bleeding risk; carefully weigh anticoagulation benefit vs risk."
    return {"score": score, "category": cat, "recommendation": rec}


def abcd2_stroke(
    age_ge60: bool = False,
    bp_ge140_90: bool = False,
    clinical_unilateral_weakness: bool = False,
    clinical_speech_impairment: bool = False,
    duration_ge60min: bool = False,
    duration_10_59min: bool = False,
    diabetes: bool = False,
) -> dict[str, Any]:
    """ABCD² score for TIA stroke risk."""
    score = 0
    if age_ge60:
        score += 1
    if bp_ge140_90:
        score += 1
    if clinical_unilateral_weakness:
        score += 2
    elif clinical_speech_impairment:
        score += 1
    if duration_ge60min:
        score += 2
    elif duration_10_59min:
        score += 1
    if diabetes:
        score += 1

    if score <= 3:
        cat, rec = "Low", "2-day stroke risk ~1%. Outpatient workup may be appropriate."
    elif score <= 5:
        cat, rec = "Moderate", "2-day stroke risk ~4%. Consider hospitalization."
    else:
        cat, rec = "High", "2-day stroke risk ~8%. Hospitalize for urgent workup."
    return {"score": score, "category": cat, "recommendation": rec}


def nih_stroke_scale(**domains: int) -> dict[str, Any]:
    """NIH Stroke Scale (simplified — sum of domain scores).

    Domains: consciousness (0-3), gaze (0-2), visual (0-3), facial_palsy (0-3),
    motor_arm_left (0-4), motor_arm_right (0-4), motor_leg_left (0-4),
    motor_leg_right (0-4), ataxia (0-2), sensory (0-2), language (0-3),
    dysarthria (0-2), neglect (0-2).
    """
    score = sum(domains.values())
    if score == 0:
        cat, rec = "No stroke symptoms", "No deficits detected."
    elif score <= 4:
        cat, rec = "Minor", "Minor stroke; consider thrombolysis if within window."
    elif score <= 15:
        cat, rec = "Moderate", "Moderate stroke; thrombolysis recommended if eligible."
    elif score <= 20:
        cat, rec = "Moderate-Severe", "Significant deficit; consider thrombectomy if LVO."
    else:
        cat, rec = "Severe", "Severe stroke; emergent intervention needed."
    return {"score": score, "category": cat, "recommendation": rec}


def bishop_score(
    dilation: int,
    effacement: int,
    station: int,
    consistency: int,
    position: int,
) -> dict[str, Any]:
    """Bishop Score for cervical readiness (each component 0-2 or 0-3)."""
    score = dilation + effacement + station + consistency + position
    if score >= 8:
        cat, rec = "Favorable", "Cervix favorable for induction; high likelihood of success."
    elif score >= 6:
        cat, rec = "Moderate", "Consider cervical ripening before induction."
    else:
        cat, rec = "Unfavorable", "Cervical ripening recommended prior to induction."
    return {"score": score, "category": cat, "recommendation": rec}


def apgar_score(
    appearance: int,
    pulse: int,
    grimace: int,
    activity: int,
    respiration: int,
) -> dict[str, Any]:
    """Apgar Score for newborn assessment (each component 0-2)."""
    score = appearance + pulse + grimace + activity + respiration
    if score >= 7:
        cat, rec = "Normal", "Routine neonatal care."
    elif score >= 4:
        cat, rec = "Moderately depressed", "Stimulation and possible interventions needed."
    else:
        cat, rec = "Severely depressed", "Immediate resuscitation required."
    return {"score": score, "category": cat, "recommendation": rec}


# Registry for dynamic lookup
CALCULATORS: dict[str, callable] = {
    "cha2ds2_vasc": cha2ds2_vasc,
    "heart_score": heart_score,
    "wells_dvt": wells_dvt,
    "wells_pe": wells_pe,
    "curb65": curb65,
    "qsofa": qsofa,
    "meld": meld,
    "child_pugh": child_pugh,
    "ckd_epi_egfr": ckd_epi_egfr,
    "apache_ii": apache_ii,
    "glasgow_coma_scale": glasgow_coma_scale,
    "bmi": bmi,
    "bsa": bsa,
    "framingham_10yr_cvd": framingham_10yr_cvd,
    "ascvd_pooled_cohort": ascvd_pooled_cohort,
    "has_bled": has_bled,
    "abcd2_stroke": abcd2_stroke,
    "nih_stroke_scale": nih_stroke_scale,
    "bishop_score": bishop_score,
    "apgar_score": apgar_score,
}
