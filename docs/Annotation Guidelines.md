# Annotation Guidelines

# VAERS ADE & DRUG Annotation Guidelines

## 1. Definitions

- **ADE (Adverse Drug Event)**  
  Any undesirable medical condition, symptom, diagnosis, or abnormal clinical finding occurring after vaccination, whether or not causality is proven.

- **DRUG**  
  The vaccine or drug product administered, annotated exactly as stated in the report.

---

## 2. General Principles

| Rule                                                             | Example ✅                            | Example ❌                                        |
|------------------------------------------------------------------|---------------------------------------|----------------------------------------------------|
| Include full medical term exactly as reported (no partial cuts). | "shortness of breath"                 | "breath"                                           |
| Include severity modifiers if part of ADE description.           | "severe headache"                     | "headache" (if severity given)                     |
| Exclude time duration from ADE span.                             | "fever" (from "fever for 2 days")     | "fever for 2 days"                                 |
| Annotate each ADE/DRUG mention separately, even if repeated.     | "fever", "headache", "fever"          | Merging both "fever"s                              |
| Use the longest medically accurate term when available.          | "acute respiratory distress syndrome" | "respiratory distress"                             |
| Keep casing exactly as in text.                                  | "shortness of breath"                 | "Shortness of Breath" (unnecessary capitalization) |

---

## 3. ADE Annotation Rules

| Case                              | Rule                                                                     | Example                                                                                             |
|-----------------------------------|--------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Specific symptom or diagnosis     | Mark full phrase describing the condition.                               | "extreme fatigue" → ADE "extreme fatigue"                                                           |
| Resolved/improved symptoms        | Still annotate; status words ("subsided", "recovered") don’t exclude it. | "headache subsided" → ADE "headache"                                                                |
| Lab results with clinical meaning | Annotate if abnormal finding is stated.                                  | "fever 101" → ADE "fever"                                                                           |
| Vague/non-clinical terms          | Skip overly general phrases.                                             | "not feeling well" → Skip                                                                           |
| Pre-existing conditions           | Skip unless worsened post-vaccination.                                   | "bone marrow disorder" (pre-existing) → Skip                                                        |
| Administrative issues             | Skip unless followed by a symptom.                                       | "product dose omission" → Skip                                                                      |
| Breakthrough infection            | ADE if after vaccination.                                                | "COVID-19 infection after vaccination"                                                              |
| Multi-word ADEs                   | Keep whole phrase if medically one concept; exclude unrelated info.      | ✅ "chronic obstructive pulmonary disease" / ❌ "chronic obstructive pulmonary disease causing cough" |
| Compound ADE phrases              | If separate symptoms, split; if fixed expression, keep as one.           | "fever and headache" → "fever", "headache" / "nausea and vomiting" → one span (classic example of ADE spans that are very often seen together in VAERS and other clinical safety datasets.)                      |

---

## 4. DRUG Annotation Rules

| Case                              | Rule                               | Example                                 |
|-----------------------------------|------------------------------------|-----------------------------------------|
| Exact name given                  | Annotate full product name.        | "Moderna COVID-19 vaccine"              |
| Code name given                   | Annotate as written.               | "mRNA-1273"                             |
| Both brand & code mentioned       | Annotate both separately.          | "mRNA-1273", "Moderna COVID-19 vaccine" |
| Generic mention                   | Annotate exactly as in text.       | "vaccine", "COVID-19 vaccinated"        |
| Brand not given                   | Do not guess — keep as stated.     | "received the vaccine" → DRUG "vaccine" |
| Vague mention with disease target | Include full phrase.               | "COVID-19 vaccinated"                   |
| Booster/general terms             | Annotate if clearly product given. | "booster shot"                          |
| Past or unrelated drug            | Skip.                              | "chemotherapy" (past use)               |
| Multiple occurrences              | Label each separately.             | "Pfizer vaccine ... Pfizer vaccine"     |

---

## 5. Skip Annotation When

- Administrative only, no symptom (e.g., “product dose omission”).
- Vague, non-clinical, or purely numeric (e.g., “body temperature”, “BP: 180/110” or “not feeling well” without diagnosis).
- Clearly pre-existing, unrelated to vaccination.
- Past/discontinued drug use.
- Non-medical events (e.g., “fell asleep”).

---

## 6. Special Notes

- Repeat all ADE/DRUG mentions — do not merge.
- Exclude time references from ADE spans.
- **Death** is not ADE unless cause is stated.  
  - "Patient died due to cardiac arrest" → ADE: "cardiac arrest", **Not**: "died".
  - "Patient died" → Skip ADE (no medical event named).
  - "Death occurred two days after vaccination" → Skip ADE (outcome only).
- Hospitalization is not ADE, but the reason for it might be.
- Synonyms in same record (e.g., "myocardial infarction", "heart attack") are both annotated as ADE.

### COVID-19 Mentions

✅ When COVID-19 **is** an ADE:  
- "COVID-19 infection" / "COVID-19 pneumonia"  
- "tested positive for COVID-19" → ADE "COVID-19"  
- "COVID-19 vaccination" → DRUG

❌ When COVID-19 is **NOT** an ADE:  
1. Only part of vaccine name.  
   - "Moderna COVID-19 vaccine" → DRUG only, no ADE.  
   - "COVID-19 vaccination completed" → DRUG only.
2. Past history before vaccination.  
   - "History of COVID-19 last year" → Skip.
3. Risk factor or contact history, not actual infection.  
   - "Exposed to COVID-19 positive person" → Skip.
4. Mentioned in a negative test result.  
   - "Tested negative for COVID-19" → Skip.

---

## 7. Quick ADE Span Checklist

✅ Symptom/diagnosis present  
✅ Post-vaccination (not history only)  
✅ Specific & clinically meaningful (not vague)  
✅ Include severity if given (e.g., `severe headache`)  
✅ Include location if given (e.g., `left arm swelling`)  
✅ Duration/time excluded  
✅ Administrative terms excluded unless symptom follows  
✅ Multi-word kept as full medical concept  
✅ Each occurrence labelled separately  
