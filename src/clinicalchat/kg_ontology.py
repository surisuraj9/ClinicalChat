"""Curated clinical ontology used to seed the knowledge graph.

Kept as plain data so it is easy to review clinically and to extend. Content is educational
and intentionally conservative; it is not a substitute for a licensed drug-interaction database.
"""

from __future__ import annotations

# --- Drugs ---------------------------------------------------------------------------------
# canonical drug -> (drug classes, aliases / brand names)
DRUGS = {
    "aspirin": (["NSAID (non-selective COX inhibitor)", "Antiplatelet"], ["acetylsalicylic acid", "ecosprin", "disprin"]),
    "ibuprofen": (["NSAID (non-selective COX inhibitor)"], ["brufen", "advil", "motrin", "combiflam"]),
    "naproxen": (["NSAID (non-selective COX inhibitor)"], ["aleve", "naprosyn"]),
    "diclofenac": (["NSAID (non-selective COX inhibitor)"], ["voltaren", "voveran"]),
    "celecoxib": (["COX-2 selective NSAID", "Non-antibiotic sulfonamide"], ["celebrex"]),
    "paracetamol": (["Analgesic / antipyretic"], ["acetaminophen", "tylenol", "crocin", "dolo", "calpol", "panadol"]),
    "sulfamethoxazole": (["Sulfonamide antibiotic"], ["co-trimoxazole", "cotrimoxazole", "bactrim", "septran", "smx-tmp", "trimethoprim-sulfamethoxazole"]),
    "sulfadiazine": (["Sulfonamide antibiotic"], []),
    "furosemide": (["Non-antibiotic sulfonamide", "Loop diuretic"], ["lasix"]),
    "hydrochlorothiazide": (["Non-antibiotic sulfonamide", "Thiazide diuretic"], ["hctz"]),
    "amoxicillin": (["Penicillin antibiotic"], ["amoxil", "augmentin", "mox"]),
    "azithromycin": (["Macrolide antibiotic"], ["zithromax", "azithral"]),
    "clarithromycin": (["Macrolide antibiotic"], ["biaxin"]),
    "ceftriaxone": (["Cephalosporin antibiotic"], ["rocephin"]),
    "clopidogrel": (["Antiplatelet"], ["plavix", "clopilet"]),
    "warfarin": (["Anticoagulant"], ["coumadin"]),
    "tenecteplase": (["Thrombolytic"], ["tnk"]),
    "metoprolol": (["Beta-blocker (cardioselective)"], ["lopressor", "metolar"]),
    "propranolol": (["Beta-blocker (non-selective)"], ["inderal"]),
    "atorvastatin": (["Statin"], ["lipitor", "atorva"]),
    "ramipril": (["ACE inhibitor"], ["altace", "cardace"]),
    "spironolactone": (["Potassium-sparing diuretic"], ["aldactone"]),
    "metformin": (["Biguanide"], ["glucophage", "glycomet"]),
    "levothyroxine": (["Thyroid hormone"], ["thyronorm", "eltroxin", "synthroid"]),
    "ferrous sulfate": (["Iron supplement"], ["iron supplement", "iron tablets", "oral iron"]),
    "calcium carbonate": (["Calcium supplement"], ["calcium supplement"]),
    "cetirizine": (["Antihistamine"], ["zyrtec", "cetzine"]),
    "epinephrine": (["Sympathomimetic (anaphylaxis rescue)"], ["adrenaline", "epipen", "epinephrine auto-injector"]),
    "salbutamol": (["Short-acting beta-2 agonist"], ["albuterol", "asthalin", "ventolin"]),
    "fluticasone/salmeterol dry-powder inhaler": (["Inhaled corticosteroid", "Lactose-containing dry-powder inhaler"], ["advair diskus", "seretide accuhaler"]),
    "budesonide": (["Inhaled corticosteroid"], ["pulmicort"]),
    "montelukast": (["Leukotriene receptor antagonist"], ["singulair", "montair"]),
    "prednisolone": (["Systemic corticosteroid"], ["prednisone", "wysolone"]),
    "omeprazole": (["Proton pump inhibitor"], ["prilosec", "omez"]),
    "pantoprazole": (["Proton pump inhibitor"], ["pantocid", "protonix"]),
    "iodinated contrast": (["Radiocontrast agent"], ["contrast dye", "ct contrast", "iv contrast"]),
    "arachis oil preparations": (["Peanut-oil-containing excipient"], ["arachis oil", "peanut oil"]),
}

# --- Allergens & cross-reactivity ------------------------------------------------------------
# canonical allergen -> (allergy category, aliases)
ALLERGENS = {
    "peanut": ("Food", ["peanuts", "groundnut"]),
    "tree nuts": ("Food", ["almond", "cashew", "walnut", "hazelnut"]),
    "lupin": ("Food", []),
    "soy": ("Food", ["soya", "soybean"]),
    "milk": ("Food", ["cow's milk", "dairy", "milk protein"]),
    "goat/sheep milk": ("Food", ["goat milk", "sheep milk"]),
    "wheat": ("Food", []),
    "gluten": ("Food", ["coeliac", "celiac"]),
    "barley/rye": ("Food", ["barley", "rye"]),
    "fish": ("Food", ["finfish", "cod", "salmon", "tuna"]),
    "shellfish": ("Food", ["shrimp", "prawn", "crab", "lobster"]),
    "dust mite": ("Environmental", ["dust mites", "house dust mite"]),
    "cockroach": ("Environmental", ["cockroaches"]),
    "pollen": ("Environmental", ["birch pollen", "grass pollen", "hay fever"]),
    "pet dander": ("Environmental", ["cat dander", "dog dander", "animal dander"]),
    "oral allergy foods (apple, carrot, celery)": ("Food", ["apple", "carrot", "celery"]),
    "latex": ("Environmental", []),
    "nickel": ("Skin", ["nickel (jewelry)", "nickel jewelry"]),
    "cobalt": ("Skin", []),
    "cold urticaria trigger": ("Skin", ["cold exposure", "cold urticaria"]),
    "ultraviolet light": ("Skin", ["uv-a light", "uv light", "sunlight"]),
    "sulfamethoxazole": ("Medication", ["sulfa", "sulfonamide", "sulpha", "co-trimoxazole", "bactrim"]),
    "ibuprofen": ("Medication", ["nsaid", "nsaids"]),
    "paracetamol": ("Medication", ["acetaminophen"]),
    "penicillin": ("Medication", ["amoxicillin", "penicillins"]),
}

# (allergen a, allergen b, risk, mechanism)
CROSS_REACTIVITY = [
    ("peanut", "lupin", "HIGH", "Legume proteins; lupin flour is a well-described peanut cross-reactant"),
    ("peanut", "tree nuts", "MODERATE", "Frequent co-sensitisation (~30-40%) though botanically unrelated"),
    ("peanut", "soy", "LOW", "Shared legume storage proteins; clinically relevant in a minority"),
    ("milk", "goat/sheep milk", "HIGH", "Caseins are >85% homologous; goat/sheep milk is not a safe substitute"),
    ("wheat", "gluten", "HIGH", "Gluten (gliadin/glutenin) is a wheat protein"),
    ("wheat", "barley/rye", "MODERATE", "Related cereal prolamins"),
    ("fish", "fish", "HIGH", "Parvalbumin: ~50% of fish-allergic patients react to other finfish species"),
    ("dust mite", "cockroach", "MODERATE", "Tropomyosin pan-allergen"),
    ("dust mite", "shellfish", "MODERATE", "Tropomyosin pan-allergen (mite-shellfish syndrome)"),
    ("cockroach", "shellfish", "MODERATE", "Tropomyosin pan-allergen"),
    ("pollen", "oral allergy foods (apple, carrot, celery)", "MODERATE", "Pollen-food (oral allergy) syndrome via PR-10 proteins"),
    ("nickel", "cobalt", "MODERATE", "Common metal co-sensitisation in contact dermatitis"),
    ("pet dander", "pet dander", "MODERATE", "Cat and dog allergens commonly co-sensitise"),
]

# allergen -> drugs/classes to avoid: (target, kind 'drug'|'class', risk, reason)
ALLERGEN_DRUG_RULES = {
    "sulfamethoxazole": [
        ("Sulfonamide antibiotic", "class", "HIGH", "Same sulfonamide-antibiotic arylamine structure; avoid all sulfonamide antibiotics"),
        ("celecoxib", "drug", "MODERATE", "Labelled contraindication in patients with sulfonamide allergy"),
        ("Non-antibiotic sulfonamide", "class", "LOW", "Cross-reactivity with non-antibiotic sulfonamides is uncommon; use with monitoring"),
    ],
    "ibuprofen": [
        ("NSAID (non-selective COX inhibitor)", "class", "HIGH", "COX-1 inhibition cross-reactivity across non-selective NSAIDs"),
        ("COX-2 selective NSAID", "class", "LOW", "Selective COX-2 inhibitors are usually tolerated; consider supervised challenge"),
    ],
    "paracetamol": [
        ("paracetamol", "drug", "HIGH", "Documented paracetamol hypersensitivity; check combination products (cold/flu remedies)"),
    ],
    "penicillin": [
        ("Penicillin antibiotic", "class", "HIGH", "Shared beta-lactam ring and side chains"),
        ("Cephalosporin antibiotic", "class", "LOW", "Low (~1-2%) cross-reactivity; higher with similar side chains"),
    ],
    "milk": [
        ("Lactose-containing dry-powder inhaler", "class", "MODERATE", "Lactose carrier may contain milk protein; labelled contraindication in severe milk protein allergy"),
    ],
    "peanut": [
        ("Peanut-oil-containing excipient", "class", "HIGH", "Arachis (peanut) oil excipients are contraindicated in peanut allergy"),
    ],
}

# allergen -> safer alternatives (drug or advice)
ALLERGEN_ALTERNATIVES = {
    "ibuprofen": ["paracetamol", "celecoxib (under supervision)"],
    "paracetamol": ["ibuprofen (if no NSAID intolerance)", "non-drug measures"],
    "sulfamethoxazole": ["amoxicillin (if no penicillin allergy)", "azithromycin"],
    "penicillin": ["azithromycin", "ceftriaxone (low risk, monitor)"],
    "milk": ["pressurised metered-dose inhaler instead of lactose dry-powder inhaler"],
}

# (drug a, drug b, severity, effect, management)
DRUG_INTERACTIONS = [
    ("aspirin", "clopidogrel", "MODERATE", "Additive bleeding risk (intended in dual antiplatelet therapy)", "Monitor for bleeding; consider PPI (pantoprazole) gastroprotection"),
    ("ibuprofen", "aspirin", "MAJOR", "Ibuprofen blocks aspirin's cardioprotective antiplatelet effect and adds GI bleeding risk", "Avoid regular ibuprofen with low-dose aspirin; prefer paracetamol"),
    ("naproxen", "aspirin", "MAJOR", "Reduced antiplatelet effect and GI bleeding", "Avoid combination"),
    ("diclofenac", "aspirin", "MAJOR", "Additive GI bleeding risk", "Avoid combination"),
    ("ibuprofen", "clopidogrel", "MAJOR", "Increased GI bleeding risk", "Avoid; use paracetamol for analgesia"),
    ("ibuprofen", "warfarin", "MAJOR", "Increased bleeding risk", "Avoid; use paracetamol"),
    ("aspirin", "warfarin", "MAJOR", "Increased bleeding risk", "Only with specialist indication; monitor INR"),
    ("clopidogrel", "omeprazole", "MODERATE", "Omeprazole inhibits CYP2C19 activation of clopidogrel", "Use pantoprazole instead"),
    ("ramipril", "ibuprofen", "MODERATE", "Reduced antihypertensive effect and acute kidney injury risk", "Avoid in CKD; monitor renal function"),
    ("ramipril", "spironolactone", "MAJOR", "Hyperkalaemia", "Monitor potassium closely"),
    ("ramipril", "sulfamethoxazole", "MODERATE", "Trimethoprim component raises potassium", "Monitor potassium"),
    ("warfarin", "sulfamethoxazole", "MAJOR", "Markedly increased INR and bleeding", "Avoid or reduce warfarin dose with close INR monitoring"),
    ("metoprolol", "salbutamol", "MODERATE", "Beta-blockade may blunt bronchodilator response", "Prefer cardioselective agents at low dose in asthma"),
    ("propranolol", "salbutamol", "MAJOR", "Non-selective beta-blocker antagonises bronchodilation and can trigger bronchospasm", "Avoid in asthma"),
    ("propranolol", "epinephrine", "MAJOR", "Unopposed alpha effect (severe hypertension) and epinephrine-resistant anaphylaxis", "Avoid non-selective beta-blockers in patients at anaphylaxis risk"),
    ("levothyroxine", "ferrous sulfate", "MODERATE", "Iron reduces levothyroxine absorption", "Separate doses by at least 4 hours"),
    ("levothyroxine", "calcium carbonate", "MODERATE", "Calcium reduces levothyroxine absorption", "Separate doses by at least 4 hours"),
    ("atorvastatin", "clarithromycin", "MAJOR", "CYP3A4 inhibition increases statin levels (myopathy/rhabdomyolysis)", "Use azithromycin or pause statin"),
    ("metformin", "iodinated contrast", "MODERATE", "Contrast-induced AKI may cause metformin accumulation and lactic acidosis", "Hold metformin around contrast if eGFR < 60; recheck renal function at 48 h"),
    ("clopidogrel", "warfarin", "MAJOR", "Increased bleeding risk", "Specialist review required"),
]

# condition -> drugs/classes to avoid: (target, kind, severity, reason)
CONDITION_CONTRAINDICATIONS = {
    "Asthma": [
        ("Beta-blocker (non-selective)", "class", "MAJOR", "Can precipitate severe bronchospasm"),
        ("NSAID (non-selective COX inhibitor)", "class", "MODERATE", "Risk of aspirin/NSAID-exacerbated respiratory disease"),
    ],
    "Chronic kidney disease": [
        ("NSAID (non-selective COX inhibitor)", "class", "MAJOR", "Nephrotoxic; can worsen renal function"),
        ("metformin", "drug", "MODERATE", "Reduce dose if eGFR 30-45; contraindicated if eGFR < 30"),
        ("iodinated contrast", "drug", "MODERATE", "Risk of contrast-associated nephropathy; hydrate and reassess"),
        ("spironolactone", "drug", "MODERATE", "Hyperkalaemia risk"),
    ],
    "Myocardial infarction": [
        ("NSAID (non-selective COX inhibitor)", "class", "MODERATE", "NSAIDs increase cardiovascular event risk after MI (aspirin excepted when indicated)"),
    ],
    "Hyperkalemia": [
        ("ramipril", "drug", "MODERATE", "ACE inhibitors raise potassium"),
        ("spironolactone", "drug", "MAJOR", "Potassium-sparing diuretic"),
    ],
    "Liver dysfunction": [
        ("paracetamol", "drug", "MODERATE", "Reduce maximum daily dose in hepatic impairment"),
        ("atorvastatin", "drug", "MODERATE", "Monitor LFTs; avoid in active liver disease"),
    ],
    "Anemia": [
        ("NSAID (non-selective COX inhibitor)", "class", "MINOR", "GI blood loss can worsen anaemia"),
    ],
}

# --- Diagnoses, tests, symptoms, treatments --------------------------------------------------
# condition -> keywords used to recognise it in free text
CONDITION_KEYWORDS = {
    "Anemia": ["anemia", "anaemia"],
    "Iron deficiency": ["iron deficiency", "iron-deficiency"],
    "Type 2 diabetes mellitus": ["diabetes", "diabetic", "t2dm"],
    "Prediabetes": ["prediabetes", "prediabetic", "impaired fasting"],
    "Dyslipidemia": ["dyslipidemia", "dyslipidaemia", "hyperlipidemia", "hypercholesterolemia"],
    "Hypothyroidism": ["hypothyroid"],
    "Hashimoto's thyroiditis": ["hashimoto", "autoimmune thyroid"],
    "Hyperthyroidism": ["hyperthyroid", "graves"],
    "Chronic kidney disease": ["chronic kidney disease", "ckd", "renal impairment", "kidney dysfunction"],
    "Liver dysfunction": ["fatty liver", "hepatitis", "liver dysfunction", "liver injury", "nafld", "hepatocellular"],
    "Myocardial infarction": ["myocardial infarction", "stemi", "nstemi", "heart attack"],
    "Coronary artery disease": ["coronary artery disease", "ischemic heart", "ischaemic heart"],
    "Hypertension": ["hypertension", "high blood pressure"],
    "Asthma": ["asthma", "bronchial asthma"],
    "Pneumonia": ["pneumonia", "consolidation"],
    "Obesity": ["obesity", "obese"],
    "Allergic rhinitis": ["allergic rhinitis", "hay fever"],
    "Anaphylaxis": ["anaphylaxis", "anaphylactic"],
    "Hyperkalemia": ["hyperkalemia", "hyperkalaemia"],
    "Infection / inflammation": ["infection", "sepsis", "inflammation"],
    "Disc herniation": ["disc herniation", "disc bulge", "herniated", "prolapse"],
    "Kidney stones": ["renal calculus", "kidney stone", "nephrolithiasis", "calculi"],
    "Gallstones": ["cholelithiasis", "gallstone", "gall stone"],
    "Stroke": ["stroke", "infarct", "cerebrovascular"],
    "Fracture": ["fracture"],
    "Vitamin D deficiency": ["vitamin d deficiency", "vit d deficiency"],
    "Gout / hyperuricemia": ["gout", "hyperuricemia", "hyperuricaemia"],
    "Coagulopathy": ["coagulopathy", "prolonged pt", "bleeding disorder"],
    "Thrombocytopenia": ["thrombocytopenia"],
    "Viral respiratory infection": ["viral fever", "common cold", "influenza", "flu", "upper respiratory infection", "urti"],
    "Gastroenteritis": ["gastroenteritis", "stomach flu", "food poisoning"],
    "Migraine": ["migraine"],
    "Allergic skin reaction": ["urticaria", "dermatitis", "eczema", "allergic rash"],
}

# canonical test -> (regex patterns matched against lower-cased parameter names)
TESTS = {
    "Hemoglobin": [r"\bh(a)?emoglobin\b(?!.*a1c)", r"^hb$"],
    "MCV": [r"\bmcv\b"],
    "RBC count": [r"\brbc\b.*count|red (blood )?cell count"],
    "WBC count": [r"\bwbc\b.*count|total leu?kocyte|white blood cell"],
    "Platelet count": [r"platelet"],
    "Eosinophils": [r"eosinophil"],
    "HbA1c": [r"hba1c|glycated|a1c"],
    "Fasting glucose": [r"fasting.*(glucose|sugar)|\bfbs\b"],
    "Postprandial glucose": [r"post ?prandial|post lunch|\bppbs\b"],
    "LDL cholesterol": [r"\bldl\b"],
    "HDL cholesterol": [r"\bhdl\b"],
    "Total cholesterol": [r"total cholesterol|^cholesterol"],
    "Triglycerides": [r"triglyceride"],
    "TSH": [r"\btsh\b"],
    "T4": [r"\b(free )?t4\b|thyroxine"],
    "Thyroid peroxidase antibody": [r"\btpo\b|anti-thyroid"],
    "Creatinine": [r"creatinine(?!.*clearance)"],
    "eGFR": [r"\begfr\b"],
    "BUN / urea": [r"\bbun\b|\burea\b"],
    "Potassium": [r"potassium"],
    "Sodium": [r"sodium"],
    "ALT": [r"\balt\b|sgpt"],
    "AST": [r"\bast\b|sgot"],
    "GGT": [r"\bggt\b|gamma ?gt|gama gt"],
    "ALP": [r"\balp\b|alkaline phosphatase|alkpo4"],
    "Bilirubin": [r"bilirubin"],
    "Albumin": [r"^albumin|serum albumin"],
    "Prothrombin time / INR": [r"prothrombin|\binr\b"],
    "Troponin": [r"troponin"],
    "Total IgE": [r"\bige\b"],
    "Uric acid": [r"uric acid"],
    "Vitamin D": [r"vitamin d|25-oh"],
    "Ferritin": [r"ferritin"],
    "CRP": [r"\bcrp\b|c-reactive"],
    "BMI": [r"\bbmi\b|body mass"],
}

# (test, direction HIGH|LOW, condition)
TEST_INDICATES = [
    ("Hemoglobin", "LOW", "Anemia"), ("RBC count", "LOW", "Anemia"),
    ("MCV", "LOW", "Iron deficiency"), ("Ferritin", "LOW", "Iron deficiency"),
    ("WBC count", "HIGH", "Infection / inflammation"), ("CRP", "HIGH", "Infection / inflammation"),
    ("Platelet count", "LOW", "Thrombocytopenia"),
    ("Eosinophils", "HIGH", "Allergic rhinitis"), ("Total IgE", "HIGH", "Allergic rhinitis"),
    ("HbA1c", "HIGH", "Type 2 diabetes mellitus"), ("Fasting glucose", "HIGH", "Type 2 diabetes mellitus"),
    ("Postprandial glucose", "HIGH", "Type 2 diabetes mellitus"),
    ("LDL cholesterol", "HIGH", "Dyslipidemia"), ("Total cholesterol", "HIGH", "Dyslipidemia"),
    ("Triglycerides", "HIGH", "Dyslipidemia"), ("HDL cholesterol", "LOW", "Dyslipidemia"),
    ("TSH", "HIGH", "Hypothyroidism"), ("T4", "LOW", "Hypothyroidism"),
    ("TSH", "LOW", "Hyperthyroidism"), ("Thyroid peroxidase antibody", "HIGH", "Hashimoto's thyroiditis"),
    ("Creatinine", "HIGH", "Chronic kidney disease"), ("eGFR", "LOW", "Chronic kidney disease"),
    ("BUN / urea", "HIGH", "Chronic kidney disease"), ("Potassium", "HIGH", "Hyperkalemia"),
    ("ALT", "HIGH", "Liver dysfunction"), ("AST", "HIGH", "Liver dysfunction"), ("GGT", "HIGH", "Liver dysfunction"),
    ("Bilirubin", "HIGH", "Liver dysfunction"), ("Albumin", "LOW", "Liver dysfunction"),
    ("Prothrombin time / INR", "HIGH", "Coagulopathy"),
    ("Troponin", "HIGH", "Myocardial infarction"),
    ("Uric acid", "HIGH", "Gout / hyperuricemia"), ("Vitamin D", "LOW", "Vitamin D deficiency"),
    ("BMI", "HIGH", "Obesity"),
]

# symptom -> conditions it can suggest
SYMPTOMS = {
    "chest pain": ["Myocardial infarction", "Coronary artery disease", "Pneumonia"],
    "shortness of breath": ["Asthma", "Pneumonia", "Anemia", "Myocardial infarction", "Anaphylaxis"],
    "breathlessness": ["Asthma", "Pneumonia", "Anemia"],
    "wheezing": ["Asthma", "Anaphylaxis"],
    "cough": ["Viral respiratory infection", "Asthma", "Pneumonia"],
    "fever": ["Viral respiratory infection", "Pneumonia", "Infection / inflammation"],
    "fatigue": ["Anemia", "Hypothyroidism", "Type 2 diabetes mellitus", "Chronic kidney disease"],
    "weakness": ["Anemia", "Hypothyroidism"],
    "pallor": ["Anemia"],
    "weight gain": ["Hypothyroidism", "Obesity"],
    "cold intolerance": ["Hypothyroidism"],
    "increased thirst": ["Type 2 diabetes mellitus"],
    "frequent urination": ["Type 2 diabetes mellitus"],
    "swelling": ["Chronic kidney disease", "Anaphylaxis"],
    "jaundice": ["Liver dysfunction"],
    "hives": ["Anaphylaxis", "Allergic rhinitis"],
    "sneezing": ["Allergic rhinitis"],
    "itchy eyes": ["Allergic rhinitis"],
    "throat tightness": ["Anaphylaxis"],
    "back pain": ["Disc herniation", "Kidney stones"],
    "flank pain": ["Kidney stones"],
    "abdominal pain": ["Gallstones", "Kidney stones", "Liver dysfunction"],
    "joint pain": ["Gout / hyperuricemia"],
    "headache": ["Migraine", "Viral respiratory infection", "Hypertension", "Stroke"],
    "weakness on one side": ["Stroke"],
    "sweating": ["Myocardial infarction"],
    "sore throat": ["Viral respiratory infection"],
    "runny nose": ["Viral respiratory infection", "Allergic rhinitis"],
    "body ache": ["Viral respiratory infection"],
    "vomiting": ["Gastroenteritis"],
    "nausea": ["Gastroenteritis", "Liver dysfunction"],
    "diarrhea": ["Gastroenteritis"],
    "dizziness": ["Anemia", "Hypertension"],
    "fainting": ["Myocardial infarction", "Anemia"],
    "palpitations": ["Hyperthyroidism", "Anemia"],
    "rash": ["Allergic skin reaction"],
    "confusion": ["Stroke"],
    "coughing blood": ["Pneumonia"],
}

# everyday phrasing -> canonical symptom (matched with word boundaries, after negation filtering)
SYMPTOM_SYNONYMS = {
    "short of breath": "shortness of breath", "difficulty breathing": "shortness of breath",
    "breathing difficulty": "shortness of breath", "can't breathe": "shortness of breath",
    "cannot breathe": "shortness of breath", "tired": "fatigue", "tiredness": "fatigue", "exhausted": "fatigue",
    "low energy": "fatigue", "temperature": "fever", "feverish": "fever", "throwing up": "vomiting",
    "vomit": "vomiting", "vomited": "vomiting", "loose motion": "diarrhea", "loose motions": "diarrhea",
    "diarrhoea": "diarrhea", "dizzy": "dizziness", "giddiness": "dizziness", "lightheaded": "dizziness",
    "fainted": "fainting", "passed out": "fainting", "blackout": "fainting", "chest tightness": "chest pain",
    "chest discomfort": "chest pain", "stomach pain": "abdominal pain", "stomach ache": "abdominal pain",
    "tummy ache": "abdominal pain", "belly pain": "abdominal pain", "itching": "rash", "itchy skin": "rash",
    "racing heart": "palpitations", "heart racing": "palpitations", "body pain": "body ache",
    "body aches": "body ache", "muscle ache": "body ache", "muscle pain": "body ache", "throat pain": "sore throat",
    "blocked nose": "runny nose", "stuffy nose": "runny nose", "slurred speech": "weakness on one side",
    "face drooping": "weakness on one side", "yellow eyes": "jaundice", "yellow skin": "jaundice",
    "coughing up blood": "coughing blood", "blood in sputum": "coughing blood", "breathless": "breathlessness",
    "lips swelling": "swelling", "swollen": "swelling", "nauseous": "nausea", "feel sick": "nausea",
}

# condition -> (confirmatory investigations, typical treatments)
CONDITION_WORKUP = {
    "Anemia": (["CBC", "Peripheral smear", "Ferritin"], ["ferrous sulfate"]),
    "Iron deficiency": (["Ferritin", "Iron studies"], ["ferrous sulfate"]),
    "Type 2 diabetes mellitus": (["HbA1c", "Fasting glucose"], ["metformin", "Lifestyle modification"]),
    "Dyslipidemia": (["Lipid profile"], ["atorvastatin", "Lifestyle modification"]),
    "Hypothyroidism": (["TSH", "Free T4"], ["levothyroxine"]),
    "Chronic kidney disease": (["eGFR", "Urine albumin-creatinine ratio", "Ultrasound KUB"], ["ramipril", "Blood pressure control"]),
    "Liver dysfunction": (["LFT", "Ultrasound abdomen"], ["Alcohol cessation / weight loss"]),
    "Myocardial infarction": (["ECG", "Troponin", "Echocardiogram"], ["aspirin", "clopidogrel", "metoprolol", "atorvastatin", "ramipril"]),
    "Asthma": (["Spirometry", "Peak flow"], ["salbutamol", "budesonide", "montelukast"]),
    "Pneumonia": (["Chest X-ray", "CBC", "CRP"], ["amoxicillin", "azithromycin"]),
    "Allergic rhinitis": (["Specific IgE", "Skin prick test"], ["cetirizine", "Allergen avoidance"]),
    "Anaphylaxis": (["Serum tryptase", "Specific IgE"], ["epinephrine"]),
    "Disc herniation": (["MRI spine"], ["Physiotherapy", "paracetamol"]),
    "Kidney stones": (["Ultrasound KUB", "CT KUB"], ["Hydration", "paracetamol"]),
    "Gallstones": (["Ultrasound abdomen"], ["Surgical review"]),
    "Stroke": (["CT brain", "MRI brain"], ["Stroke team activation"]),
    "Hyperkalemia": (["ECG", "Repeat potassium"], ["Review potassium-raising drugs"]),
    "Gout / hyperuricemia": (["Serum uric acid"], ["Lifestyle modification"]),
    "Obesity": (["BMI", "Waist circumference"], ["Lifestyle modification"]),
    "Viral respiratory infection": (["Clinical examination", "CBC", "CRP"], ["Rest and fluids", "paracetamol"]),
    "Gastroenteritis": (["Clinical examination", "Stool test", "Electrolytes"], ["Oral rehydration"]),
    "Migraine": (["Clinical examination"], ["Headache diary", "paracetamol"]),
    "Allergic skin reaction": (["Clinical examination", "Skin prick test"], ["cetirizine", "Allergen avoidance"]),
}
