import spacy

nlp = spacy.load("en_core_web_sm")

samples = {
    "001": "(& Recommended | Hiro HuaNG\nAlbany, NY 12208 @ (555)535-5555 @ Hiro Huang(@cxample.com\nPROFESSIONAL SUMMARY",
    "002": "AYA NAKAMURA\nLAKESIDE, CA 92041 (555)555-5555 AYA NAKAMURAGEXAMPLE.COM\nRESUME OBJECTIVE",
    "003": "Spanish\nFeiner a)\nFeiner a)\nFeiner a)\nMING DAVIS\nPROFESSIONAL SUMMARY",
    "004": "DAVID ANDERSON\nPROFESSIONAL SUMMARY\nDynamic Blue Collar Worker",
}

for fname, text in samples.items():
    doc = nlp(text)
    print(fname, "->", [ent.text for ent in doc.ents if ent.label_ == "PERSON"])