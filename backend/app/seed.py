"""SYNTHETIC stand-ins for the three official HackFusion data assets.

The official Grievance Category Master, Mock Grievance History and Department
Directory are not released yet. Everything here is generated and clearly
labelled synthetic; loaders in engine.py read from the database, so the real
files can be swapped in by replacing this module's output.
"""
import random
from datetime import datetime, timedelta

LOCALITIES = [("Shivaji Nagar", "शिवाजी नगर"), ("Vazirabad", "वजीराबाद"), ("Taroda", "तरोडा"),
              ("CIDCO", "सिडको"), ("Shrinagar", "श्रीनगर"), ("Bhagya Nagar", "भाग्य नगर"),
              ("Vishnupuri", "विष्णुपुरी"), ("Itwara", "इतवारा")]

# id, category, subcategory, department, assignee, sla_days, sensitive, identity_hiding, auto_ok, severity
CATEGORIES = [
    dict(id="CM-01", category="Street Lighting", subcategory="Streetlight not working", department="Electrical",
         assignee="Junior Engineer (Electrical)", sla_days=3, sensitive=0, identity_hiding=0, auto_ok=1, severity=2,
         keywords=["streetlight", "street light", "street lamp", "light pole", "lamp post", "पथदिवा", "पथदिवे", "रस्त्यावरील दिवा",
                   "रस्त्यावरील दिवे", "स्ट्रीट लाइट", "स्ट्रीट लाईट", "स्ट्रीटलाइट", "खांबावरील दिवा", "बत्ती", "street ki light", "khamba"],
         resolution="Faulty lamp replaced and line checked by the Electrical department."),
    dict(id="CM-02", category="Water Supply", subcategory="No water / leakage", department="Water Supply",
         assignee="Deputy Engineer (Water)", sla_days=2, sensitive=0, identity_hiding=0, auto_ok=0, severity=4,
         keywords=["water supply", "no water", "pipeline", "pipe leak", "leakage", "tap water", "पाणी", "नळ", "जलवाहिनी", "पानी", "नल",
                   "पाइपलाइन", "पाईपलाईन", "paani", "pani nahi"],
         resolution="Pipeline repaired and supply restored after field inspection."),
    dict(id="CM-03", category="Drainage", subcategory="Blocked drain / sewage overflow", department="Sanitation",
         assignee="Sanitary Inspector", sla_days=4, sensitive=0, identity_hiding=0, auto_ok=0, severity=3,
         keywords=["drain", "drainage", "sewage", "sewer", "gutter", "नाला", "नाली", "गटार", "गटर", "सीवर", "सांडपाणी", "ड्रेनेज", "तुंबल"],
         resolution="Drain desilted and blockage cleared by the Sanitation team."),
    dict(id="CM-04", category="Solid Waste", subcategory="Garbage not collected", department="Solid Waste Management",
         assignee="Ward Sanitary Supervisor", sla_days=2, sensitive=0, identity_hiding=0, auto_ok=1, severity=2,
         keywords=["garbage", "waste", "trash", "dustbin", "rubbish", "कचरा", "कचऱ्या", "घंटागाडी", "कूड़ा", "कूड़े", "कचरे", "kachra", "kooda"],
         resolution="Collection vehicle rescheduled and the spot cleared."),
    dict(id="CM-05", category="Roads", subcategory="Pothole / damaged road", department="Public Works",
         assignee="Sub-Engineer (Roads)", sla_days=7, sensitive=0, identity_hiding=0, auto_ok=0, severity=3,
         keywords=["pothole", "potholes", "road damage", "damaged road", "broken road", "खड्डा", "खड्डे", "खड्ड्या", "गड्ढा", "गड्ढे", "सड़क टूट",
                   "रस्ता खराब", "रस्त्याची दुरवस्था", "gaddha", "gadde"],
         resolution="Pothole patched with hot-mix by the Public Works team."),
    dict(id="CM-06", category="Harassment", subcategory="Harassment / ragging / misconduct by staff", department="Vigilance & Internal Committee",
         assignee="Presiding Officer (Internal Committee)", sla_days=5, sensitive=1, identity_hiding=1, auto_ok=0, severity=5,
         keywords=["harass", "misbehav", "molest", "ragging", "threaten", "threat", "abus", "stalk", "inappropriate", "छळ", "छळवणूक", "गैरवर्तन",
                   "धमकी", "धमक", "रॅगिंग", "रैगिंग", "उत्पीड़न", "छेड़", "छेडछाड", "अश्लील", "बदतमीज़ी", "दुर्व्यवहार", "विनयभंग", "अभद्र"],
         resolution="Handled confidentially by the Internal Committee."),
    dict(id="CM-07", category="Corruption", subcategory="Bribe demand / corruption", department="Vigilance & Internal Committee",
         assignee="Vigilance Officer", sla_days=5, sensitive=1, identity_hiding=1, auto_ok=0, severity=5,
         keywords=["bribe", "corrupt", "kickback", "under the table", "speed money", "extra money", "chai pani", "chai-pani", "लाच", "भ्रष्टाचार",
                   "रिश्वत", "घूस", "चाय-पानी", "चायपानी", "चहापाणी", "चहा-पाणी", "पैसे मागित", "पैसे मागत", "पैसे मांग", "पैसा मांग", "rishwat", "ghoos"],
         resolution="Referred to the Vigilance Officer for confidential inquiry."),
    dict(id="CM-08", category="Certificates", subcategory="Certificate delay / status", department="Revenue & Records",
         assignee="Records Clerk", sla_days=7, sensitive=0, identity_hiding=0, auto_ok=1, severity=2,
         keywords=["certificate", "bonafide", "transcript", "दाखला", "दाखले", "प्रमाणपत्र", "सर्टिफिकेट"],
         resolution="Application located; certificate issued and status shared with the applicant."),
    dict(id="CM-09", category="Public Health", subcategory="Stray animals / mosquitoes", department="Health",
         assignee="Health Officer", sla_days=5, sensitive=0, identity_hiding=0, auto_ok=0, severity=3,
         keywords=["stray dog", "dog bite", "dogs", "mosquito", "dengue", "malaria", "fogging", "कुत्रे", "कुत्र्या", "कुत्ते", "कुत्तों", "डास", "मच्छर",
                   "डेंग्यू", "डेंगू", "फॉगिंग", "धुरळणी"],
         resolution="Fogging carried out / animal control team deployed."),
    dict(id="CM-10", category="Electrical Hazard", subcategory="Live wire / transformer hazard", department="Electrical",
         assignee="Assistant Engineer (Electrical)", sla_days=1, sensitive=0, identity_hiding=0, auto_ok=0, severity=5,
         keywords=["live wire", "electric shock", "hanging wire", "transformer", "sparking", "short circuit", "वीजेची तार", "विजेची तार", "तार तुटल",
                   "करंट", "शॉक", "ट्रान्सफॉर्मर", "ट्रांसफार्मर", "बिजली का तार", "तार लटक", "ठिणग्या", "चिंगारी"],
         resolution="Line isolated and hazard removed by the Electrical emergency crew."),
]

DEPARTMENTS = [
    dict(name="Electrical", jurisdiction="All wards", email="electrical@example-civic.in", whatsapp="+91-00000-00001", supervisor="Executive Engineer (Electrical)"),
    dict(name="Water Supply", jurisdiction="All wards", email="water@example-civic.in", whatsapp="+91-00000-00002", supervisor="Executive Engineer (Water)"),
    dict(name="Sanitation", jurisdiction="All wards", email="sanitation@example-civic.in", whatsapp="+91-00000-00003", supervisor="Chief Sanitary Officer"),
    dict(name="Solid Waste Management", jurisdiction="All wards", email="swm@example-civic.in", whatsapp="+91-00000-00004", supervisor="SWM Head"),
    dict(name="Public Works", jurisdiction="All wards", email="pwd@example-civic.in", whatsapp="+91-00000-00005", supervisor="Executive Engineer (PWD)"),
    dict(name="Vigilance & Internal Committee", jurisdiction="Confidential", email="vigilance@example-civic.in", whatsapp="+91-00000-00006", supervisor="Commissioner's Office"),
    dict(name="Revenue & Records", jurisdiction="All wards", email="records@example-civic.in", whatsapp="+91-00000-00007", supervisor="Deputy Registrar"),
    dict(name="Health", jurisdiction="All wards", email="health@example-civic.in", whatsapp="+91-00000-00008", supervisor="Medical Officer of Health"),
]

T = {
    "CM-01": dict(en=["The streetlight near {loc} main road has not been working for many days.", "Street light pole in {loc} is off at night, very dark."],
                  hi=["{loc} में स्ट्रीट लाइट कई दिनों से बंद है, रात को अंधेरा रहता है।", "{loc} की गली की बत्ती खराब है, कृपया ठीक करें।"],
                  mr=["{loc} मधील पथदिवा बरेच दिवस बंद आहे, रात्री अंधार असतो.", "{loc} येथील रस्त्यावरील दिवे लागत नाहीत, कृपया दुरुस्त करा."]),
    "CM-02": dict(en=["No water supply in {loc} since two days.", "There is a pipeline leakage in {loc}, water is being wasted."],
                  hi=["{loc} में दो दिन से पानी नहीं आ रहा है।", "{loc} में पाइपलाइन से पानी बह रहा है।"],
                  mr=["{loc} मध्ये दोन दिवसांपासून नळाला पाणी येत नाही.", "{loc} येथे जलवाहिनी फुटली आहे, पाणी वाया जात आहे."]),
    "CM-03": dict(en=["The drain in {loc} is blocked and sewage is overflowing on the road.", "Drainage choked near {loc}, bad smell everywhere."],
                  hi=["{loc} में नाली जाम है और गंदा पानी सड़क पर बह रहा है।", "{loc} का सीवर ओवरफ्लो हो रहा है।"],
                  mr=["{loc} मधील गटार तुंबले आहे, सांडपाणी रस्त्यावर येत आहे.", "{loc} येथील नाला साफ केलेला नाही, दुर्गंधी पसरली आहे."]),
    "CM-04": dict(en=["Garbage has not been collected in {loc} for a week.", "The dustbin at {loc} is overflowing, waste is on the road."],
                  hi=["{loc} में एक हफ्ते से कूड़ा नहीं उठाया गया है।", "{loc} में कचरे का ढेर लगा है।"],
                  mr=["{loc} मध्ये आठवडाभर कचरा उचललेला नाही.", "{loc} मध्ये घंटागाडी येत नाही, कचरा साचला आहे."]),
    "CM-05": dict(en=["Big potholes on the road in {loc}, accidents are happening.", "Road damaged near {loc} after rains."],
                  hi=["{loc} की सड़क पर बड़े गड्ढे हैं, दुर्घटनाएं हो रही हैं।", "{loc} में सड़क टूट गई है।"],
                  mr=["{loc} मधील रस्त्यावर मोठे खड्डे पडले आहेत, अपघात होत आहेत.", "{loc} येथील रस्ता खराब झाला आहे."]),
    "CM-06": dict(en=["A staff member at the {loc} office is harassing me and behaving inappropriately.", "Seniors are ragging juniors in the hostel near {loc}."],
                  hi=["{loc} कार्यालय का कर्मचारी मेरा उत्पीड़न कर रहा है।", "{loc} में एक अधिकारी ने मेरे साथ दुर्व्यवहार किया और धमकी दी।"],
                  mr=["{loc} कार्यालयातील कर्मचारी माझा छळ करत आहे.", "{loc} येथील अधिकाऱ्याने गैरवर्तन केले आणि धमकी दिली."]),
    "CM-07": dict(en=["The clerk at {loc} office is asking for a bribe to process my file.", "Corruption in {loc} ward office, they demand extra money."],
                  hi=["{loc} कार्यालय का बाबू फाइल के लिए रिश्वत मांग रहा है।", "{loc} में काम के लिए घूस मांगी जा रही है।"],
                  mr=["{loc} कार्यालयातील लिपिक फाईलसाठी लाच मागत आहे.", "{loc} प्रभाग कार्यालयात कामासाठी पैसे मागितले जातात, भ्रष्टाचार आहे."]),
    "CM-08": dict(en=["I applied for an income certificate at {loc} office a month ago, still not received.", "My bonafide certificate is delayed, applied from {loc}."],
                  hi=["{loc} कार्यालय में आय प्रमाणपत्र के लिए आवेदन किया था, अभी तक नहीं मिला।", "{loc} से जाति प्रमाणपत्र का आवेदन लंबित है।"],
                  mr=["{loc} कार्यालयात उत्पन्नाचा दाखला मागितला होता, अजून मिळाला नाही.", "{loc} येथून जातीचे प्रमाणपत्र अर्ज प्रलंबित आहे."]),
    "CM-09": dict(en=["Stray dogs are attacking people in {loc}.", "Too many mosquitoes in {loc}, dengue cases rising, need fogging."],
                  hi=["{loc} में आवारा कुत्ते लोगों को काट रहे हैं।", "{loc} में मच्छर बहुत बढ़ गए हैं, फॉगिंग करवाइए।"],
                  mr=["{loc} मध्ये भटके कुत्रे लोकांवर हल्ला करत आहेत.", "{loc} मध्ये डास वाढले आहेत, धुरळणी करावी."]),
    "CM-10": dict(en=["A live wire is hanging low near {loc}, very dangerous.", "Transformer in {loc} is sparking."],
                  hi=["{loc} में बिजली का तार लटक रहा है, करंट का खतरा है।", "{loc} के ट्रांसफार्मर से चिंगारी निकल रही है।"],
                  mr=["{loc} येथे विजेची तार तुटली आहे, शॉक बसण्याचा धोका आहे.", "{loc} मधील ट्रान्सफॉर्मरमधून ठिणग्या पडत आहेत."]),
}
WEIGHTS = {"CM-01": 14, "CM-02": 13, "CM-03": 12, "CM-04": 14, "CM-05": 11, "CM-06": 5, "CM-07": 5, "CM-08": 10, "CM-09": 9, "CM-10": 7}
SLOW = {"CM-03": 1.25, "CM-05": 1.2, "CM-02": 1.0, "CM-09": 0.9}  # how close mean resolution time runs to the SLA
LOC_SLOW = {"Taroda": 1.3, "Itwara": 1.15}


def history(n=420, now=None):
    rnd = random.Random(7)
    now = now or datetime.now()
    rows = []
    cats = {c["id"]: c for c in CATEGORIES}
    ids = list(WEIGHTS)
    for i in range(n):
        cid = rnd.choices(ids, [WEIGHTS[k] for k in ids])[0]
        lang = rnd.choices(["mr", "hi", "en"], [45, 30, 25])[0]
        loc = rnd.choice(LOCALITIES)
        i = rnd.randrange(len(T[cid][lang]))
        text = T[cid][lang][i].format(loc=loc[0] if lang == "en" else loc[1])
        text_en = T[cid]["en"][i].format(loc=loc[0])  # aligned English version (real data: translated once at ingestion)
        sub = now - timedelta(days=rnd.uniform(8, 150))
        c = cats[cid]
        days = max(0.3, rnd.gauss(c["sla_days"] * SLOW.get(cid, 0.7) * LOC_SLOW.get(loc[0], 1.0), c["sla_days"] * 0.35))
        rows.append(dict(text=text, text_en=text_en, lang=lang, category_id=cid, locality=loc[0], submitted=sub.isoformat(timespec="seconds"),
                         resolved=(sub + timedelta(days=days)).isoformat(timespec="seconds"), status="Resolved",
                         resolution=c["resolution"], feedback=rnd.choices(["Satisfied", "Not satisfied"], [78, 22])[0],
                         breached=int(days > c["sla_days"])))
    # a recurring incident so the recurrence detector has something real to find
    for d in (41, 27, 12):
        sub = now - timedelta(days=d)
        rows.append(dict(text="तरोडा मधील गटार तुंबले आहे, सांडपाणी रस्त्यावर येत आहे.", text_en="The drain in Taroda is blocked and sewage is overflowing on the road.", lang="mr", category_id="CM-03", locality="Taroda",
                         submitted=sub.isoformat(timespec="seconds"), resolved=(sub + timedelta(days=6)).isoformat(timespec="seconds"),
                         status="Resolved", resolution=cats["CM-03"]["resolution"], feedback="Not satisfied", breached=1))
    return rows
