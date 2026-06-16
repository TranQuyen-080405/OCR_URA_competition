"""Brand regex rules and extract_product (notebook Cell 3)."""

from __future__ import annotations

import re

from src.run_logger import RunLogger

BRAND_RULES = [
    (r"ha\s*long\s*canfoco.*pate.*c[ộo]t|c[ộo]t\s*đ[èe]n.*ha\s*long\s*canfoco", "Ha Long Canfoco Pate Cột Đèn", []),
    (r"ha\s*long\s*canfoco.*pate|canfoco.*pate\s*c[ộo]t|pate\s*c[ộo]t\s*đ[èe]n.*canfoco", "Ha Long Canfoco Pate", []),
    (r"ha\s*long\s*canfoco|halong\s*canfoco|canfood|canfoco", "Ha Long Canfoco", []),
    (r"đ[ồo]\s*h[ộo]p\s*h[ạa]\s*long|do\s*hop\s*ha\s*long", "Đồ Hộp Hạ Long", []),
    (r"pate\s*c[ộo]t\s*đ[èe]n|pate\s*cot\s*den|c[ộo]t\s*đ[èe]n\s*h[ảa]i\s*ph[òo]ng", "Pate Cột Đèn Hải Phòng", []),
    (r"h[ạa]\s*long\s*pate|pate\s*h[ạa]\s*long", "Ha Long Canfoco Pate", []),
    (r"vinamilk", "Vinamilk", ["flex", "adm gold", "sure", "canxi",
                                 "colosbaby", "colos baby", "ong tho", "ông thọ", "dielac", "grow"]),
    (r"th true|thtrue", "TH True Milk", ["true yogurt", "grow", "school milk", "true butter"]),
    (r"dutch lady|cô gái", "Dutch Lady", ["grow", "complete", "canxi", "mom"]),
    (r"nutifood|nuti\b", "Nutifood", ["growplus", "grow plus", "pedia", "iq"]),
    (r"ensure\b", "Abbott Ensure", ["gold", "original", "plus"]),
    (r"pediasure", "Abbott PediaSure", []),
    (r"similac", "Abbott Similac", []),
    (r"glucerna", "Abbott Glucerna", []),
    (r"milo\b", "Nestlé Milo", []),
    (r"nestle|nestlé", "Nestlé", ["milo", "coffee mate", "carnation", "nestea", "nan", "sữa bột"]),
    (r"aptamil", "Aptamil", []),
    (r"friso\b", "Friso", ["gold", "comfort", "prestige"]),
    (r"meiji\b", "Meiji", ["growing", "step"]),
    (r"ba vi\b|bavi\b|ba vì", "Ba Vì", ["gold"]),
    (r"lothamilk", "Lothamilk", ["canxi"]),
    (r"yomost", "Yomost", []),
    (r"dalat milk|đà lạt", "Đà Lạt Milk", []),
    (r"kun\b|kun milk", "Kun", ["chocolate", "strawberry"]),
    (r"fami\b", "Fami", ["canxi", "kid"]),
    (r"anlene", "Anlene", ["gold", "concentrate"]),
    (r"anchor\b", "Anchor", ["butter", "cream"]),
    (r"vissan", "Vissan", ["pate heo", "pate ga", "pate gà",
                           "xuc xich", "xúc xích", "lap xuong", "lạp xưởng"]),
    (r"hafi\b", "Hafi", ["pate"]),
    (r"ba huan|ba huân", "Ba Huân", ["pate"]),
    (r"san ha\b|san hà", "San Hà", ["pate"]),
    (r"\bcp\b|c\.p\.", "CP", ["pate", "xúc xích"]),
    (r"long bien|long biên", "Long Biên", ["pate"]),
    (r"\bpate\b|patê", "Pate", []),
]

RULE_TESTS = [
    ("HA LONG CANFOCO Pate Cột Đèn tạm dừng sản xuất", "Ha Long Canfoco Pate Cột Đèn"),
    ("HALONG CANFOCO pate cot den Hải Phòng", "Ha Long Canfoco Pate Cột Đèn"),
    ("ĐỒ HỘP HẠ LONG ISO 22000", "Đồ Hộp Hạ Long"),
    ("Pate cột đèn ai khóc nỗi đau này", "Pate Cột Đèn Hải Phòng"),
    ("Sữa tươi Vinamilk Flex 180ml giảm 20%", "Vinamilk Flex"),
    ("MILO Nestle chocolate 3 in 1", "Nestlé Milo"),
    ("Pate Heo Vissan 170g combo 3 hộp", "Vissan Pate Heo"),
    ("TH True Milk tươi tiệt trùng ít béo", "TH True Milk"),
    ("Dutch Lady Grow+ 900g", "Dutch Lady Grow"),
    ("No brand in this text", ""),
]


def extract_product(text: str) -> str:
    """Return 'Brand ProductLine', 'Brand', or '' if no match."""
    if not text or not text.strip():
        return ""
    tl = text.lower().replace("patê", "pate")
    for pattern, brand, lines in BRAND_RULES:
        if re.search(pattern, tl, re.IGNORECASE):
            for line in lines:
                if re.search(line, tl, re.IGNORECASE):
                    return f"{brand} {line.replace('+', '+').title()}"
            return brand
    return ""


def run_brand_self_tests(logger: RunLogger) -> bool:
    logger.section("Cell 3 — Brand Rules (self-tests)")

    all_pass = True
    for text, expected in RULE_TESTS:
        got = extract_product(text)
        ok = got == expected
        all_pass = all_pass and ok
        status = "PASS" if ok else "FAIL"
        logger.log(f"{status}: '{text[:45]}' -> got='{got}' | expected='{expected}'")

    logger.log("")
    logger.log("All self-tests passed." if all_pass else "Some self-tests failed — check BRAND_RULES.")
    logger.log(f"Total rules loaded: {len(BRAND_RULES)}")
    return all_pass
