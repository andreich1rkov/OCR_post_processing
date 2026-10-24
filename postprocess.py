"""Постобработка OCR паспорта РФ (версия 2): принять документ и выдать 10 полей либо отклонить.

Запуск:
    python postprocess.py data/test_predictions.jsonl answers.jsonl
Сборка словаря из разметки train (один раз):
    python postprocess.py --build-model data/train_labels.jsonl model.json

Принцип: документ принимаем, только если каждое поле подтверждено независимо от самой OCR-строки.
  * ФИО — сверкой с первой строкой MRZ (две строки внизу страницы, та же информация в латинице);
    при расхождении в одной букве побеждает чтение с заметно большей уверенностью;
  * серия/номер, даты, код подразделения — по MRZ с контрольными цифрами (MRZ надёжнее, если цифры сошлись);
  * название органа и место рождения — словарём слов, уверенностью КАЖДОГО символа и геометрией рамки;
    ошибки в одну букву исправляем только если буква прочитана неуверенно и есть единственный кандидат;
  * действительность паспорта на 01.10.2026 — по дате рождения и дате выдачи.
Справочник subdivisions.csv — DaData, лицензия CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).
"""
import csv
import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
          "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",
          "Issue_organization_ru"]
AS_OF = date(2026, 10, 1)

PARAMS = {
    "min_conf": 0.95,       # мин. уверенность любого неисправленного символа (с пробелами) в органе и месте
    "low_conf": 0.90,       # символ с уверенностью ниже считается спорным и может быть исправлен по словарю
    "name_hi": 0.98,        # ФИО: уверенность «победителя» при расхождении в одной букве
    "name_lo": 0.90,        # ФИО: уверенность «проигравшего»
    "max_px_line": 38,      # макс. высота рамки на строку
    "grace_days": 0,        # 0 — паспорт недействителен со дня 20/45-летия; 90 — с учётом срока на замену
    "fix_names": True, "fix_numbers": True, "fix_words": True, "fix_dots": True,
    "empty_middle": True,   # пустое отчество допустимо, если MRZ тоже без отчества
    "near_edge": False,     # (устарело) лишняя буква в начале/конце
    "near_del": True,       # неизвестное слово места рождения, ставшее известным после удаления одной буквы
    "short_unknown": True,  # неизвестное слово места рождения из 1–3 символов
    "amb_guard": True,      # неизвестное слово = сокращение + слово («ГМОСКВЕ»)
    "glue_check": True,     # «ОБЛ.В» — склейка после сокращения, после которого так не пишут
    "end_model": True,      # поле оборвано: конец, после которого в разметке обычно что-то есть
    "start_model": False,   # то же для начала (на train ни одной ошибки не предотвращает)
}

TR = {"А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "2", "Ж": "J",
      "З": "Z", "И": "I", "Й": "Q", "К": "K", "Л": "L", "М": "M", "Н": "N", "О": "O",
      "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U", "Ф": "F", "Х": "H", "Ц": "C",
      "Ч": "3", "Ш": "4", "Щ": "W", "Ъ": "X", "Ы": "Y", "Ь": "9", "Э": "6", "Ю": "7",
      "Я": "8", "-": "<", " ": "<"}
INV = {v: k for k, v in TR.items() if k not in "- "}
WORD = re.compile(r"[А-ЯЁ0-9]+")


def norm(s):
    return " ".join(str(s or "").split()).upper()


def alpha_tokens(s):
    return WORD.findall(norm(s))


# ------------------------------------------------------------------ чтение OCR
def get_field(doc, name):
    """Поле визуальной зоны посимвольно: строки со списками символов, уверенностей и флагов «исправлено»."""
    v = (doc.get("viz") or {}).get(name)
    if not isinstance(v, dict):
        return None
    box = v.get("box")
    if not (isinstance(box, list) and len(box) == 4):
        return None
    lines = []
    for l in v.get("lines") or []:
        if not isinstance(l, dict):
            continue
        t = str(l.get("text") or "").upper()
        c = [float(x) for x in (l.get("char_conf") or [])]
        if not t:
            continue
        if len(c) != len(t):
            c = [min(c) if c else 0.0] * len(t)
        lines.append({"ch": list(t), "cf": c, "ex": [False] * len(t)})
    return {"lines": lines, "box": box} if lines else None


def ftext(f):
    return norm(" ".join("".join(l["ch"]) for l in f["lines"]))


def fmin(f):
    return min([c for l in f["lines"] for c, e in zip(l["cf"], l["ex"]) if not e] or [1.0])


def lines_too_tall(f, p):
    return f["box"][3] / len(f["lines"]) > p["max_px_line"]


# ------------------------------------------------------------------ даты / паспорт
def parse_date(s):
    try:
        return datetime.strptime(s, "%d.%m.%Y").date()
    except (ValueError, TypeError):
        return None


def add_years(d, y):
    try:
        return d.replace(year=d.year + y)
    except ValueError:
        return d.replace(year=d.year + y, day=28)


def mk_date(y, m, d):
    try:
        return date(y, m, d)
    except ValueError:
        return None


def resolve_issue(s):
    c = [x for x in (mk_date(1900 + int(s[:2]), int(s[2:4]), int(s[4:])),
                     mk_date(2000 + int(s[:2]), int(s[2:4]), int(s[4:])))
         if x and date(1997, 1, 1) <= x <= AS_OF]
    return c[0] if len(c) == 1 else None


def resolve_birth(s, issue):
    if not issue:
        return None
    c = []
    for cent in (1900, 2000):
        x = mk_date(cent + int(s[:2]), int(s[2:4]), int(s[4:]))
        if x and add_years(x, 14) <= issue and issue.year - x.year <= 110:
            c.append(x)
    return c[0] if len(c) == 1 else None


def passport_valid(birth, issue, grace_days=0):
    """С 14 до 20 лет паспорт действует до 20-летия, с 20 до 45 — до 45-летия, после 45 — бессрочно."""
    if not birth or not issue or issue > AS_OF or issue < birth:
        return False
    age = (issue.year - birth.year) - ((issue.month, issue.day) < (birth.month, birth.day))
    if age < 14:
        return False
    if age >= 45:
        return True
    limit = add_years(birth, 20 if age < 20 else 45)
    return AS_OF < limit + timedelta(days=grace_days)


# ------------------------------------------------------------------ MRZ
def cd(s):
    """Контрольная цифра ICAO (веса 7-3-1 по кругу)."""
    t = 0
    for i, ch in enumerate(s):
        if ch.isdigit():
            v = int(ch)
        elif ch == "<":
            v = 0
        elif "A" <= ch <= "Z":
            v = ord(ch) - 55
        else:
            return "?"
        t += v * (7, 3, 1)[i % 3]
    return str(t % 10)


def mrz_read(doc):
    """-> (строка1, строка2, уверенности1, уверенности2) без пробелов, либо None."""
    m = doc.get("mrz")
    ls = (m.get("lines") or []) if isinstance(m, dict) else []
    if len(ls) != 2 or not all(isinstance(l, dict) for l in ls):
        return None
    out = []
    for l in ls:
        t = str(l.get("text") or "").upper()
        c = list(l.get("char_conf") or [])
        if len(c) != len(t):
            c = [0.5] * len(t)
        pairs = [(a, b) for a, b in zip(t, c) if not a.isspace()]
        out.append(("".join(a for a, _ in pairs), [float(b) for _, b in pairs]))
    return out[0][0], out[1][0], out[0][1], out[1][1]


def name_area(last, first, middle):
    tr = lambda s: "".join(TR.get(ch, "?") for ch in norm(s))
    return tr(last) + "<<" + tr(first) + "<" + tr(middle)


def fuse_names(F, l1, c1, p, notes):
    """ФИО против первой строки MRZ. -> (три имени | None, причины)."""
    i = l1.find("RUS")
    if not (0 <= i <= 3):
        return None, ["MRZ: нет префикса страны"]
    area = l1[i + 3:]
    carea = c1[i + 3:]
    if F["Last_name_ru"] is None or F["First_name_ru"] is None:
        return None, ["фамилия или имя не найдены"]
    last, first = ftext(F["Last_name_ru"]), ftext(F["First_name_ru"])
    if F["Middle_name_ru"] is None:
        if not p["empty_middle"]:
            return None, ["отчество не найдено"]
        middle = ""
        mconf = []
    else:
        middle = ftext(F["Middle_name_ru"])
        mconf = None
    names = [last, first, middle]
    # уверенности по символам визуальной зоны (в порядке ФИО)
    def confs_of(fld, text):
        cs = [c for l in fld["lines"] for c in l["cf"]]
        return cs if len(cs) == len(text) else [min(cs)] * len(text)
    vc = confs_of(F["Last_name_ru"], last) + [None, None] + confs_of(F["First_name_ru"], first) + [None]
    vc += confs_of(F["Middle_name_ru"], middle) if middle else []
    E = name_area(*names)
    if "?" in E:
        return None, ["ФИО с неизвестными буквами"]
    if len(E) > 39 and any(c is not None and c < p["name_hi"] for c in vc[39:]):
        return None, ["ФИО: часть за пределами MRZ прочитана неуверенно"]   # в MRZ влезает только 39 символов
    Ep = (E + "<" * 39)[:39]
    P = (area + "<" * 39)[:39] if len(area.rstrip("<")) <= 39 else None
    if P is None:
        return None, ["MRZ: строка ФИО слишком длинная"]
    cp = (carea + [0.0] * 39)[:39]
    bad = [j for j in range(39) if Ep[j] != P[j]]
    if not bad:
        return names, []
    if not p["fix_names"] or len(bad) > 2:
        return None, ["ФИО не совпало с MRZ"]
    chars = list(E)
    for j in bad:
        if j >= len(E) or vc[j] is None or Ep[j] == "<" or P[j] == "<" or P[j] not in INV:
            return None, ["ФИО не совпало с MRZ"]
        cv, cm = vc[j], cp[j]
        if cm >= p["name_hi"] and cv <= p["name_lo"]:
            chars[j] = P[j]                      # MRZ уверенно, визуальная зона нет -> берём MRZ
            notes.append("ФИО исправлено по MRZ")
        elif cv >= p["name_hi"] and cm <= p["name_lo"]:
            notes.append("ФИО: MRZ прочитана хуже")  # MRZ ошиблась -> остаётся визуальная зона
        else:
            return None, ["ФИО: расхождение с MRZ нельзя разрешить по уверенности"]
    # собрать имена обратно из исправленной строки
    s = "".join(chars)
    a = s.find("<<")
    last2 = s[:a]
    rest = s[a + 2:]
    b = len(names[1])
    first2, middle2 = rest[:b], rest[b + 1:b + 1 + len(names[2])]
    dec = lambda t: "".join(INV.get(ch, ch) if ch not in "<" else "-" for ch in t)
    def rebuild(orig, new):
        return "".join(INV[n] if (n != o and n in INV and TR.get(o) != n) else o for o, n in zip(orig, new))
    return [rebuild(names[0], last2), rebuild(names[1], first2), rebuild(names[2], middle2)], []


def fuse_numbers(F, l2, c2, out, p, notes):
    """Серия/номер, даты, код, пол: из MRZ (по контрольным цифрам) и визуальной зоны."""
    R = []
    sex = ftext(F["Sex_ru"]) if F["Sex_ru"] else ""
    if sex not in ("МУЖ.", "ЖЕН."):
        return ["пол не распознан"]
    vz = {k: (ftext(F[k]) if F[k] else None) for k in ("Birth_date", "Issue_date", "Issue_organisation_code", "Licence_number")}
    vdg = re.sub(r"\D", "", vz["Licence_number"] or "")
    v_num_ok = bool(vz["Licence_number"]) and re.fullmatch(r"[\d ]+", vz["Licence_number"]) and len(vdg) == 10
    v_b = vz["Birth_date"] if vz["Birth_date"] and re.fullmatch(r"\d\d\.\d\d\.\d{4}", vz["Birth_date"]) else None
    v_i = vz["Issue_date"] if vz["Issue_date"] and re.fullmatch(r"\d\d\.\d\d\.\d{4}", vz["Issue_date"]) else None
    v_c = vz["Issue_organisation_code"] if vz["Issue_organisation_code"] and re.fullmatch(r"\d{3}-\d{3}", vz["Issue_organisation_code"]) else None
    # сегменты MRZ, которые можно прочитать по позициям (строка ровно 44 символа)
    segA = segB = segC = None
    if len(l2) == 44:
        if l2[0:9].isdigit() and cd(l2[0:9]) == l2[9]:
            segA = (l2[0:2], l2[2], l2[3:9])
        if l2[13:19].isdigit() and cd(l2[13:19]) == l2[19]:
            segB = l2[13:19]
        if l2[28:35].isdigit() and l2[35:41].isdigit() and cd(l2[28:41]) == l2[42]:
            segC = (l2[28], l2[29:35], l2[35:41])
    # где позиции недоступны — ищем куски, ожидаемые по визуальной зоне (подтверждение совпадением)
    def found(piece):
        return l2.find(piece) if piece else -1
    ia = ib = ic = -1
    if segA is None and v_num_ok:
        doc9 = vdg[0:2] + vdg[2] + vdg[4:10]
        ia = found(doc9 + cd(doc9))
        if ia >= 0:
            segA = (vdg[0:2], vdg[2], vdg[4:10])
    if segC is None and v_i and v_c and v_num_ok:
        C = vdg[3] + v_i[8:10] + v_i[3:5] + v_i[0:2] + v_c.replace("-", "")
        ic = found(C)
        if ic >= 0 and l2[ic + 14:ic + 15] == cd(C):
            segC = (C[0], C[1:7], C[7:])
    if segB is None and v_b:
        B = v_b[8:10] + v_b[3:5] + v_b[0:2]
        ib = found(B + cd(B))
        if ib >= 0:
            segB = B
    # пол: символ MRZ сразу после даты рождения
    ib_pos = 13 if len(l2) == 44 else (found(segB + cd(segB)) if segB else -1)
    if ib_pos < 0 or l2[ib_pos + 7:ib_pos + 8] != ("M" if sex == "МУЖ." else "F"):
        R.append("пол не подтверждён MRZ")
    use_mrz = p["fix_numbers"]
    issue = bd = None
    if segC:
        issue = resolve_issue(segC[1])
        if issue is None and v_i:
            R.append("дата выдачи нереалистична")
    if segB and issue:
        bd = resolve_birth(segB, issue)
        if bd is None:
            R.append("дата рождения нереалистична")
    if not (segA and segC and segB and issue and bd):
        R.append("серия/номер/даты/код не подтверждены MRZ")
        return R
    num = segA[0] + segA[1] + segC[0] + segA[2]
    code = segC[2][:3] + "-" + segC[2][3:]
    new = {"Licence_number": f"{num[:2]} {num[2:4]} {num[4:]}", "Issue_organisation_code": code,
           "Issue_date": issue.strftime("%d.%m.%Y"), "Birth_date": bd.strftime("%d.%m.%Y")}
    vv = {"Licence_number": f"{vdg[:2]} {vdg[2:4]} {vdg[4:]}" if v_num_ok else None, "Issue_organisation_code": v_c,
          "Issue_date": v_i, "Birth_date": v_b}
    for k, v in new.items():
        if vv[k] != v:
            if not use_mrz:
                R.append(f"{k} не совпало с MRZ")
            else:
                notes.append(f"{k} взято из MRZ")
        out[k] = v
    out["Sex_ru"] = sex
    return R


# ------------------------------------------------------------------ словари и исправление слов
def load_subdivision_words(path):
    words = set()
    try:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                words.update(alpha_tokens(r.get("name")))
    except (OSError, TypeError):
        pass
    return words


def build_vocab(label_rows):
    """Статистика по разметке train (в кросс-валидации — только по обучающим фолдам)."""
    from collections import Counter
    org, bp = set(), set()
    cnt = Counter()                           # токены места рождения как есть (с точками)
    dotted = Counter()                        # сокращения с точкой (органы и места)
    glue = {"org": Counter(), "bp": Counter()}    # «Г.МОСКВА» (склеено) / «Г. МОСКВА» (с пробелом)
    spaced = {"org": Counter(), "bp": Counter()}
    ends = {"org": Counter(), "bp": Counter()}
    starts = {"org": Counter(), "bp": Counter()}
    pairs = {"org": Counter(), "bp": Counter()}
    for g in label_rows:
        f = g.get("fields") or {}
        for kind, key, words in (("org", "Issue_organization_ru", org), ("bp", "Birth_place_ru", bp)):
            t = norm(f.get(key))
            words.update(alpha_tokens(t))
            toks = t.split()
            if kind == "bp":
                cnt.update(toks)
            for tok in toks:
                if tok.endswith(".") and tok[:-1].isalpha():
                    dotted[tok[:-1]] += 1
            for m in re.finditer(r"\b([А-ЯЁ]{1,4})\.(\s|(?=[А-ЯЁ]))", t):
                (spaced if m.group(2) else glue)[kind][m.group(1)] += 1
            for x, y in zip(toks, toks[1:]):
                pairs[kind][x + " " + y] += 1
            if len(toks) >= 2:
                ends[kind][" ".join(toks[-2:])] += 1
                starts[kind][" ".join(toks[:2])] += 1
    bp_dotted = sorted(t[:-1] for t, n in cnt.items() if t.endswith(".") and n >= 20 and t[:-1].isalpha() and cnt[t[:-1]] == 0)
    no_glue = {k: sorted(a for a in set(glue[k]) | set(spaced[k])
                         if glue[k][a] + spaced[k][a] >= 20 and glue[k][a] / (glue[k][a] + spaced[k][a]) < 0.05) for k in glue}
    bad_end = {k: sorted(p_ for p_, n in pairs[k].items() if n >= 3 and ends[k][p_] / n < 0.2) for k in pairs}
    bad_start = {k: sorted(p_ for p_, n in pairs[k].items() if n >= 3 and starts[k][p_] / n < 0.2) for k in pairs}
    return {"org_words": sorted(org), "bp_words": sorted(bp), "bp_dotted": bp_dotted,
            "dot_abbr": sorted(a for a, n in dotted.items() if n >= 5), "no_glue": no_glue,
            "bad_end": bad_end, "bad_start": bad_start}


def _indexes(words):
    sub, dele = {}, {}
    for w in words:
        if len(w) < 4:
            continue
        for i in range(len(w)):
            sub.setdefault(w[:i] + "*" + w[i + 1:], set()).add(w)
            dele.setdefault(w[:i] + w[i + 1:], set()).add(w)
    return {"words": words, "sub": sub, "del": dele}


def make_model(vocab, dir_words):
    org = set(vocab["org_words"]) | set(dir_words)
    bp = set(vocab["bp_words"]) | set(dir_words)
    dotted = set(vocab["bp_dotted"])
    abbr = set(vocab.get("dot_abbr", []))
    mdl = {"org": _indexes(org), "bp": _indexes(bp), "bp_dotted": dotted, "abbr": abbr,
           "no_glue": {k: set(v) for k, v in vocab.get("no_glue", {}).items()},
           "bad_end": {k: set(v) for k, v in vocab.get("bad_end", {}).items()},
           "bad_start": {k: set(v) for k, v in vocab.get("bad_start", {}).items()}}
    mdl["org"]["amb"] = abbr                      # слова, к которым легко «приклеивается» потерянная точка
    mdl["bp"]["amb"] = abbr - dotted
    return mdl


def find_subdivisions():
    for q in (HERE / "reference" / "subdivisions.csv", Path("reference/subdivisions.csv"),
              HERE / "subdivisions.csv", Path("subdivisions.csv")):
        if q.exists():
            return q
    return None


def load_model():
    dirw = load_subdivision_words(find_subdivisions())
    vocab = {"org_words": [], "bp_words": [], "bp_dotted": [], "dot_abbr": [], "no_glue": {}, "bad_end": {}, "bad_start": {}}
    mj = HERE / "model.json"
    if mj.exists():
        vocab = json.loads(mj.read_text(encoding="utf-8"))
    else:
        for cand in (HERE / "data" / "train_labels.jsonl", Path("data/train_labels.jsonl")):
            if cand.exists():
                vocab = build_vocab(read_jsonl(cand))
                break
    return make_model(vocab, dirw)


def correct_token(tok, cf, idx, low):
    """Единственный кандидат из словаря, отличающийся одной буквой там, где символ прочитан неуверенно."""
    n = len(tok)
    if n < 4:
        return None
    W = idx["words"]
    cands = {}
    for i in range(n):
        if cf[i] < low:
            for w in idx["sub"].get(tok[:i] + "*" + tok[i + 1:], ()):
                if w != tok:
                    cands[w] = ("sub", i)
            w = tok[:i] + tok[i + 1:]
            if w in W:
                cands[w] = ("del", i)
    for w in idx["del"].get(tok, ()):
        for k in range(len(w)):
            if w[:k] + w[k + 1:] == tok:
                nb = [cf[j] for j in (k - 1, k) if 0 <= j < n]
                if nb and min(nb) < low:
                    cands[w] = ("ins", k)
                break
    if len(cands) != 1:
        return None
    (w, (kind, i)), = cands.items()
    # исправление последней буквы токена в сокращение (ОБЛЮ -> ОБЛ) неоднозначно: на деле там могла быть точка
    if w in idx["amb"] and kind in ("del", "sub") and i == n - 1:
        return None
    return w, kind, i


def ambiguous_split(tok, idx):
    """Неизвестное слово = сокращение + известное слово («ГМОСКВЕ»): возможно, потеряна точка или пробел."""
    return any(tok[:k] in idx["amb"] and tok[k:] in idx["words"] and len(tok[k:]) >= 3 for k in range(1, min(5, len(tok))))


def fix_words(fld, idx, low, notes):
    """Исправляет неизвестные слова (справа налево, чтобы не сбивать позиции)."""
    for l in fld["lines"]:
        s = "".join(l["ch"])
        for m in reversed(list(WORD.finditer(s))):
            tok = m.group()
            if tok in idx["words"]:
                continue
            if ambiguous_split(tok, idx):
                continue
            r = correct_token(tok, l["cf"][m.start():m.end()], idx, low)
            if not r:
                continue
            w, kind, i = r
            a = m.start()
            if kind == "sub":
                l["ch"][a + i] = w[i]; l["cf"][a + i] = 1.0; l["ex"][a + i] = True
            elif kind == "del":
                del l["ch"][a + i]; del l["cf"][a + i]; del l["ex"][a + i]
            else:
                l["ch"].insert(a + i, w[i]); l["cf"].insert(a + i, 1.0); l["ex"].insert(a + i, True)
            notes.append("слово исправлено по словарю")


def fix_dots(fld, dotted, notes):
    """В метках места рождения сокращения «С», «Д», «ОБЛ» … всегда с точкой: потерянную точку возвращаем."""
    for l in fld["lines"]:
        s = "".join(l["ch"])
        for m in reversed(list(re.finditer(r"\S+", s))):
            if m.group() in dotted:
                e = m.end()
                for j in range(m.start(), min(e + 1, len(l["ch"]))):
                    l["ex"][j] = True
                l["ch"].insert(e, "."); l["cf"].insert(e, 1.0); l["ex"].insert(e, True)
                notes.append("потерянная точка возвращена")


def style_checks(fld, kind, mdl, p):
    """Проверки «стиля» записи по статистике разметки: склейка после сокращений, обрыв поля."""
    R = []
    text = ftext(fld)
    if p["glue_check"]:
        for m in re.finditer(r"\b([А-ЯЁ]{1,4})\.(?=[А-ЯЁ])", text):
            if m.group(1) in mdl["no_glue"].get(kind, ()):
                R.append("склейка после сокращения"); break
    if p["end_model"]:
        toks = text.split()
        if len(toks) >= 2:
            if " ".join(toks[-2:]) in mdl["bad_end"].get(kind, ()):
                R.append("поле, похоже, оборвано в конце")
            if p["start_model"] and " ".join(toks[:2]) in mdl["bad_start"].get(kind, ()):
                R.append("поле, похоже, оборвано в начале")
    if kind == "org" and p["amb_guard"]:
        for l in fld["lines"]:
            if any(t not in mdl["org"]["words"] and ambiguous_split(t, mdl["org"]) for t in WORD.findall("".join(l["ch"]))):
                R.append("орган: возможна потерянная точка/пробел"); break
    return R


# ------------------------------------------------------------------ решение по документу
def analyze(doc, mdl, p=PARAMS):
    """-> (decision, fields | None, причины отказа либо заметки об исправлениях)."""
    F = {f: get_field(doc, f) for f in FIELDS}
    if F["Issue_organization_ru"] is None or F["Birth_place_ru"] is None:
        return "reject", None, ["орган или место рождения не найдены"]  # поля не найдены — исправлять нечего
    R, notes, out = [], [], {}
    mz = mrz_read(doc)
    if mz is None:
        return "reject", None, ["MRZ не найдена или нет второй строки"]
    l1, l2, c1, c2 = mz
    names, r = fuse_names(F, l1, c1, p, notes)
    R += r
    if names:
        out["Last_name_ru"], out["First_name_ru"], out["Middle_name_ru"] = names
    R += fuse_numbers(F, l2, c2, out, p, notes)

    # название органа
    org = F["Issue_organization_ru"]
    if p["fix_words"]:
        fix_words(org, mdl["org"], p["low_conf"], notes)
    if any(t not in mdl["org"]["words"] for l in org["lines"] for t in WORD.findall("".join(l["ch"]))):
        R.append("орган: слово не из словаря")
    R += style_checks(org, "org", mdl, p)
    if fmin(org) < p["min_conf"]:
        R.append("орган: низкая уверенность")
    if lines_too_tall(org, p):
        R.append("орган: потеряна строка")
    out["Issue_organization_ru"] = ftext(org)

    # место рождения
    bp = F["Birth_place_ru"]
    if p["fix_dots"]:
        fix_dots(bp, mdl["bp_dotted"], notes)
    if p["fix_words"]:
        fix_words(bp, mdl["bp"], p["low_conf"], notes)
    if p["fix_dots"]:
        fix_dots(bp, mdl["bp_dotted"], notes)
    R += style_checks(bp, "bp", mdl, p)
    if fmin(bp) < p["min_conf"]:
        R.append("место: низкая уверенность")
    for l in bp["lines"]:
        for t in WORD.findall("".join(l["ch"])):
            if t in mdl["bp"]["words"]:
                continue
            W = mdl["bp"]["words"]
            if p["short_unknown"] and len(t) <= 3 and not t.isdigit():
                R.append("место: неизвестное короткое слово")
            if p["near_del"] and len(t) >= 5 and any(t[:i] + t[i + 1:] in W for i in range(len(t))):
                R.append("место: слово = известное + лишняя буква")
            if p["amb_guard"] and ambiguous_split(t, mdl["bp"]):
                R.append("место: возможна потерянная точка/пробел")
    if lines_too_tall(bp, p):
        R.append("место: потеряна строка")
    out["Birth_place_ru"] = ftext(bp)

    if not passport_valid(parse_date(out.get("Birth_date")), parse_date(out.get("Issue_date")), p["grace_days"]):
        R.append("паспорт недействителен на 01.10.2026")
    best = {f: out.get(f) or (ftext(F[f]) if F[f] else "") for f in FIELDS}
    if R:
        return "reject", best, R          # значения нужны только для диагностики (кросс-валидация)
    return "accept", best, notes


# ------------------------------------------------------------------ ввод-вывод
def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}, строка {n}: {e}")
    return rows


def main(argv):
    if len(argv) == 4 and argv[1] == "--build-model":
        v = build_vocab(read_jsonl(argv[2]))
        Path(argv[3]).write_text(json.dumps(v, ensure_ascii=False), encoding="utf-8")
        print(f"словарь записан в {argv[3]}: слов органа {len(v['org_words'])}, места {len(v['bp_words'])}, сокращений с точкой {len(v['bp_dotted'])}")
        return 0
    if len(argv) != 3:
        print("Использование: python postprocess.py data/test_predictions.jsonl answers.jsonl")
        return 2
    mdl = load_model()
    docs = read_jsonl(argv[1])
    acc = 0
    with open(argv[2], "w", encoding="utf-8") as out:
        for doc in docs:
            try:
                dec, fields, _ = analyze(doc, mdl)
            except Exception:
                dec, fields = "reject", None
            ans = {"doc_id": doc.get("doc_id"), "decision": dec}
            if dec == "accept":
                ans["fields"] = fields
                acc += 1
            out.write(json.dumps(ans, ensure_ascii=False) + "\n")
    print(f"Всего документов: {len(docs)}\nПринято: {acc}\nОтклонено: {len(docs) - acc}\nЗаписано: {argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))