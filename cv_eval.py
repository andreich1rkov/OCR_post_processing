"""Честная оценка: 5-fold кросс-валидация на train.
Словарь слов строится только по обучающим фолдам, проверяемые документы его не «видели».
Запуск: python cv_eval.py [seed ...]"""
import json, random, sys, collections
import postprocess as pp

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
PRED = load("data/train_predictions.jsonl")
LAB = {d["doc_id"]: d for d in load("data/train_labels.jsonl")}
DIRW = pp.load_subdivision_words(pp.find_subdivisions())

def truth_valid(g):
    f = g["fields"]
    return pp.passport_valid(pp.parse_date(f["Birth_date"]), pp.parse_date(f["Issue_date"]), pp.PARAMS["grace_days"])

def run(params=None, k=5, seed=0):
    params = params or pp.PARAMS
    ids = [d["doc_id"] for d in PRED]; random.Random(seed).shuffle(ids)
    folds = [set(ids[i::k]) for i in range(k)]
    res = {"acc": 0, "bad": 0, "bad_ids": [], "by_src": collections.Counter(), "bad_src": collections.Counter(),
           "reasons": collections.Counter()}
    for fo in folds:
        vocab = pp.build_vocab([LAB[i] for i in ids if i not in fo])
        mdl = pp.make_model(vocab, DIRW)
        for d in PRED:
            if d["doc_id"] not in fo: continue
            dec, fields, rs = pp.analyze(d, mdl, params)
            if dec != "accept":
                for r in set(rs): res["reasons"][r] += 1
                continue
            g = LAB[d["doc_id"]]
            wrong = [f for f in pp.FIELDS if pp.norm(fields[f]) != pp.norm(g["fields"][f])]
            res["acc"] += 1; res["by_src"][d["source"]] += 1
            for n_ in set(rs): res.setdefault("notes", collections.Counter())[n_] += 1
            if wrong or not truth_valid(g):
                res["bad"] += 1; res["bad_ids"].append((d["doc_id"], wrong)); res["bad_src"][d["source"]] += 1
    return res

if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [0, 1, 2]
    tot_a = tot_b = 0
    for s in seeds:
        r = run(seed=s)
        tot_a += r["acc"]; tot_b += r["bad"]
        print(f"seed {s}: принято {r['acc']} из {len(PRED)} ({r['acc']/len(PRED):.1%}), с ошибкой {r['bad']} ({r['bad']/max(r['acc'],1):.2%})")
        print("   по источникам: принято", dict(r["by_src"]), "ошибок", dict(r["bad_src"]))
    print(f"в среднем: принято {tot_a/len(seeds):.0f}, ошибок {tot_b/len(seeds):.1f}")