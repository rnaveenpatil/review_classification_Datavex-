"""Fine-tune a Laya decision model (a "System-1" model) on your own labelled data.

This is the recipe behind TextCortex/raya, generalised to any choice/score task:

* soft targets: each label's share of the annotator votes (a 50/50 target where two disagree);
* every question phrasing in the task is trained on, and choice options are shuffled, so the
  model learns the content rather than the wording or the option order;
* soft-target cross-entropy with 1/sqrt(frequency) class weights, so rare labels still count;
* the best epoch is picked on validation, then a temperature per question is fitted there so
  the output probabilities are calibrated;
* the result is a normal Laya checkpoint: ``laya.Agent("<out>")`` loads it.

Quick start (CPU/Apple Silicon works for small data; a GPU is much faster):

    python train.py --task task.example.json --data data/example.jsonl --out my-router

Start from Raya instead of stock Laya to adapt the router to your own traffic:

    python train.py --task task.example.json --data my_data.jsonl --base TextCortex/raya --out my-router
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

import laya
from laya.common import QTYPES, build_sequence, collate_items, temp_bucket

from common import load_rows, load_task, row_state

BASE_FILES = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]


def resolve_base(base: str, subfolder: str | None) -> Path:
    """Local directory holding the base checkpoint (downloads it from the Hub if needed)."""
    if Path(base).is_dir():
        return Path(base) / subfolder if subfolder else Path(base)
    from huggingface_hub import snapshot_download

    patterns = [f"{subfolder}/{p}" for p in BASE_FILES] if subfolder else BASE_FILES
    local = Path(snapshot_download(base, allow_patterns=patterns))
    return local / subfolder if subfolder else local


def internal(question: dict) -> dict:
    return laya.Agent._to_internal(question)


def make_item(tok, cfg, state, qi: int, question: dict, labels: list[str], target: list[float], shuffle: bool):
    """Tokenise one (example, question) pair and align its target with the option order."""
    q = internal(question)
    if q["t"] == "choice":
        keys = list(q["crit"].keys())
        by_label = dict(zip(labels, target))
        order = list(range(len(keys)))
        if shuffle:
            random.shuffle(order)
        seq, markers = build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"], option_order=order)
        tgt = [by_label[keys[j]] for j in order]
    else:  # ordinal score: the option order is the meaning, never shuffle
        seq, markers = build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"])
        tgt = list(target)
    return {"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]], "target": tgt,
            "question": qi, "label_target": list(target)}


def length_batches(items, bs: int, shuffle: bool):
    """Batches of similar length (less padding), in random order when training."""
    idx = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    chunks = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    if shuffle:
        random.shuffle(chunks)
    return chunks


def forward(model, batch, dev):
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        logits, _ = model(batch["input_ids"].to(dev), batch["attention_mask"].to(dev), batch["marker_pos"].to(dev),
                          batch["marker_mask"].to(dev), batch["qtype"].to(dev))
    return logits.float()


@torch.no_grad()
def predict_logits(model, items, pad_id, dev, bs=16):
    model.eval()
    out = [None] * len(items)
    for chunk in length_batches(items, bs, shuffle=False):
        batch = collate_items([[items[i] for i in chunk]], pad_id)
        logits = forward(model, batch, dev).cpu()
        for j, i in enumerate(chunk):
            out[i] = logits[j, :len(items[i]["markers"])].numpy()
    return out


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = z / t
    p = np.exp(z - z.max())
    return p / p.sum()


def evaluate(logits, items, rows_of_items, n_questions: int, n_labels: int, temps=None):
    """Accuracy and macro-F1 per question, on examples with a single gold label."""
    res = {}
    for qi in range(n_questions):
        preds, golds = [], []
        for lg, it, row in zip(logits, items, rows_of_items):
            if it["question"] != qi or row["gold"] is None:
                continue
            preds.append(int(softmax(lg, (temps or {}).get(qi, 1.0)).argmax()))
            golds.append(int(np.argmax(it["target"])))
        if not golds:
            continue
        f1s = []
        for k in range(n_labels):
            tp = sum(p == k == g for p, g in zip(preds, golds))
            fp = sum(p == k != g for p, g in zip(preds, golds))
            fn = sum(g == k != p for p, g in zip(preds, golds))
            f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        res[qi] = {"acc": round(float(np.mean([p == g for p, g in zip(preds, golds)])), 4),
                   "macro_f1": round(float(np.mean(f1s)), 4), "n": len(golds)}
    return res


def soft_nll(logits, items) -> float:
    total = 0.0
    for lg, it in zip(logits, items):
        total -= float((np.array(it["target"]) * np.log(softmax(lg) + 1e-12)).sum())
    return total / len(items)


def fit_temperature(logits, items, qi: int) -> float:
    """Temperature minimising the soft-target NLL of one question on validation."""
    sel = [(lg, np.array(it["target"])) for lg, it in zip(logits, items) if it["question"] == qi]
    best_t, best_nll = 1.0, float("inf")
    for t in np.arange(0.5, 5.01, 0.05):
        nll = -sum(float((tg * np.log(softmax(lg, t) + 1e-12)).sum()) for lg, tg in sel)
        if nll < best_nll:
            best_t, best_nll = round(float(t), 2), nll
    return best_t


def split_rows(rows, val_frac: float, seed: int):
    """Rows marked ``"split": "val"`` are validation; otherwise a random ``val_frac`` share is."""
    marked = [r for r in rows if r.get("split") == "val"]
    if marked:
        return [r for r in rows if r.get("split") != "val"], marked
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n_val = max(1, int(len(rows) * val_frac))
    return rows[n_val:], rows[:n_val]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="task.json: labels + question phrasings")
    ap.add_argument("--data", required=True, help="training JSONL (see common.py for the format)")
    ap.add_argument("--val-data", help="optional separate validation JSONL")
    ap.add_argument("--out", required=True, help="output checkpoint directory")
    ap.add_argument("--base", default="convaiinnovations/laya",
                    help="base checkpoint: a Hub repo id or local dir (e.g. TextCortex/raya to adapt Raya)")
    ap.add_argument("--subfolder", default=None,
                    help="checkpoint subfolder; defaults to 'multilingual' for convaiinnovations/laya "
                         "(pass --subfolder . for Laya's English ModernBERT-large checkpoint)")
    ap.add_argument("--encoder", default=None,
                    help="build a fresh decision head on this HF encoder (e.g. jhu-clsp/mmBERT-small) instead of --base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=None, help="default 1e-4, or 3e-4 for a fresh head")
    ap.add_argument("--max-tokens", type=int, default=512, help="input token budget while training")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--select", choices=["acc", "nll"], default="acc", help="best-epoch criterion on validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    ap.add_argument("--train-embeddings", action="store_true",
                    help="also train the token-embedding table (frozen by default: saves memory, rarely helps)")
    ap.add_argument("--push-to-hub", metavar="REPO_ID", help="upload the result to this Hugging Face model repo")
    ap.add_argument("--private", action="store_true", help="with --push-to-hub: create the repo as private")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task = load_task(args.task)
    labels, questions = task["labels"], task["questions"]
    rows = load_rows(args.data, labels)
    if args.val_data:
        train_rows, val_rows = rows, load_rows(args.val_data, labels)
    else:
        train_rows, val_rows = split_rows(rows, args.val_frac, args.seed)
    mass = np.array([sum(r["target"][i] for r in train_rows) for i in range(len(labels))])
    print(f"train {len(train_rows)} rows, validation {len(val_rows)} rows, {len(questions)} question phrasing(s)")
    print("train label mass:", dict(zip(labels, mass.round(1).tolist())))
    if (mass == 0).any():
        raise SystemExit(f"no training examples for: {[l for l, m in zip(labels, mass) if m == 0]}")

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                        else "mps" if torch.backends.mps.is_available() else "cpu"))
    if args.encoder:
        from transformers import AutoTokenizer
        from laya.common import build_model

        # Borrow Laya's head/config layout, but train the decision head from scratch on a new encoder.
        ref_dir = resolve_base("convaiinnovations/laya", "multilingual")
        cfg = dict(json.loads((ref_dir / "rl_agent_config.json").read_text()), encoder=args.encoder,
                   max_len=args.max_tokens, head_max_len=192, temperature=[1.0, 1.0, 1.0], temperature_by_options={})
        tok = AutoTokenizer.from_pretrained(args.encoder)
        model = build_model(cfg, pretrained=True).to(dev)
        base_dir = None
        lr_head = args.lr_head or 3e-4
        print(f"device {dev}: fresh decision head on {args.encoder}")
    else:
        subfolder = args.subfolder if args.subfolder is not None else (
            "multilingual" if args.base == "convaiinnovations/laya" else None)
        subfolder = None if subfolder in ("", ".") else subfolder
        base_dir = resolve_base(args.base, subfolder)
        agent = laya.Agent(str(base_dir), device=str(dev))
        model, tok, cfg = agent.model, agent.tok, agent.cfg
        lr_head = args.lr_head or 1e-4
        print(f"device {dev}: fine-tuning {args.base}{'/' + subfolder if subfolder else ''}")

    # Weight each example by its target's class weight (~1/sqrt(frequency), mean 1).
    cw = (mass.sum() / mass) ** 0.5
    cw = torch.tensor(cw / cw.mean(), dtype=torch.float32)
    print("class weights:", {l: round(float(w), 2) for l, w in zip(labels, cw)})

    if not args.train_embeddings:
        for p in model.encoder.get_input_embeddings().parameters():
            p.requires_grad_(False)
    if dev.type != "cuda":
        model.encoder.gradient_checkpointing_enable()  # trade speed for memory off-GPU
    train_cfg = dict(cfg, max_len=min(cfg["max_len"], args.max_tokens))

    def items_for(rs, shuffle, c):
        its, owners = [], []
        for r in rs:
            state = row_state(r)
            for qi, q in enumerate(questions):
                its.append(make_item(tok, c, state, qi, q, labels, r["target"], shuffle))
                owners.append(r)
        return its, owners

    val_items, val_owners = items_for(val_rows, False, train_cfg)
    before = predict_logits(model, val_items, tok.pad_token_id, dev)
    print("validation before training:", json.dumps(evaluate(before, val_items, val_owners, len(questions), len(labels))))

    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr_encoder}, {"params": head_params, "lr": lr_head}],
                            weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_rows) * len(questions) / args.batch_size)
    total = steps_per_epoch * args.epochs
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))

    best_crit, best_state, best_logits = float("inf"), None, None
    history, step, t0 = [], 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        items, _ = items_for(train_rows, True, train_cfg)  # fresh option shuffle every epoch
        running = []
        for chunk in length_batches(items, args.batch_size, shuffle=True):
            batch = collate_items([[items[i] for i in chunk]], tok.pad_token_id)
            logits = forward(model, batch, dev)
            tgt = batch["target"][:, :logits.size(1)].to(dev)
            logp = F.log_softmax(logits.masked_fill(~batch["marker_mask"].to(dev), -1e4), -1)
            per_example = -(tgt * logp).sum(-1)
            w = torch.tensor([float((torch.tensor(items[i]["label_target"]) * cw).sum()) for i in chunk], device=dev)
            loss = (per_example * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            running.append(loss.item())
            if dev.type == "mps" and step % 25 == 0:
                torch.mps.empty_cache()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{total} loss {np.mean(running[-50:]):.4f} ({time.time() - t0:.0f}s)",
                      flush=True)
        val_logits = predict_logits(model, val_items, tok.pad_token_id, dev)
        ev = evaluate(val_logits, val_items, val_owners, len(questions), len(labels))
        mean_acc = float(np.mean([v["acc"] for v in ev.values()])) if ev else 0.0
        nll = soft_nll(val_logits, val_items)
        crit = -mean_acc if args.select == "acc" else nll
        history.append({"epoch": epoch, "val_mean_acc": round(mean_acc, 4), "val_soft_nll": round(nll, 4),
                        "val": ev, "seconds": round(time.time() - t0)})
        print(f"epoch {epoch}: validation {json.dumps(ev)} mean acc {mean_acc:.4f} soft NLL {nll:.4f}", flush=True)
        if crit < best_crit:
            best_crit = crit
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_logits = val_logits

    temps = {qi: fit_temperature(best_logits, val_items, qi) for qi in range(len(questions))}
    final = evaluate(best_logits, val_items, val_owners, len(questions), len(labels), temps)
    print("fitted temperatures:", temps, "validation after calibration:", json.dumps(final))

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    if base_dir is None:
        out.mkdir(parents=True)
        tok.save_pretrained(out / "tokenizer")
        model.encoder.config.save_pretrained(out / "encoder")
    else:
        shutil.copytree(base_dir, out, ignore=shutil.ignore_patterns(
            "model.safetensors", "*.onnx", "onnx", "multilingual", "typed-decisions", "assets", "*.md", ".git*", ".cache"))
    save_file({k: v.contiguous() for k, v in best_state.items()}, str(out / "model.safetensors"))

    # Laya applies one temperature per (question type, option count) bucket; average within a bucket.
    buckets: dict[str, list[float]] = {}
    for qi, q in enumerate(questions):
        iq = internal(q)
        buckets.setdefault(temp_bucket(QTYPES[iq["t"]], len(iq["crit"])), []).append(temps[qi])
    new_cfg = dict(cfg)
    new_cfg["temperature_by_options"] = {**cfg.get("temperature_by_options", {}),
                                         **{b: round(float(np.mean(ts)), 2) for b, ts in buckets.items()}}
    new_cfg["training"] = dict(cfg.get("training", {}), fine_tuned={
        "base": args.encoder or args.base, "train_rows": len(train_rows), "val_rows": len(val_rows),
        "epochs": args.epochs, "select": args.select, "seed": args.seed, "labels": labels})
    (out / "rl_agent_config.json").write_text(json.dumps(new_cfg, indent=2))
    (out / "task.json").write_text(json.dumps(task, indent=2, ensure_ascii=False))
    (out / "training_log.json").write_text(json.dumps(
        {"args": vars(args), "history": history, "temperatures": temps, "validation": final}, indent=2))
    print(f"saved {out}  (load it with laya.Agent({str(out)!r}))")

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.push_to_hub)
        print(f"uploaded to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
