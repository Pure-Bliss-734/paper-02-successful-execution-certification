"""Rebuild each scientific request and replay its recorded stage dependency."""

from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
import hashlib, json, sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools/scorers"))
from scaffold import original_controlled, ablation_prompt, interventions
from scaffold import (
    controlled as methods,
    matched,
    future,
    natural,
    glm_future,
    glm_shared,
)
from pcg_v3_m30.methods import normalize_carried_state_v2
from response_replay import decode


def read(p):
    return json.loads(p.read_bytes())


def lines(p):
    with p.open(encoding="utf8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sha(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def attempt_value(r):
    ch = (r.get("returned") or {}).get("choices") or []
    txt = ch[0].get("text") if ch else None
    return decode(txt, wrapped=True) or {}


class Scaffold:
    def __init__(self):
        d = ROOT / "protocol"
        self.schedule = lines(d / "episodes.jsonl")
        self.profiles = read(d / "generation_profiles.json")
        self.tasks = {
            k: read(d / (k + "_tasks.json"))
            for k in ("controlled", "repo", "future", "repair")
        }
        self.tasks["deletion"] = interventions.deletion_arms(
            read(d / "ablation_inputs.json"), self.tasks["future"]
        )
        self.repair_states = {}
        repair_refs = {
            r["task_id"]: r["expected_state"]
            for r in lines(ROOT / "data/state_targets.jsonl")
        }
        for ep in self.schedule:
            if (
                ep["layer"] == "natural_repair"
                and ep["candidate"] not in self.repair_states
            ):
                spec = self.tasks["repair"][ep["candidate"]]
                ref = repair_refs[ep["task_id"]]
                self.repair_states[ep["candidate"]] = interventions.repair_arm(
                    spec, ref
                )
        self.gold = read(d / "controlled_gold_states.json")
        self.calls = defaultdict(list)
        self.parsed = {}
        for p in (ROOT / "evidence/calls").glob("*.jsonl"):
            for a in lines(p):
                self.calls[(p.stem, a["episode"], a["stage"])].append(a)
        for p in (ROOT / "data").glob("parsed_*.jsonl"):
            for r in lines(p):
                self.parsed[r["row_id"]] = r["parsed_output"]
        self.sel = {}
        for key, aa in self.calls.items():
            chosen = [a for a in aa if a["selected"]]
            # Some exhausted failures have no admitted attempt in the historical stage packet.
            assert len(chosen) <= 1, key
            if not chosen:
                assert all(a["response_status"] == "no_return" for a in aa), key
            self.sel[key] = (
                chosen[0]
                if chosen
                else sorted(aa, key=lambda a: a["attempt_index"])[-1]
            )

    def history(self, ref, stage=None):
        layer, ep = ref.split("/")
        choices = [
            v
            for (l, e, s), v in self.sel.items()
            if l == layer and e == ep and (stage is None or stage == s)
        ]
        assert len(choices) == 1, (ref, stage)
        return choices[0]["request"]

    def messages(self, row, stage, config, previous):
        l = row["layer"]
        t = row["task_id"]
        c = row.get("condition")
        model = config["model"]
        profile = SimpleNamespace(
            model=model,
            temperature=config["temperature"],
            thinking="disabled",
            max_tokens=config["max_tokens"],
        )
        is_glm = l == "glm52_third_family_180"
        if (
            l
            in (
                "controlled_formal_1200",
                "qwen",
                "repository_positive_control_300",
                "capacity_sensitivity_27",
            )
            or is_glm
            and stage == "state_generation"
        ):
            task = SimpleNamespace(
                **self.tasks[
                    "repo"
                    if l
                    in ("repository_positive_control_300", "capacity_sensitivity_27")
                    else "controlled"
                ][t]
            )
            if is_glm:
                c = "N"
            renderer = original_controlled if l == "controlled_formal_1200" else methods
            if stage in ("state_generation", "history_direct"):
                request = renderer.build_initial_request(c, task, profile)
            elif stage == "state_review":
                norm = normalize_carried_state_v2(
                    previous.get("carried_state"), "structured"
                )
                feedback = (
                    methods.BLIND_REVIEW_FEEDBACK
                    if c == "B"
                    else methods.visible_checker_category(
                        norm, task.visible_checker_reference
                    )["category"]
                )
                request = renderer.build_review_request(
                    condition=c,
                    task=task,
                    candidate_state=norm["state"],
                    feedback_category=feedback,
                    config=profile,
                )
            elif stage == "state_only_action":
                request = methods.build_state_only_action_request(
                    task=task,
                    carried_state=previous.get("carried_state"),
                    config=profile,
                )
            else:
                raise ValueError((l, stage))
            return request["messages"]
        if l == "paired_intervention_480":
            if row["input_kind"] == "model_state":
                p = self.parsed[row["source_episode"]]
                state = (p.get("carried_state") or {}).get("state")
                if not isinstance(state, dict):
                    state = {"entities": [], "relations": [], "exclusions": []}
            else:
                state = self.tasks["controlled"][t]["visible_checker_reference"][
                    "state"
                ]
            return matched.prompt_messages(state, row["action_type"])
        if is_glm and stage in (
            "gold_state_action_full_contract",
            "state_only_action_full_contract",
        ):
            if stage.startswith("gold"):
                state = self.gold[t]
            else:
                s = normalize_carried_state_v2(previous.get("carried_state"), "natural")
                state = (
                    s["state"]
                    if s["parse_success"]
                    else {"entities": [], "relations": [], "exclusions": []}
                )
            return glm_shared.full_action_messages(state, row["action_type"])
        if l in ("future", "stage2_ablation_240") or is_glm:
            task = self.tasks["future"][t]
            conf = {
                "deepseek": {"model": model, "max_tokens": config["max_tokens"]},
                "qwen": {"model": model, "max_tokens": config["max_tokens"]},
            }
            if stage == "future_base_state":
                return glm_future.future_stage1_messages(task)
            if stage == "future_transition_action":
                return glm_future.future_stage2_messages(
                    previous.get("carried_state") or "", task
                )
            provider = row.get("provider", "deepseek")
            provider = "deepseek" if provider == "glm" else provider
            if stage == "stage1":
                return future.stage1_request(provider, task, conf)["messages"]
            carried = (
                self.tasks["deletion"][t][row["arm"]]
                if l == "stage2_ablation_240"
                else previous.get("carried_state") or ""
            )
            renderer = ablation_prompt if l == "stage2_ablation_240" else future
            return renderer.stage2_request(provider, carried, task, conf)["messages"]
        if l == "natural_repair":
            cand = dict(
                self.tasks["repair"][row["candidate"]],
                repaired_state=self.repair_states[row["candidate"]],
            )
            return natural.build_request(cand, row["arm"])["messages"]
        if l == "bridge":
            return self.history(row["source_episode"])["messages"]
        raise ValueError(l)

    def verify_requests(self):
        result = Counter()
        missing = []
        for row in self.schedule:
            last = {}
            for stage in row["stages"]:
                s = stage["name"]
                profile = self.profiles[stage["profile"]]
                key = (row["layer"], row["episode"], s)
                a = self.sel[key]
                try:
                    request = dict(
                        profile, messages=self.messages(row, s, profile, last)
                    )
                    assert request == a["request"]
                    for trial in self.calls[key]:
                        assert (
                            request == trial["request"]
                            and sha(request) == trial["request_sha256"]
                        )
                except (AssertionError, KeyError, ValueError, AttributeError) as e:
                    missing.append((key, type(e).__name__, str(e)[:80]))
                else:
                    result[row["layer"]] += 1
                last = attempt_value(a)
        return result, missing


if __name__ == "__main__":
    runner = Scaffold()
    ok, errors = runner.verify_requests()
    print(ok, Counter((row[0], kind, msg) for row, kind, msg in errors).most_common(16))
    if errors:
        raise SystemExit(1)
