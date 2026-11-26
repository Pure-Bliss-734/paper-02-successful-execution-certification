"""Drive a saved episode offline or run one user-selected episode against a compatible provider."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from urllib import error, parse, request

sys.dont_write_bytecode = True
from scaffold.run import ROOT, Scaffold, attempt_value, lines, read, sha
from scaffold.score import check as score_auxiliary, controlled_scores
from collections import Counter


def verify_evidence(scaffold):
    expected = read(ROOT / "references/response_coverage.json")
    actual = {}
    for layer, spec in expected.items():
        attempts = [
            a for (l, e, s), aa in scaffold.calls.items() if l == layer for a in aa
        ]
        counted = Counter()
        for a in attempts:
            assert sha(a["request"]) == a["request_sha256"]
            counted[a["response_status"]] += 1
            if a["response_status"] == "no_return":
                assert a["returned"] is None and "recorded_body_sha256" not in a
                continue
            assert len(a["recorded_body_sha256"]) == 64
            for choice in a["returned"]["choices"]:
                assert (
                    hashlib.sha256(choice["text"].encode("utf8")).hexdigest()
                    == choice["text_sha256"]
                )
                counted["choices"] += 1
        totals = {
            "attempts": len(attempts),
            "logical_calls": len({(a["episode"], a["stage"]) for a in attempts}),
            "episodes": len({a["episode"] for a in attempts}),
            "responses": counted["response_present"],
            "unreturned": counted["no_return"],
            "choices": counted["choices"],
        }
        assert totals == spec, (layer, totals, spec)
        actual[layer] = totals
    return actual


def verify():
    scaffold = Scaffold()
    coverage = verify_evidence(scaffold)
    requested, bad = scaffold.verify_requests()
    if bad:
        raise ValueError(f"Request reconstruction mismatch: {bad[:3]}")
    scores, errors = score_auxiliary(scaffold)
    if errors:
        raise ValueError(f"Endpoint reconstruction mismatch: {errors[:3]}")
    primary, errors = controlled_scores(scaffold)
    if errors:
        raise ValueError(f"Controlled replay mismatch: {errors[:3]}")
    diagnostics = {}
    for name, layer in [("repair_smoke", "natural_repair"), ("bridge_smoke", "bridge")]:
        records = lines(ROOT / "diagnostics" / (name + ".jsonl"))
        matching = 0
        assert len(records) == 10
        for row in records:
            assert (
                sha(row["request"]) == row["request_sha256"]
                and row["formal_layer"] == layer
            )
            matches = [
                v
                for (l, e, s), v in scaffold.sel.items()
                if l == layer
                and e == row["formal_episode"]
                and v["request"] == row["request"]
            ]
            assert len(matches) == 1 and len(row["returned"]["choices"]) == 1
            choice = row["returned"]["choices"][0]
            assert (
                hashlib.sha256(choice["text"].encode("utf8")).hexdigest()
                == choice["text_sha256"]
            )
            matching += choice["text"] == matches[0]["returned"]["choices"][0]["text"]
        assert matching == (10 if name == "repair_smoke" else 8)
        diagnostics[name] = {"attempts": len(records), "same_text_as_formal": matching}
    return {
        "infrastructure_diagnostics": diagnostics,
        "evidence": coverage,
        "requests_reconstructed": dict(requested),
        "auxiliary_outcome_replays": dict(scores),
        "controlled_replays": dict(primary),
        "fresh_provider_calls": 0,
    }


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("Refusing to forward a credential to another endpoint")


def live_response(endpoint, token, payload):
    opener = request.build_opener(NoRedirect())
    req = request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf8"),
        method="POST",
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        },
    )
    for attempt in (1, 2):
        try:
            with opener.open(req, timeout=180) as stream:
                raw = json.load(stream)
            break
        except error.HTTPError as exc:
            if attempt == 2 or exc.code != 429 and not (500 <= exc.code < 600):
                raise
        except error.URLError:
            if attempt == 2:
                raise
    if len(raw.get("choices", [])) != 1:
        raise ValueError("Expected one returned choice")
    choice = raw["choices"][0]
    text = choice["message"].get("content")
    if not isinstance(text, str):
        raise ValueError("Response lacks a text message")
    return {
        "returned": {
            "returned_model": raw.get("model"),
            "choices": [
                {
                    "text": text,
                    "finish_reason": choice.get("finish_reason"),
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            ],
        }
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer")
    parser.add_argument("--episode")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Send one episode to the specified HTTPS provider. Charges may apply.",
    )
    parser.add_argument(
        "--endpoint",
        help="OpenAI-compatible HTTPS chat completions endpoint for the requested model",
    )
    args = parser.parse_args()
    if args.live and not (args.layer and args.episode and args.endpoint):
        parser.error("Live access requires layer, episode and endpoint.")
    if bool(args.layer) != bool(args.episode):
        parser.error("Specify both layer and episode.")
    if not args.episode:
        print(json.dumps(verify(), indent=2, sort_keys=True))
        return
    if args.live:
        url = parse.urlsplit(args.endpoint)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.fragment
            or url.query
        ):
            parser.error(
                "Use an HTTPS endpoint without embedded credentials, query or fragment."
            )
        token = os.environ.get("HANDOFF_PROVIDER_KEY")
        if not token:
            parser.error("Set HANDOFF_PROVIDER_KEY for live execution.")
    scaffold = Scaffold()
    row = next(
        (
            r
            for r in scaffold.schedule
            if (r["layer"], r["episode"]) == (args.layer, args.episode)
        ),
        None,
    )
    if row is None:
        parser.error("The requested episode is absent from the schedule.")
    last = {}
    stages = []
    for descriptor in row["stages"]:
        stage = descriptor["name"]
        cfg = scaffold.profiles[descriptor["profile"]]
        payload = dict(cfg, messages=scaffold.messages(row, stage, cfg, last))
        observed = scaffold.sel[row["layer"], row["episode"], stage]
        if args.live:
            response = live_response(args.endpoint, token, payload)
        else:
            if payload != observed["request"]:
                raise ValueError("Request differs from the saved stage.")
            response = observed
        stages.append(
            {"stage": stage, "request": payload, "response": response["returned"]}
        )
        last = attempt_value(response)
    print(
        json.dumps(
            {
                "layer": args.layer,
                "episode": args.episode,
                "mode": "fresh_inference" if args.live else "recorded",
                "stages": stages,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
