from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import SETTINGS, get_logger
from app.modules.eval_utils import (
    answer_correctness_score,
    answer_relevance_score,
    context_precision_score,
    faithfulness_score,
)
from app.modules.pipeline import PIPELINE
from app.modules.session import SessionMode

_log = get_logger("run_evaluation")

# Edit this for your own document(s) and questions.
QUERIES: list[tuple[str, str]] = [
    ("মাদাম লোইসেলের স্বামীর পেশা কী ছিল?", "কেরানি"),
    ("মাদাম লোইসেলের স্বামী তাকে 'বলে' যাওয়ার জন্য কী পরার পরামর্শ দিয়েছিলেন?", "সুন্দর পোশাক"),
    ("যদি মাদাম লোইসেল সেদিন রাতে হারটি না হারাতেন, তাহলে তার জীবন কেমন হতো?", "স্বাভাবিক ও সম্পদের মায়ায় ভরা থাকতো"),
    ("মাদাম লোইসেল 'বল'-এ যাওয়ার জন্য তার স্বামী তাকে কত টাকা দিতে ইচ্ছা প্রকাশ করেন?", " চারশত ফ্রাঁ।"),
    ("গল্পের শেষে মাদাম ফোরসটিয়ার মাদাম লোইসেলকে হারটি সম্পর্কে কী জানান?", " হারটি নকল ছিল")
]


def run_evaluation() -> dict:
    """Run every (question, expected_answer) pair through the real pipeline and score it."""
    results = []
    for question, expected_answer in QUERIES:
        _log.info("Evaluating: %s", question)
        response = PIPELINE.run(question, session_id=f"eval-{abs(hash(question))}",
                                 session_mode=SessionMode.FRESH)
        scores = {
            "faithfulness": faithfulness_score(response.answer, response.expanded_contexts),
            "answer_relevance": answer_relevance_score(question, response.answer),
            "context_precision": context_precision_score(question, response.expanded_contexts),
            "answer_correctness": answer_correctness_score(question, response.answer, expected_answer),
        }
        results.append({
            "question": question, "expected_answer": expected_answer,
            "generated_answer": response.answer,
            "question_type": response.classification.question_type,
            "is_high_risk": response.classification.is_high_risk,
            "confidence": response.validation.confidence,
            "llm_validated": response.llm_validated, "retried": response.retried,
            "latency_seconds": response.latency_seconds, "scores": scores,
        })

    averages = {k: sum(r["scores"][k] for r in results) / len(results) for k in results[0]["scores"]}
    averages["latency_seconds"] = sum(r["latency_seconds"] for r in results) / len(results)

    report = {"run_timestamp": datetime.now().isoformat(timespec="seconds"),
              "groq_model": SETTINGS.groq_model, "n_queries": len(results),
              "results": results, "averages": averages}

    SETTINGS.evaluation_dir.mkdir(parents=True, exist_ok=True)
    out_path = SETTINGS.evaluation_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _log.info("Saved evaluation report -> %s", out_path)
    return report


def _print_summary(report: dict) -> None:
    print(f"\n=== Evaluation run: {report['run_timestamp']} (model: {report['groq_model']}) ===\n")
    for r in report["results"]:
        print(f"Q: {r['question']}\n  expected : {r['expected_answer']}\n"
              f"  generated: {r['generated_answer']}\n  scores   : {r['scores']}\n")
    print("--- Averages across all queries ---")
    for k, v in report["averages"].items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    _print_summary(run_evaluation())