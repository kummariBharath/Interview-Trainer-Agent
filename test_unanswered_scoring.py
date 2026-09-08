"""
test_unanswered_scoring.py
---------------------------
Automated test suite verifying the fix for unanswered question scoring.

Tests 1-6 map directly to the required test cases in the problem description:
  TEST 1: Blank string submission -> 0 points, skipped, no Granite LLM call.
  TEST 2: Whitespace-only submission -> 0 points.
  TEST 3: Real but poor answer -> Normal Granite evaluation with low score.
  TEST 4: Strong answer -> Normal Granite evaluation with appropriate high score.
  TEST 5: Mixed session (some skipped, some answered) -> Skipped = 0, no score leak.
  TEST 6: Complete interview final report -> Accurately counts answered/unanswered and scores.
"""

import sys
from typing import Any
from unittest.mock import MagicMock, patch

from agents.evaluator import evaluate_answer, is_unanswered_response
from agents.interview_agent import CandidateProfile, InterviewAgent


def test_1_blank_answer():
    print("\n--- TEST 1: Blank string submission ---")
    # Patch _get_client so we can verify Granite is NEVER called for blank inputs
    with patch("agents.evaluator._get_client") as mock_get_client:
        res = evaluate_answer(
            question="Explain Python decorators.",
            answer="",
            role="Python Developer",
            experience="2-5 years",
            interview_type="Technical",
        )
        mock_get_client.assert_not_called()
        assert res["score"] == 0, f"Expected score 0, got {res['score']}"
        assert "skipped" in res["relevance"].lower() or "unanswered" in res["relevance"].lower()
        print("[PASS] TEST 1 PASSED: Blank answer returned score=0 without calling Granite.")


def test_2_whitespace_answer():
    print("\n--- TEST 2: Whitespace-only submission ---")
    with patch("agents.evaluator._get_client") as mock_get_client:
        res = evaluate_answer(
            question="Explain Python decorators.",
            answer="   \n\t  ",
            role="Python Developer",
            experience="2-5 years",
            interview_type="Technical",
        )
        mock_get_client.assert_not_called()
        assert res["score"] == 0, f"Expected score 0, got {res['score']}"
        print("[PASS] TEST 2 PASSED: Whitespace answer returned score=0 without calling Granite.")


def test_3_poor_answer():
    print("\n--- TEST 3: Real but poor answer ---")
    # Mock Granite response to simulate a poor answer evaluation (score 2)
    mock_granite_resp = '{"score": 2, "relevance": "Weak answer", "technical_accuracy": "Incorrect", "completeness": "Very low", "clarity": "Poor", "strengths": [], "weaknesses": ["Lacks depth"], "missing_points": ["Everything"], "improvement_advice": "Study more.", "better_answer_hint": "A decorator is..."}'
    
    with patch("agents.evaluator._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.generate_eval.return_value = mock_granite_resp
        mock_get_client.return_value = mock_client

        res = evaluate_answer(
            question="Explain Python decorators.",
            answer="I don't really know, maybe it's something with @ symbol.",
            role="Python Developer",
            experience="2-5 years",
            interview_type="Technical",
        )
        mock_client.generate_eval.assert_called_once()
        assert res["score"] == 2, f"Expected low score (2), got {res['score']}"
        print(f"[PASS] TEST 3 PASSED: Poor answer called Granite and returned appropriate low score ({res['score']}).")


def test_4_strong_answer():
    print("\n--- TEST 4: Strong answer ---")
    mock_granite_resp = '{"score": 9, "relevance": "Highly relevant", "technical_accuracy": "Accurate", "completeness": "Comprehensive", "clarity": "Excellent", "strengths": ["Clear explanation", "Good code example"], "weaknesses": [], "missing_points": [], "improvement_advice": "Great job.", "better_answer_hint": ""}'

    with patch("agents.evaluator._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.generate_eval.return_value = mock_granite_resp
        mock_get_client.return_value = mock_client

        res = evaluate_answer(
            question="Explain Python decorators.",
            answer="A decorator in Python is a design pattern used to extend or modify the behavior of a function or method without modifying its source code. It wraps another function and uses @decorator syntax.",
            role="Python Developer",
            experience="2-5 years",
            interview_type="Technical",
        )
        mock_client.generate_eval.assert_called_once()
        assert res["score"] == 9, f"Expected high score (9), got {res['score']}"
        print(f"[PASS] TEST 4 PASSED: Strong answer called Granite and returned high score ({res['score']}).")


def test_5_mixed_skipped_and_answered():
    print("\n--- TEST 5: Several questions with mixed skipped and answered ---")
    profile = CandidateProfile(
        name="Test User",
        role="Python Developer",
        experience="2-5 years",
        interview_type="Technical",
        difficulty="Medium",
    )
    agent = InterviewAgent(profile=profile)
    
    # Mock question generation so it doesn't need live Granite API
    with patch("agents.interview_agent.generate_question") as mock_gen_q, \
         patch("agents.evaluator._get_client") as mock_get_client:

        mock_gen_q.return_value = "Explain Python memory management and garbage collection in detail?"
        
        mock_client = MagicMock()
        # Q1 answered strongly (score 9)
        mock_client.generate_eval.return_value = '{"score": 9, "relevance": "Good", "technical_accuracy": "Good", "completeness": "Good", "clarity": "Good", "strengths": ["Clear"], "weaknesses": [], "missing_points": [], "improvement_advice": "Good", "better_answer_hint": ""}'
        mock_get_client.return_value = mock_client

        agent.start_session()
        
        # Q1: Strong answer
        eval1 = agent.submit_answer("Memory management in Python uses reference counting and a generational garbage collector.")
        assert eval1["score"] == 9, f"Q1 expected 9, got {eval1['score']}"

        # Q2: Skipped answer button string
        eval2 = agent.submit_answer("[Skipped — no answer provided]")
        assert eval2["score"] == 0, f"Q2 expected 0, got {eval2['score']}"

        # Q3: Whitespace only
        eval3 = agent.submit_answer("   ")
        assert eval3["score"] == 0, f"Q3 expected 0, got {eval3['score']}"

        # Q4: Strong answer (score 8)
        mock_client.generate_eval.return_value = '{"score": 8, "relevance": "Good", "technical_accuracy": "Good", "completeness": "Good", "clarity": "Good", "strengths": ["Clear"], "weaknesses": [], "missing_points": [], "improvement_advice": "Good", "better_answer_hint": ""}'
        eval4 = agent.submit_answer("Python's GC manages cyclic references using container objects inspection.")
        assert eval4["score"] == 8, f"Q4 expected 8, got {eval4['score']}"

        # Check session evaluations list
        session = agent.get_session()
        scores = [e["score"] for e in session.evaluations]
        assert scores == [9, 0, 0, 8], f"Expected scores [9, 0, 0, 8], got {scores}"
        
        print("[PASS] TEST 5 PASSED: Skipped questions scored 0, answered questions scored independently without leak.")


def test_6_final_report_unanswered():
    print("\n--- TEST 6: Complete interview final report with unanswered questions ---")
    profile = CandidateProfile(
        name="Test Candidate",
        role="Python Developer",
        experience="2-5 years",
        interview_type="Technical",
        difficulty="Medium",
    )
    agent = InterviewAgent(profile=profile)

    with patch("agents.interview_agent.generate_question") as mock_gen_q, \
         patch("agents.evaluator._get_client") as mock_get_client:

        mock_gen_q.return_value = "Explain Python decorators and generators in detail?"
        
        mock_client = MagicMock()
        mock_client.generate_eval.return_value = '{"score": 8, "relevance": "Good", "technical_accuracy": "Good", "completeness": "Good", "clarity": "Good", "strengths": ["Good"], "weaknesses": [], "missing_points": [], "improvement_advice": "Good", "better_answer_hint": ""}'
        mock_get_client.return_value = mock_client

        agent.start_session()

        # Submit 5 questions: Q1 answered (8), Q2 skipped (0), Q3 answered (8), Q4 skipped (0), Q5 skipped (0)
        agent.submit_answer("Detailed answer to question 1")
        agent.submit_answer("[Skipped — no answer provided]")
        agent.submit_answer("Detailed answer to question 3")
        agent.submit_answer("")
        agent.submit_answer("   ")

        agent.get_session().is_complete = True
        report = agent.generate_final_report()

        assert report["total_questions"] == 5, f"Expected 5 total questions, got {report['total_questions']}"
        assert report["questions_answered"] == 2, f"Expected 2 answered questions, got {report['questions_answered']}"
        assert report["questions_unanswered"] == 3, f"Expected 3 unanswered questions, got {report['questions_unanswered']}"
        assert report["questions_skipped"] == 3, f"Expected 3 skipped questions, got {report['questions_skipped']}"
        
        # Overall score average = (8 + 0 + 8 + 0 + 0) / 5 = 16 / 5 = 3.2
        assert report["overall_score"] == 3.2, f"Expected overall score 3.2, got {report['overall_score']}"

        # Verify breakdown items
        breakdown = report["question_breakdown"]
        assert breakdown[0]["score"] == 8
        assert breakdown[1]["score"] == 0
        assert breakdown[1]["answer_preview"] == "[Skipped — no answer provided]"
        assert breakdown[2]["score"] == 8
        assert breakdown[3]["score"] == 0
        assert breakdown[4]["score"] == 0

        print("[PASS] TEST 6 PASSED: Final report accurately displays total, answered, unanswered counts, and correct score calculation.")


def run_all_tests():
    print("============================================================")
    print(" Running Unanswered Question Scoring Bug Verification Tests")
    print("============================================================")
    test_1_blank_answer()
    test_2_whitespace_answer()
    test_3_poor_answer()
    test_4_strong_answer()
    test_5_mixed_skipped_and_answered()
    test_6_final_report_unanswered()
    print("\n============================================================")
    print(" ALL 6 VERIFICATION TESTS PASSED SUCCESSFULLY! [PASS]")
    print("============================================================\n")


if __name__ == "__main__":
    run_all_tests()
