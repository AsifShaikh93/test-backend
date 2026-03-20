from evaluator import evaluate_answer

async def evaluate_interview_answers(answers):
    results = []
    for qa in answers:
        question = qa.get("question")
        answer = qa.get("answer", "")

        if not answer.strip():
            evaluation = {"score": 0, "feedback": "No answer provided."}
        else:
            evaluation = evaluate_answer(question, answer)

        results.append({
            "question": question,
            "score": evaluation.get("score", 0),
            "feedback": evaluation.get("feedback", "N/A")
        })
    return results
