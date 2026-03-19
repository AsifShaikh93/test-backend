from llm import llm
import json

def evaluate_answer(question, answer):
    prompt = f"""
    You are a strict technical interview evaluator.
    Analyze the candidate's answer based on the question.
    
    Question: {question}
    Answer: {answer}

    Return ONLY a valid JSON object with this structure:
    {{
     "score": <number 0-10>,
     "feedback": "<concise feedback>"
    }}
    """
    response = llm.invoke(prompt)
    content = response.content.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(content)
    except:
        return {"score": 0, "feedback": "Failed to parse evaluation."}