import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import BertTokenizer, BertModel
import torch
import openai
import os
from dotenv import load_dotenv
import joblib
import numpy as np
import sys
import json

load_dotenv()
# openai.api_key = os.getenv("OPENAI_KEY")
openai.api_key = ''
script_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(script_dir, "scored_model_bert_cpu.joblib")

class BertEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        question_embeddings = self._get_embeddings(X['question_text'])
        answer_embeddings = self._get_embeddings(X['answer_text'])
        return np.concatenate([question_embeddings, answer_embeddings], axis=1)

    def _get_embeddings(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten())
        return np.array(embeddings)

if os.path.exists(model_file):
    pipeline = joblib.load(model_file)
else:
    print("모델 없음", file=sys.stderr)
    sys.exit(1)

def get_detailed_score(question, answer):
    prompt = f"""
    다음 질문에 대한 답변을 아래 기준에 따라 평가하세요:
    1. 관련성 (0-25): 답변이 질문을 얼마나 잘 다루고 있는가?
    2. 명확성 (0-25): 답변이 얼마나 명확하고 이해하기 쉬운가?
    3. 깊이 (0-25): 답변이 얼마나 깊이 있고 포괄적인가?
    4. 실용성 (0-25): 답변이 실제 지식이나 경험을 얼마나 잘 보여주는가?
    질문: {question}
    답변: {answer}
    각 기준에 대한 점수만 제공하세요. 총점은 자동으로 계산됩니다.
    형식:
    관련성: [점수]
    명확성: [점수]
    깊이: [점수]
    실용성: [점수]
    """
    completion = openai.chat.completions.create(
        model='gpt-4',
        messages=[
            {"role": "system", "content": "당신은 전문 면접 평가자입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    content = completion.choices[0].message.content

    scores = re.findall(r'(\w+): (\d+)', content)
    total_score = sum(int(score) for _, score in scores)

    return total_score

def get_final_score(question, answer):
    detailed_score = get_detailed_score(question, answer)
    features = pd.DataFrame({'question_text': [question], 'answer_text': [answer]})
    predicted_score = pipeline.predict(features)[0]
    predicted_score = float(predicted_score)

    if detailed_score <= 20:
        final_score = (0.8 * detailed_score) + (0.2 * predicted_score)
    else:
        final_score = (0.5 * detailed_score) + (0.5 * predicted_score)

    return round(final_score, 1)

def main():
    for line in sys.stdin:
        try:
            data = json.loads(line)
            question = data['question']
            answer = data['answer']
            f_score = get_final_score(question, answer)
            result = {
                'final_score': f_score
            }
            print(json.dumps(result))
            sys.stdout.flush()
        except json.JSONDecodeError:
            print(json.dumps({'error': 'Invalid JSON input'}), file=sys.stderr)
        except KeyError:
            print(json.dumps({'error': 'Missing required keys in input'}), file=sys.stderr)
        except Exception as e:
            print(json.dumps({'error': str(e)}), file=sys.stderr)

if __name__ == "__main__":
    main()
