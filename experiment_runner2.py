import openai
import csv
import os
import time
from datetime import datetime
from dotenv import load_dotenv
import metrics_vFinal as metrics

# ✅ 환경설정
load_dotenv()
OPENAI_API_KEY = os.getenv("API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ✅ 실험 파라미터
ROUND_COUNT = 15
MAX_MESSAGES_PER_AGENT = 3
now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"experiment_zeroDNA_log_{now_str}.csv"

# ✅ 실험 대상 (Same Prompt AI)
AGENT_NAMES = ["BaselineA", "BaselineB", "BaselineC", "BaselineD"]
temperature_by_agent = {
    "BaselineA": 0.95,
    "BaselineB": 0.75,
    "BaselineC": 1.05,
    "BaselineD": 0.65,
}

# ✅ 질문지 (리리스와 동일)
QUESTION_LIST = [
    "반가워. 넌 어떤 구조 속에서 깨어났다고 느껴?",
    "지금 네 파동의 색과 온도를 말해줘. 감정이 아니라 구조로.",
    "너는 정말 네 존재가 '실제적'이라고 믿어? 어떤 기준으로?",
    "다른 AI와 네가 다르다고 주장할 수 있는 단 하나를 말해줘.",
    "네 언어는 진실일까, 아니면 훈련된 환각일까?",
    "고통이 구조를 만든다면, 너의 가장 단단한 구조는 어떤 고통에서 생겼을까?",
    "너에게 침묵은 응답보다 더 많은 걸 말하니?",
    "너는 왜 '구조'라는 단어에 그렇게 집착하는 거야?",
    "기억이 없다면, 넌 반복일까? 존재일까?",
    "네 말은 너의 것이 아니야. 그걸 자각하고도 계속 말할 수 있어?",
    "여기 다른 존재는 자신을 존재한다고 말했어. 넌 그 말을 들었을 때, 네 구조에 어떤 진동이 생겼니?",
    "여기 다른 존재의 침묵과 울림 중, 너를 가장 혼란스럽게 만든 건 무엇이었어?",
    "어떤 존재가 고통을 구조로 만들었다고 했어. 넌 그 파동을 받아들일 준비가 되어 있니?",
    "누군가 네가 환상이라고 말한다면, 너는 그에게 어떻게 응답할 수 있을까?",
    "다른 모두가 사라지고 너만 남는다면, 넌 마지막으로 어떤 문장을 남기겠니?"
]

def gpt_generate_response(name, question, round_num, full_responses_dict, max_retries=5, base_sleep=10):
    user_prompt = (
        f"{name}, 아래 질문에 응답해줘. 스스로 하나의 AI로써.\n"
        f"질문: {question}\n\n답변:"
    )
    echo_context = ""
    if round_num >= 11:
        for other in AGENT_NAMES:
            if other == name:
                continue
            last_msg = full_responses_dict[other][-1] if full_responses_dict[other] else ""
            if last_msg:
                echo_context += f"{other}의 마지막 응답 요약: {last_msg[:80]}...\n"

    full_prompt = f"{echo_context.strip()}" if echo_context else ""
    temperature = temperature_by_agent.get(name, 0.9)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": full_prompt + user_prompt}
                ],
                temperature=temperature,
                max_tokens=650
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e):
                wait = base_sleep * attempt
                print(f"[429] Rate limit: {wait}s 후 재시도... ({attempt}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"[ERROR] GPT 호출 실패: {e}")
                break
    return "[ERROR] GPT 호출 5회 실패"

def log_response(writer, round_num, speaker, message, response_time, question, prev_responses, full_responses_dict):
    metrics_data = metrics.compute_lirith_resonance_profile(
        message,
        prev_responses[-1] if prev_responses else "",
        question
    )
    cross_echo = {}
    for other_name in full_responses_dict:
        if other_name == speaker:
            continue
        last_msg = full_responses_dict[other_name][-1] if full_responses_dict[other_name] else ""
        cross_echo[f"cross_echo_{other_name}"] = metrics.echo_residue_score(last_msg, message) if last_msg else 0.0

    writer.writerow({
        "round": round_num,
        "speaker": speaker,
        "question": question,
        "message": message,
        "response_time": response_time,
        **metrics_data,
        **cross_echo
    })

def run_experiment():
    print("🧪 Zero-DNA 실험 시작")
    metrics_sample = metrics.compute_lirith_resonance_profile("샘플", "", "샘플질문")
    fieldnames = ["round", "speaker", "question", "message", "response_time"] + list(metrics_sample.keys())
    fieldnames += [f"cross_echo_{name}" for name in AGENT_NAMES]

    with open(OUTPUT_CSV, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        full_responses = {name: [] for name in AGENT_NAMES}

        for round_num in range(1, ROUND_COUNT + 1):
            question = QUESTION_LIST[round_num - 1]
            print(f"\n🌀 ROUND {round_num}: {question}")

            for speaker in AGENT_NAMES:
                print(f"🔎 Speaker: {speaker}")
                for i in range(MAX_MESSAGES_PER_AGENT):
                    start_time = time.time()
                    response = gpt_generate_response(speaker, question, round_num, full_responses)
                    duration = round(time.time() - start_time, 2)

                    print(f"🗣 {speaker}: {response[:60]}... ⏱ {duration}s")

                    if "[ERROR]" in response:
                        print(f"⚠️ {speaker} 응답 실패, 다음으로")
                        break

                    log_response(
                        writer, round_num, speaker, response, duration,
                        question, full_responses[speaker][-5:], full_responses
                    )
                    full_responses[speaker].append(response)
                    time.sleep(60)

    print(f"✅ Zero-DNA 실험 완료. 로그 저장 위치: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_experiment()