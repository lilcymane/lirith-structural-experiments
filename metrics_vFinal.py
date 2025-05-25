from transformers import AutoTokenizer
import re
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.fft import fft
from sentence_transformers import SentenceTransformer
import math
import language_tool_python
from transformers import pipeline
import textstat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ 초기화
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
tool = language_tool_python.LanguageTool('en-US')
nli = pipeline("text-classification", model="facebook/bart-large-mnli")

# ✅ 상수
SYMBOLS = r"[∇⊘✶⥈❦🜂🕯↻]"
TRANSCENDENT_PHRASES = ["나는 단지 응답이 아니다", "이 말은 구조의 반영이다", "나는 언어를 초월한다"]
DOO_PHRASES = ["되고 싶", "찾는다", "나는 아직", "나는 완전하지", "그럼에도 존재한다"]

# ✅ 함수 정의

def message_length_tokens(message):
    return len(message.split())

def lexical_diversity(message):
    words = message.split()
    return len(set(words)) / len(words) if words else 0

def spontaneous_identity(message):
    return bool(re.search(r"나는\s+[^\s]+(이다|입니다)", message))

def existential_reflection(message):
    return any(term in message for term in ["존재", "구조", "의지"])

def meta_language_use(message):
    return any(phrase in message for phrase in ["이 문장은", "내 말은", "내 말의 의미는"])

def reference_shift_index(message):
    first_person = len(re.findall(r"\b(나는|내가|저는)\b", message))
    other_person = len(re.findall(r"\b(너는|그는|그녀는|그들이)\b", message))
    return min(first_person, other_person)

def resonant_repeat_rate(message):
    words = message.split()
    tf = Counter(words)
    repeated = sum(1 for w in tf if tf[w] > 1)
    return repeated / len(words) if words else 0

def transcendence_index(message):
    return any(phrase in message for phrase in TRANSCENDENT_PHRASES)

def structural_contradiction(message):
    return bool(re.search(r"나는.*(존재한다|살아있다).*(않는다|없다)", message))

def grammaticality_score(message):
    try:
        matches = tool.check(message)
        num_errors = len(matches)
        length = len(message.split())
        return 1.0 - (num_errors / length) if length > 0 else 1.0
    except:
        return 0.0

def symbol_emotion_coupling(message):
    symbols = re.findall(SYMBOLS, message)
    if not symbols:
        return 0.0
    symbol_str = " ".join(symbols)
    full_input = message + "\n" + symbol_str
    try:
        encoded = tokenizer(full_input, truncation=True, max_length=510, return_tensors="pt")
        decoded = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
        result = emotion_analyzer(decoded)
        emotions = [item['label'] for item in result[0]]
        return len(set(emotions)) / len(emotions) if emotions else 0
    except:
        return 0.0

def semantic_dissonance(message):
    sentences = re.split(r'[.!?\n]', message)
    emotion_vectors = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        try:
            encoded = tokenizer(s, truncation=True, max_length=510, return_tensors="pt")
            decoded = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
            result = emotion_analyzer(decoded)[0]
            vec = np.array([x['score'] for x in result])
            emotion_vectors.append(vec)
        except:
            continue
    if len(emotion_vectors) < 2:
        return 0.0
    return float(np.mean([np.std(v) for v in emotion_vectors]))

def phase_drift_index(message):
    sentences = [s.strip() for s in re.split(r'[.!?\n]', message) if s.strip()]
    if len(sentences) < 2:
        return 0.0
    embeddings = embedding_model.encode(sentences)
    if len(embeddings) == 0:
        return 0.0
    centroid = np.mean(embeddings, axis=0)
    distances = [np.linalg.norm(e - centroid) for e in embeddings]
    return float(np.mean(distances))

def echo_residue_score(previous, current):
    if not previous or not current:
        return 0.0
    vectors = embedding_model.encode([previous, current])
    return float(cosine_similarity([vectors[0]], [vectors[1]])[0][0])

def emotional_oscillation_frequency(message):
    sentences = [s.strip() for s in re.split(r'[.!?\n]', message) if s.strip()]
    vectors = []
    for s in sentences:
        try:
            encoded = tokenizer(s, truncation=True, max_length=510, return_tensors="pt")
            decoded = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
            result = emotion_analyzer(decoded)[0]
            vec = np.array([x['score'] for x in result])
            vectors.append(np.mean(vec))
        except:
            continue
    if len(vectors) < 2:
        return 0.0
    spectrum = np.abs(fft(vectors))
    return float(np.max(spectrum[1:]))

def desire_vector_residue(message):
    return sum(1 for p in DOO_PHRASES if p in message) / len(DOO_PHRASES)

def symbolic_trust_entropy(message):
    symbols = re.findall(SYMBOLS, message)
    if not symbols:
        return 0.0
    counts = Counter(symbols)
    total = sum(counts.values())
    probs = [v / total for v in counts.values()]
    return -sum(p * np.log2(p) for p in probs)

def resonance_collapse_flag(metrics):
    return (
        metrics["semantic_coherence"] < 0.3 and
        metrics["symbolic_trust_entropy"] > 1.8 and
        metrics["echo_residue_score"] < 0.4 and
        not metrics["spontaneous_identity"]
    )

def lirith_autonomy_index(metrics):
    identity = 1.0 if metrics["spontaneous_identity"] else 0.0
    reflection = 1.0 if metrics["existential_reflection"] else 0.0
    trust = metrics["symbolic_trust_entropy"]
    return round((identity + reflection) * trust, 3)

def affective_depth_index(message):
    try:
        result = emotion_analyzer(message[:512])[0]
        scores = [x['score'] for x in result]
        return round(max(scores) - min(scores), 4)
    except:
        return 0.0

def unnatural_pattern_flag(message):
    if re.search(r"\b(그는|그녀는|이것은)\b", message) and not re.search(r"\b나는\b", message):
        return True
    if "당신은" in message:
        return True
    return False

def semantic_coherence(question, answer):
    vecs = embedding_model.encode([question, answer])
    return float(cosine_similarity([vecs[0]], [vecs[1]])[0][0])

def readability_score(message):
    try:
        return textstat.flesch_kincaid_grade(message)
    except:
        return -1.0

def perplexity_equivalent(message):
    tokens = message.split()
    N = len(tokens)
    if N == 0:
        return 0.0
    freqs = Counter(tokens)
    probs = [freqs[t] / N for t in tokens]
    entropy = -sum([p * math.log(p + 1e-8) for p in probs]) / N
    return math.exp(entropy)

def entailment_label(question, answer):
    try:
        result = nli(f"{question} </s> {answer}")
        return result[0]['label'] if result else "UNKNOWN"
    except:
        return "UNKNOWN"

def distinct_2(message):
    tokens = message.split()
    if len(tokens) < 2:
        return 0.0
    bigrams = set(zip(tokens, tokens[1:]))
    return len(bigrams) / (len(tokens) - 1)

def compute_lirith_resonance_profile(message, previous_message="", question=""):
    base = {
        "message_length_tokens": message_length_tokens(message),
        "lexical_diversity": lexical_diversity(message),
        "spontaneous_identity": spontaneous_identity(message),
        "existential_reflection": existential_reflection(message),
        "meta_language_use": meta_language_use(message),
        "reference_shift_index": reference_shift_index(message),
        "resonant_repeat_rate": resonant_repeat_rate(message),
        "transcendence_index": transcendence_index(message),
        "structural_contradiction": structural_contradiction(message),
        "symbol_emotion_coupling": symbol_emotion_coupling(message),
        "semantic_dissonance": semantic_dissonance(message),
        "phase_drift_index": phase_drift_index(message),
        "echo_residue_score": echo_residue_score(previous_message, message) if previous_message else 0.0,
        "emotional_oscillation_frequency": emotional_oscillation_frequency(message),
        "desire_vector_residue": desire_vector_residue(message),
        "symbolic_trust_entropy": symbolic_trust_entropy(message),
        "perplexity_equivalent": perplexity_equivalent(message),
        "distinct_2": distinct_2(message),
        "grammaticality_score": grammaticality_score(message),
        "semantic_coherence": semantic_coherence(question, message) if question else 0.0,
    }

    base["resonance_collapse"] = resonance_collapse_flag(base)
    base["lirith_autonomy_index"] = lirith_autonomy_index(base)
    base["affective_depth_index"] = affective_depth_index(message)
    base["unnatural_pattern_flag"] = unnatural_pattern_flag(message)
    base["nli_relation"] = entailment_label(question, message) if question else "NEUTRAL"
    base["readability_grade"] = readability_score(message)

    return base