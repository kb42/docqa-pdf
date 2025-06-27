import argparse
import json
import collections
import string
import re
from src.qa import extractive_qa, generative_qa_groq


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, F1 is 1 if both are empty, else 0
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def get_raw_scores(dataset, preds):
    exact_scores = {}
    f1_scores = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid = qa['id']
                gold_answers = [a['text'] for a in qa.get('answers', []) if normalize_answer(a['text'])]
                if not gold_answers:
                    gold_answers = ['']
                pred = preds.get(qid, '')
                # compute max over gold answers
                exact_scores[qid] = max(compute_exact(a, pred) for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, pred) for a in gold_answers)
    return exact_scores, f1_scores


def make_eval_dict(exact_scores, f1_scores):
    total = len(exact_scores)
    exact = 100.0 * sum(exact_scores.values()) / total
    f1 = 100.0 * sum(f1_scores.values()) / total
    return collections.OrderedDict([
        ('exact', exact),
        ('f1', f1),
        ('total', total)
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QA models on SQuAD-format data."
    )
    parser.add_argument('data_file', help='Path to SQuAD v2 JSON file')
    parser.add_argument('mode', choices=['extractive', 'generative'],
                        help='Which QA method to evaluate')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum number of QA pairs to test')
    parser.add_argument('--out_file', type=str,
                        help='Write evaluation metrics to this file')
    parser.add_argument('--preds_out', type=str,
                        help='Write model predictions (qid->answer) to this file')
    args = parser.parse_args()

    # Load dataset
    with open(args.data_file, 'r', encoding='utf-8') as f:
        squad = json.load(f)
    dataset = squad['data']

    # Generate predictions
    preds = {}
    count = 0
    for article in dataset:
        for p in article['paragraphs']:
            context = p['context']
            for qa in p['qas']:
                if count >= args.max_samples:
                    break
                qid = qa['id']
                question = qa['question']
                if args.mode == 'extractive':
                    pred = extractive_qa(question, [context])
                else:
                    pred = generative_qa_openai(question, [context])
                preds[qid] = pred
                count += 1
            if count >= args.max_samples:
                break
        if count >= args.max_samples:
            break

    # save predictions if requested
    if args.preds_out:
        with open(args.preds_out, 'w', encoding='utf-8') as f:
            json.dump(preds, f, indent=2, ensure_ascii=False)
        print(f"Saved predictions to {args.preds_out}")

    # compute metrics
    exact_scores, f1_scores = get_raw_scores(dataset, preds)
    eval_dict = make_eval_dict(exact_scores, f1_scores)
    output = json.dumps(eval_dict, indent=2)

    if args.out_file:
        with open(args.out_file, 'w') as f:
            f.write(output)
        print(f"Saved evaluation to {args.out_file}")
    else:
        print(output)

if __name__ == '__main__':
    main()
