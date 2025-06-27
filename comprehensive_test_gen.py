"""
Clean, report-ready Generative QA evaluation script with publication-quality visualizations.
Perfect for screenshots and academic reports.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

# Import your generative QA function from qa.py
from src.qa import generative_qa_groq  

# Set publication-quality plotting style
plt.style.use('default')
sns.set_palette("colorblind")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'sans-serif'
})

def normalize_answer(s: str) -> str:
    """Normalize answer for evaluation."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact(a_gold: str, a_pred: str) -> bool:
    """Compute exact match with fix for empty answers."""
    # Normalize first
    a_gold = a_gold.strip()
    a_pred = a_pred.strip()
    
    # CRITICAL FIX: Both empty should be exact match
    if not a_gold and not a_pred:
        return True
    
    return normalize_answer(a_gold) == normalize_answer(a_pred)

def compute_f1(a_gold: str, a_pred: str) -> float:
    """Compute F1 score with fix for empty answers."""
    # Normalize first
    a_gold = a_gold.strip()
    a_pred = a_pred.strip()
    
    # CRITICAL FIX: Both empty should be perfect F1
    if not a_gold and not a_pred:
        return 1.0
    
    # If only one is empty, F1 should be 0
    if not a_gold or not a_pred:
        return 0.0
    
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = set(gold_toks) & set(pred_toks)
    
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(pred_toks) if pred_toks else 0.0
    recall = len(common) / len(gold_toks) if gold_toks else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)

def load_and_sample_data(data_file, num_samples=100):
    """Load balanced sample of SQuAD data."""
    with open(data_file, 'r', encoding='utf-8') as f:
        squad = json.load(f)
    
    qas_list = []
    for article in squad['data']:
        for para in article['paragraphs']:
            for qa in para['qas']:
                qas_list.append((qa, para['context']))
    
    # Separate by type for balanced sampling
    answerable = [(qa, ctx) for qa, ctx in qas_list if not qa.get('is_impossible', False)]
    unanswerable = [(qa, ctx) for qa, ctx in qas_list if qa.get('is_impossible', False)]
    
    # Take balanced sample
    n_answerable = min(num_samples // 2, len(answerable))
    n_unanswerable = min(num_samples - n_answerable, len(unanswerable))
    
    sample = answerable[:n_answerable] + unanswerable[:n_unanswerable]
    
    print(f"Dataset Summary:")
    print(f"   Total Questions: {len(sample)}")
    print(f"   Answerable: {n_answerable}")
    print(f"   Unanswerable: {n_unanswerable}")
    
    return sample

def evaluate_model(test_data):
    """Evaluate generative model on test data."""
    results = []
    
    print(f"\nEvaluating GPT-3.5-turbo Generative QA...")
    
    for i, (qa_item, context) in enumerate(test_data):
        if (i + 1) % 25 == 0:
            print(f"   Progress: {i+1}/{len(test_data)}")
        
        question = qa_item["question"]
        golds = [a["text"] for a in qa_item["answers"] if a["text"].strip()]
        if not golds:
            golds = [""]
        is_impossible = qa_item.get("is_impossible", False)
        
        # Get generative prediction
        try:
            pred = generative_qa_openai_optimized(question, context)
            pred = pred.strip() if pred else ""
        except Exception as e:
            print(f"   Error with question {i+1}: {e}")
            pred = ""
        
        # Compute metrics using fixed functions
        if golds:
            em = max(compute_exact(gold, pred) for gold in golds)
            f1 = max(compute_f1(gold, pred) for gold in golds)
        else:
            em = 1 if not pred else 0
            f1 = 1.0 if not pred else 0.0
        
        # Determine correctness based on question type
        if is_impossible:
            correct = (pred == "")  # Correct if no answer provided for impossible question
        else:
            correct = (em > 0)      # Correct if exact match for answerable question
        
        results.append({
            'question': question,
            'predicted': pred,
            'gold_answers': golds,
            'is_impossible': is_impossible,
            'em': em,
            'f1': f1,
            'correct': correct
        })
    
    return results

def compute_metrics(results):
    """Compute comprehensive metrics."""
    answerable = [r for r in results if not r['is_impossible']]
    unanswerable = [r for r in results if r['is_impossible']]
    
    metrics = {
        'total_questions': len(results),
        'answerable_count': len(answerable),
        'unanswerable_count': len(unanswerable),
        
        # Overall metrics
        'overall_accuracy': np.mean([r['correct'] for r in results]) * 100,
        'overall_f1': np.mean([r['f1'] for r in results]) * 100,
        'overall_em': np.mean([r['em'] for r in results]) * 100,
        
        # Answerable metrics
        'answerable_accuracy': np.mean([r['correct'] for r in answerable]) * 100 if answerable else 0,
        'answerable_f1': np.mean([r['f1'] for r in answerable]) * 100 if answerable else 0,
        'answerable_em': np.mean([r['em'] for r in answerable]) * 100 if answerable else 0,
        
        # Unanswerable metrics
        'unanswerable_accuracy': np.mean([r['correct'] for r in unanswerable]) * 100 if unanswerable else 0,
        
        # Answer behavior analysis
        'total_non_empty': sum(1 for r in results if r['predicted']),
        'answerable_non_empty': sum(1 for r in answerable if r['predicted']),
        'unanswerable_non_empty': sum(1 for r in unanswerable if r['predicted']),
    }
    
    return metrics

def create_main_results_plot(metrics):
    """Create the main results visualization for the report."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Overall Performance Bar Chart
    categories = ['Overall\nAccuracy', 'Overall\nF1 Score', 'Overall\nExact Match']
    values = [metrics['overall_accuracy'], metrics['overall_f1'], metrics['overall_em']]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    bars1 = ax1.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Score (%)', fontweight='bold')
    ax1.set_title('Overall Model Performance', fontweight='bold', pad=20)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars1, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Plot 2: Performance by Question Type
    question_types = ['Answerable\nQuestions', 'Unanswerable\nQuestions']
    answerable_acc = metrics['answerable_accuracy']
    unanswerable_acc = metrics['unanswerable_accuracy']
    accuracies = [answerable_acc, unanswerable_acc]
    colors2 = ['#4CAF50', '#FF6B6B']
    
    bars2 = ax2.bar(question_types, accuracies, color=colors2, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.set_title('Performance by Question Type', fontweight='bold', pad=20)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels and counts
    counts = [metrics['answerable_count'], metrics['unanswerable_count']]
    for bar, acc, count in zip(bars2, accuracies, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax2.text(bar.get_x() + bar.get_width()/2, 5, 
                f'n={count}', ha='center', va='bottom', fontsize=9, alpha=0.7)
    
    # Plot 3: Detailed Answerable Question Metrics
    ans_categories = ['Accuracy', 'F1 Score', 'Exact Match']
    ans_values = [metrics['answerable_accuracy'], metrics['answerable_f1'], metrics['answerable_em']]
    colors3 = ['#FF9800', '#9C27B0', '#607D8B']
    
    bars3 = ax3.bar(ans_categories, ans_values, color=colors3, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Score (%)', fontweight='bold')
    ax3.set_title('Answerable Questions - Detailed Metrics', fontweight='bold', pad=20)
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars3, ans_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Plot 4: Answer Behavior Analysis
    behavior_categories = ['Answerable\nAnswered', 'Unanswerable\nAnswered']
    ans_total = metrics['answerable_count']
    unans_total = metrics['unanswerable_count']
    behavior_values = [
        (metrics['answerable_non_empty'] / ans_total * 100) if ans_total > 0 else 0,
        (metrics['unanswerable_non_empty'] / unans_total * 100) if unans_total > 0 else 0
    ]
    colors4 = ['#4CAF50', '#F44336']
    
    bars4 = ax4.bar(behavior_categories, behavior_values, color=colors4, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Percentage Answered (%)', fontweight='bold')
    ax4.set_title('Answer Behavior Analysis', fontweight='bold', pad=20)
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars4, behavior_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout(pad=3.0)
    plt.suptitle('GPT-3.5-turbo Generative QA Evaluation Results', fontsize=18, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.93)
    
    return fig

def print_executive_summary(metrics):
    """Print clean executive summary for the report."""
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY - GPT-3.5-turbo Generative QA on SQuAD 2.0")
    print("="*80)
    
    print(f"\nDATASET:")
    print(f"   Total Questions Evaluated: {metrics['total_questions']}")
    print(f"   Answerable Questions: {metrics['answerable_count']}")
    print(f"   Unanswerable Questions: {metrics['unanswerable_count']}")
    
    print(f"\nKEY PERFORMANCE METRICS:")
    print(f"   Overall Accuracy: {metrics['overall_accuracy']:.1f}%")
    print(f"   Overall F1 Score: {metrics['overall_f1']:.1f}%")
    print(f"   Overall Exact Match: {metrics['overall_em']:.1f}%")
    
    print(f"\nDETAILED BREAKDOWN:")
    print(f"   Answerable Questions:")
    print(f"     Accuracy: {metrics['answerable_accuracy']:.1f}%")
    print(f"     F1 Score: {metrics['answerable_f1']:.1f}%")
    print(f"     Exact Match: {metrics['answerable_em']:.1f}%")
    print(f"   Unanswerable Questions:")
    print(f"     Accuracy: {metrics['unanswerable_accuracy']:.1f}%")
    
    print(f"\nMODEL BEHAVIOR ANALYSIS:")
    ans_answer_rate = (metrics['answerable_non_empty'] / metrics['answerable_count'] * 100) if metrics['answerable_count'] > 0 else 0
    unans_answer_rate = (metrics['unanswerable_non_empty'] / metrics['unanswerable_count'] * 100) if metrics['unanswerable_count'] > 0 else 0
    
    print(f"   Answerable Questions Answered: {ans_answer_rate:.1f}%")
    print(f"   Unanswerable Questions Answered: {unans_answer_rate:.1f}%")
    
    if metrics['overall_f1'] >= 70:
        print(f"   Performance Status: STRONG (F1 >= 70%)")
    elif metrics['overall_f1'] >= 60:
        print(f"   Performance Status: MODERATE (F1 >= 60%)")
    else:
        print(f"   Performance Status: NEEDS IMPROVEMENT (F1 < 60%)")
    
    if metrics['unanswerable_accuracy'] > metrics['answerable_accuracy']:
        print(f"   Primary Strength: Better at rejecting unanswerable questions")
    else:
        print(f"   Primary Strength: Better at answering valid questions")
    
    print("\n" + "="*80)

def print_detailed_error_analysis(results):
    """Print detailed error analysis."""
    print("\nDETAILED ERROR ANALYSIS")
    print("="*60)
    
    # Categorize errors
    wrong_predictions = [r for r in results if not r['correct']]
    empty_predictions = [r for r in results if not r['predicted']]
    wrong_non_empty = [r for r in wrong_predictions if r['predicted']]
    
    print(f"\nFAILURE BREAKDOWN:")
    print(f"   Total failures: {len(wrong_predictions)}/{len(results)} ({len(wrong_predictions)/len(results):.1%})")
    print(f"   Empty answers: {len(empty_predictions)}/{len(results)} ({len(empty_predictions)/len(results):.1%})")
    print(f"   Wrong answers: {len(wrong_non_empty)}/{len(results)} ({len(wrong_non_empty)/len(results):.1%})")
    
    # Show examples of wrong non-empty answers
    if wrong_non_empty:
        print(f"\nSAMPLE WRONG ANSWERS:")
        for i, r in enumerate(wrong_non_empty[:5]):  # Show first 5
            print(f"{i+1}. Q: {r['question']}")
            print(f"   Gold: {r['gold_answers'][0] if r['gold_answers'] else ''}")
            print(f"   Predicted: '{r['predicted']}'")
            print()

def main():
    """Main execution function."""
    # Configuration
    DATA_FILE = "data/squad/test_set.json"  # Update path as needed
    NUM_SAMPLES = 100
    
    print("GPT-3.5-turbo Generative QA Evaluation for Academic Report")
    print("="*60)
    
    # Load data
    test_data = load_and_sample_data(DATA_FILE, NUM_SAMPLES)
    
    # Evaluate model
    results = evaluate_model(test_data)
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    # Print executive summary
    print_executive_summary(metrics)
    
    # Print detailed error analysis
    print_detailed_error_analysis(results)
    
    # Create visualization
    print("\nGenerating publication-quality visualization...")
    fig = create_main_results_plot(metrics)
    
    # Save high-quality plot
    plt.savefig('generative_qa_evaluation_results.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("High-resolution plot saved as 'generative_qa_evaluation_results.png'")
    
    plt.show()
    
    # Final scores summary
    print(f"\nFINAL SCORES")
    print("="*60)
    print(f"Overall Exact Match (EM): {metrics['overall_em']:.1f}%")
    print(f"Overall F1 Score: {metrics['overall_f1']:.1f}%")
    print("="*60)
    
    return metrics, results

if __name__ == "__main__":
    metrics, results = main()