import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from src.qa import extractive_qa
from src.evaluate_squad_old import compute_exact, compute_f1

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SQuADTester:
    def __init__(self, data_file, num_samples=300, confidence_threshold=0.4):
        self.data_file = data_file
        self.num_samples = num_samples
        self.confidence_threshold = confidence_threshold
        self.results = []
        self.output_dir = f"squad_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """Load and sample SQuAD data."""
        print(f"Loading data from {self.data_file}")
        with open(self.data_file, 'r', encoding='utf-8') as f:
            squad = json.load(f)
        
        qas_list = []
        for article in squad['data']:
            for para in article['paragraphs']:
                for qa in para['qas']:
                    qas_list.append((qa, para['context']))
        
        # Sample evenly from answerable and unanswerable
        answerable = [(qa, ctx) for qa, ctx in qas_list if not qa.get('is_impossible', False)]
        unanswerable = [(qa, ctx) for qa, ctx in qas_list if qa.get('is_impossible', False)]
        
        # Try to get balanced sample
        target_answerable = min(self.num_samples // 2, len(answerable))
        target_unanswerable = min(self.num_samples - target_answerable, len(unanswerable))
        
        sampled_answerable = answerable[:target_answerable]
        sampled_unanswerable = unanswerable[:target_unanswerable]
        
        self.test_data = sampled_answerable + sampled_unanswerable
        
        print(f"Dataset successfully loaded with {len(self.test_data)} questions:")
        print(f"  Answerable questions: {len(sampled_answerable)}")
        print(f"  Unanswerable questions: {len(sampled_unanswerable)}")
        print(f"  Balance ratio: {len(sampled_answerable)}/{len(sampled_unanswerable)}")
        
        return self.test_data
    
    def run_evaluation(self):
        """Run the model on all test data."""
        print(f"\nStarting evaluation with RoBERTa-base-squad2 model")
        print(f"Configuration:")
        print(f"  Model: deepset/roberta-base-squad2")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        print(f"  Test questions: {len(self.test_data)}")
        print(f"  Expected runtime: ~{len(self.test_data) * 0.5:.0f} seconds")
        
        start_time = datetime.now()
        
        for i, (qa_item, context) in enumerate(self.test_data):
            if i % 25 == 0:
                elapsed = (datetime.now() - start_time).seconds
                print(f"  Progress: {i+1:3d}/{len(self.test_data)} questions processed ({elapsed:3d}s elapsed)")
            
            qid = qa_item["id"]
            question = qa_item["question"]
            golds = [a["text"] for a in qa_item["answers"] if a["text"].strip()]
            if not golds:
                golds = [""]
            is_impossible = qa_item.get("is_impossible", False)
            
            # Get prediction with confidence
            pred, confidence, is_answerable = extractive_qa(
                question, 
                context, 
                return_confidence=True,
                confidence_threshold=self.confidence_threshold
            )
            
            pred = pred.strip() if pred else ""
            
            # Compute metrics
            em = max(compute_exact(a, pred) for a in golds) if golds else 0
            f1 = max(compute_f1(a, pred) for a in golds) if golds else 0
            
            # Determine correctness
            if is_impossible:
                correct = (pred == "")  # Correct if no answer given
            else:
                correct = (em > 0)  # Correct if exact match > 0
            
            result = {
                'qid': qid,
                'question': question,
                'context': context,
                'gold_answers': golds,
                'predicted': pred,
                'confidence': confidence,
                'is_answerable_pred': is_answerable,
                'is_impossible_true': is_impossible,
                'em': em,
                'f1': f1,
                'correct': correct,
                'question_length': len(question.split()),
                'context_length': len(context.split()),
                'answer_length': len(pred.split()) if pred else 0
            }
            
            self.results.append(result)
        
        total_time = (datetime.now() - start_time).seconds
        print(f"  Evaluation completed in {total_time} seconds")
        print(f"  Average time per question: {total_time/len(self.test_data):.2f} seconds")
        return self.results
    
    def compute_overall_metrics(self):
        """Compute overall performance metrics."""
        total_questions = len(self.results)
        answerable_results = [r for r in self.results if not r['is_impossible_true']]
        unanswerable_results = [r for r in self.results if r['is_impossible_true']]
        
        # Overall metrics
        overall_em = sum(r['em'] for r in self.results) / total_questions * 100
        overall_f1 = sum(r['f1'] for r in self.results) / total_questions * 100
        overall_accuracy = sum(r['correct'] for r in self.results) / total_questions * 100
        
        # Answerable metrics
        if answerable_results:
            answerable_em = sum(r['em'] for r in answerable_results) / len(answerable_results) * 100
            answerable_f1 = sum(r['f1'] for r in answerable_results) / len(answerable_results) * 100
            answerable_accuracy = sum(r['correct'] for r in answerable_results) / len(answerable_results) * 100
        else:
            answerable_em = answerable_f1 = answerable_accuracy = 0
        
        # Unanswerable metrics  
        if unanswerable_results:
            unanswerable_accuracy = sum(r['correct'] for r in unanswerable_results) / len(unanswerable_results) * 100
        else:
            unanswerable_accuracy = 0
        
        metrics = {
            'total_questions': total_questions,
            'answerable_count': len(answerable_results),
            'unanswerable_count': len(unanswerable_results),
            'overall_em': overall_em,
            'overall_f1': overall_f1,
            'overall_accuracy': overall_accuracy,
            'answerable_em': answerable_em,
            'answerable_f1': answerable_f1,
            'answerable_accuracy': answerable_accuracy,
            'unanswerable_accuracy': unanswerable_accuracy,
            'confidence_threshold': self.confidence_threshold
        }
        
        return metrics
    
    def plot_confidence_distributions(self, metrics):
        """Create confidence distribution plots."""
        # Separate results by type and correctness
        correct_answerable = [r['confidence'] for r in self.results 
                            if not r['is_impossible_true'] and r['correct']]
        incorrect_answerable = [r['confidence'] for r in self.results 
                              if not r['is_impossible_true'] and not r['correct']]
        correct_unanswerable = [r['confidence'] for r in self.results 
                              if r['is_impossible_true'] and r['correct']]
        incorrect_unanswerable = [r['confidence'] for r in self.results 
                                if r['is_impossible_true'] and not r['correct']]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Confidence distributions by question type
        ax1.hist([r['confidence'] for r in self.results if not r['is_impossible_true']], 
                bins=20, alpha=0.7, label='Answerable', color='skyblue', density=True)
        ax1.hist([r['confidence'] for r in self.results if r['is_impossible_true']], 
                bins=20, alpha=0.7, label='Unanswerable', color='lightcoral', density=True)
        ax1.axvline(self.confidence_threshold, color='red', linestyle='--', label=f'Threshold ({self.confidence_threshold})')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Density')
        ax1.set_title('Confidence Distribution by Question Type')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Confidence by correctness
        if correct_answerable:
            ax2.hist(correct_answerable, bins=15, alpha=0.7, label='Correct', color='green', density=True)
        if incorrect_answerable:
            ax2.hist(incorrect_answerable, bins=15, alpha=0.7, label='Incorrect', color='red', density=True)
        ax2.axvline(self.confidence_threshold, color='black', linestyle='--', label=f'Threshold')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Density')
        ax2.set_title('Answerable Questions: Confidence by Correctness')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance vs Question Length
        question_lengths = [r['question_length'] for r in self.results]
        correctness = [r['correct'] for r in self.results]
        
        # Bin by question length
        length_bins = range(5, 25, 3)
        avg_performance = []
        bin_centers = []
        
        for i in range(len(length_bins)-1):
            mask = [(l >= length_bins[i] and l < length_bins[i+1]) for l in question_lengths]
            if any(mask):
                perf = np.mean([correctness[j] for j in range(len(mask)) if mask[j]])
                avg_performance.append(perf * 100)
                bin_centers.append((length_bins[i] + length_bins[i+1]) / 2)
        
        if avg_performance:
            ax3.bar(bin_centers, avg_performance, width=2, alpha=0.7, color='purple')
            ax3.set_xlabel('Question Length (words)')
            ax3.set_ylabel('Accuracy (%)')
            ax3.set_title('Accuracy vs Question Length')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: F1 Score Distribution
        f1_scores = [r['f1'] for r in self.results if not r['is_impossible_true']]
        if f1_scores:
            ax4.hist(f1_scores, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax4.axvline(np.mean(f1_scores), color='red', linestyle='--', label=f'Mean F1: {np.mean(f1_scores):.3f}')
            ax4.set_xlabel('F1 Score')
            ax4.set_ylabel('Frequency')
            ax4.set_title('F1 Score Distribution (Answerable Questions)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=2.0)
        plt.suptitle('Confidence and Performance Analysis', fontsize=16, fontweight='bold', y=0.96)
        plt.subplots_adjust(top=0.88, hspace=0.35, wspace=0.25)
        plt.savefig(f'{self.output_dir}/confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_performance_breakdown(self, metrics):
        """Create performance breakdown visualizations."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Overall Performance Bar Chart
        categories = ['Overall\nAccuracy', 'Answerable\nAccuracy', 'Unanswerable\nAccuracy', 'Overall\nF1', 'Overall\nEM']
        values = [metrics['overall_accuracy'], metrics['answerable_accuracy'], 
                 metrics['unanswerable_accuracy'], metrics['overall_f1'], metrics['overall_em']]
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum']
        
        bars = ax1.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Score (%)')
        ax1.set_title('Model Performance Breakdown')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Confusion Matrix for Answerability
        true_answerable = sum(1 for r in self.results if not r['is_impossible_true'])
        true_unanswerable = sum(1 for r in self.results if r['is_impossible_true'])
        
        # Predicted answerable/unanswerable based on whether we gave an answer
        pred_answerable_given_true_answerable = sum(1 for r in self.results 
                                                   if not r['is_impossible_true'] and r['predicted'])
        pred_unanswerable_given_true_answerable = true_answerable - pred_answerable_given_true_answerable
        
        pred_answerable_given_true_unanswerable = sum(1 for r in self.results 
                                                     if r['is_impossible_true'] and r['predicted'])
        pred_unanswerable_given_true_unanswerable = true_unanswerable - pred_answerable_given_true_unanswerable
        
        confusion_matrix = np.array([
            [pred_answerable_given_true_answerable, pred_unanswerable_given_true_answerable],
            [pred_answerable_given_true_unanswerable, pred_unanswerable_given_true_unanswerable]
        ])
        
        im = ax2.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
        ax2.set_title('Answerability Confusion Matrix')
        tick_marks = np.arange(2)
        ax2.set_xticks(tick_marks)
        ax2.set_yticks(tick_marks)
        ax2.set_xticklabels(['Predicted\nAnswerable', 'Predicted\nUnanswerable'])
        ax2.set_yticklabels(['Actually\nAnswerable', 'Actually\nUnanswerable'])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                ax2.text(j, i, str(confusion_matrix[i, j]), ha="center", va="center", 
                        color="white" if confusion_matrix[i, j] > confusion_matrix.max()/2 else "black",
                        fontsize=14, fontweight='bold')
        
        # Plot 3: Answer Length Distribution
        answer_lengths = [r['answer_length'] for r in self.results if r['predicted']]
        if answer_lengths:
            ax3.hist(answer_lengths, bins=range(0, max(answer_lengths)+2), alpha=0.7, color='green', edgecolor='black')
            ax3.set_xlabel('Answer Length (words)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Answer Lengths')
            ax3.axvline(np.mean(answer_lengths), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(answer_lengths):.1f} words')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance vs Confidence Threshold
        thresholds = np.arange(0.1, 0.9, 0.05)
        threshold_f1_scores = []
        threshold_em_scores = []
        
        for thresh in thresholds:
            total_em = 0
            total_f1 = 0
            
            for r in self.results:
                if r['is_impossible_true']:
                    # For unanswerable: correct if confidence < threshold
                    if r['confidence'] < thresh:
                        total_em += 1
                        total_f1 += 1
                else:
                    # For answerable: use actual scores if confidence >= threshold
                    if r['confidence'] >= thresh:
                        total_em += r['em']
                        total_f1 += r['f1']
            
            threshold_em_scores.append(total_em / len(self.results) * 100)
            threshold_f1_scores.append(total_f1 / len(self.results) * 100)
        
        ax4.plot(thresholds, threshold_f1_scores, 'b-', label='F1 Score', linewidth=2)
        ax4.plot(thresholds, threshold_em_scores, 'r-', label='Exact Match', linewidth=2)
        ax4.axvline(self.confidence_threshold, color='green', linestyle='--', 
                   label=f'Current Threshold ({self.confidence_threshold})', linewidth=2)
        ax4.set_xlabel('Confidence Threshold')
        ax4.set_ylabel('Score (%)')
        ax4.set_title('Performance vs Confidence Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=2.0)
        plt.suptitle('Performance Breakdown Analysis', fontsize=16, fontweight='bold', y=0.96)
        plt.subplots_adjust(top=0.88, hspace=0.35, wspace=0.25)
        plt.savefig(f'{self.output_dir}/performance_breakdown.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def analyze_failures(self):
        """Analyze and categorize failure cases."""
        failures = [r for r in self.results if not r['correct']]
        
        # Categorize failures
        false_positives = [r for r in failures if r['is_impossible_true'] and r['predicted']]
        false_negatives = [r for r in failures if not r['is_impossible_true']]
        
        print(f"\n" + "="*80)
        print("DETAILED FAILURE ANALYSIS")
        print("="*80)
        print(f"Total failures: {len(failures)} out of {len(self.results)} questions ({len(failures)/len(self.results)*100:.1f}%)")
        print(f"False positives (incorrectly answered unanswerable): {len(false_positives)}")
        print(f"False negatives (failed to answer answerable): {len(false_negatives)}")
        
        # Analyze false positives
        if false_positives:
            print(f"\nFALSE POSITIVE ANALYSIS:")
            avg_fp_confidence = np.mean([r['confidence'] for r in false_positives])
            print(f"  Average confidence: {avg_fp_confidence:.3f}")
            print(f"  Distribution by confidence ranges:")
            high_conf_fp = len([r for r in false_positives if r['confidence'] > 0.7])
            med_conf_fp = len([r for r in false_positives if 0.4 <= r['confidence'] <= 0.7])
            low_conf_fp = len([r for r in false_positives if r['confidence'] < 0.4])
            print(f"    High confidence (>0.7): {high_conf_fp}")
            print(f"    Medium confidence (0.4-0.7): {med_conf_fp}")
            print(f"    Low confidence (<0.4): {low_conf_fp}")
            
            # Show top 3 most confident false positives
            fp_by_confidence = sorted(false_positives, key=lambda x: x['confidence'], reverse=True)
            print(f"\n  Most confident false positives:")
            for i, r in enumerate(fp_by_confidence[:3]):
                print(f"  {i+1}. Question: {r['question'][:100]}...")
                print(f"     Predicted answer: '{r['predicted']}' (confidence: {r['confidence']:.3f})")
                print(f"     Context snippet: {r['context'][:150]}...")
                print()
        
        # Analyze false negatives
        if false_negatives:
            print(f"FALSE NEGATIVE ANALYSIS:")
            avg_fn_confidence = np.mean([r['confidence'] for r in false_negatives])
            print(f"  Average confidence: {avg_fn_confidence:.3f}")
            print(f"  Distribution by confidence ranges:")
            high_conf_fn = len([r for r in false_negatives if r['confidence'] > 0.7])
            med_conf_fn = len([r for r in false_negatives if 0.4 <= r['confidence'] <= 0.7])
            low_conf_fn = len([r for r in false_negatives if r['confidence'] < 0.4])
            print(f"    High confidence (>0.7): {high_conf_fn}")
            print(f"    Medium confidence (0.4-0.7): {med_conf_fn}")
            print(f"    Low confidence (<0.4): {low_conf_fn}")
            
            # Show top 3 lowest confidence false negatives
            fn_by_confidence = sorted(false_negatives, key=lambda x: x['confidence'])
            print(f"\n  Lowest confidence false negatives:")
            for i, r in enumerate(fn_by_confidence[:3]):
                print(f"  {i+1}. Question: {r['question'][:100]}...")
                print(f"     Gold answer: {r['gold_answers'][0] if r['gold_answers'] else 'N/A'}")
                print(f"     Predicted: '{r['predicted']}' (confidence: {r['confidence']:.3f})")
                print(f"     Context snippet: {r['context'][:150]}...")
                print()
        
        return failures
    
    def generate_report(self, metrics):
        """Generate a comprehensive text report."""
        report_path = f"{self.output_dir}/evaluation_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ROBERTA-BASE-SQUAD2 COMPREHENSIVE EVALUATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: deepset/roberta-base-squad2\n")
            f.write(f"Dataset: SQuAD 2.0\n")
            f.write(f"Test samples: {metrics['total_questions']}\n")
            f.write(f"Confidence threshold: {metrics['confidence_threshold']}\n\n")
            
            f.write("OVERALL PERFORMANCE METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.1f}%\n")
            f.write(f"Overall F1 Score: {metrics['overall_f1']:.1f}%\n")
            f.write(f"Overall Exact Match: {metrics['overall_em']:.1f}%\n\n")
            
            f.write("PERFORMANCE BY QUESTION TYPE\n")
            f.write("-" * 40 + "\n")
            f.write(f"Answerable Questions ({metrics['answerable_count']} total):\n")
            f.write(f"  Accuracy: {metrics['answerable_accuracy']:.1f}%\n")
            f.write(f"  F1 Score: {metrics['answerable_f1']:.1f}%\n")
            f.write(f"  Exact Match: {metrics['answerable_em']:.1f}%\n\n")
            f.write(f"Unanswerable Questions ({metrics['unanswerable_count']} total):\n")
            f.write(f"  Accuracy: {metrics['unanswerable_accuracy']:.1f}%\n\n")
            
            # Confidence analysis
            successful_confidences = [r['confidence'] for r in self.results if r['correct']]
            failed_confidences = [r['confidence'] for r in self.results if not r['correct']]
            
            f.write("CONFIDENCE SCORE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Successful predictions average confidence: {np.mean(successful_confidences):.3f}\n")
            f.write(f"Failed predictions average confidence: {np.mean(failed_confidences):.3f}\n")
            f.write(f"Confidence separation gap: {np.mean(successful_confidences) - np.mean(failed_confidences):.3f}\n\n")
            
            # Answer length analysis
            answer_lengths = [r['answer_length'] for r in self.results if r['predicted']]
            if answer_lengths:
                f.write("ANSWER LENGTH STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Average answer length: {np.mean(answer_lengths):.1f} words\n")
                f.write(f"Median answer length: {np.median(answer_lengths):.1f} words\n")
                f.write(f"Max answer length: {max(answer_lengths)} words\n")
                f.write(f"Single-word answers: {sum(1 for l in answer_lengths if l == 1)} ({sum(1 for l in answer_lengths if l == 1)/len(answer_lengths)*100:.1f}%)\n\n")
            
            f.write("MODEL PERFORMANCE INSIGHTS\n")
            f.write("-" * 40 + "\n")
            if metrics['overall_f1'] >= 75:
                f.write("Performance assessment: EXCELLENT - Model exceeds 75% F1 score target.\n")
            elif metrics['overall_f1'] >= 70:
                f.write("Performance assessment: GOOD - Model achieves solid performance above 70% F1.\n")
            else:
                f.write("Performance assessment: NEEDS IMPROVEMENT - Performance below 70% F1 target.\n")
            
            if metrics['unanswerable_accuracy'] > metrics['answerable_accuracy']:
                f.write("Model strength: Better at identifying unanswerable questions than answering valid ones.\n")
            else:
                f.write("Model strength: Better at answering valid questions than rejecting unanswerable ones.\n")
            
            confidence_gap = np.mean(successful_confidences) - np.mean(failed_confidences)
            if confidence_gap > 0.15:
                f.write("Confidence calibration: EXCELLENT - Strong separation between correct and incorrect predictions.\n")
            elif confidence_gap > 0.1:
                f.write("Confidence calibration: GOOD - Reasonable separation between correct and incorrect predictions.\n")
            else:
                f.write("Confidence calibration: POOR - Limited separation between correct and incorrect predictions.\n")
        
        print(f"\nDetailed report saved to: {report_path}")
        return report_path
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting comprehensive SQuAD 2.0 analysis")
        print("This will generate detailed performance metrics, visualizations, and failure analysis")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Run evaluation
        self.run_evaluation()
        
        # Compute metrics
        metrics = self.compute_overall_metrics()
        
        # Print detailed results
        print(f"\n" + "="*80)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*80)
        print(f"Model evaluated: deepset/roberta-base-squad2")
        print(f"Total questions tested: {metrics['total_questions']}")
        print(f"Confidence threshold used: {metrics['confidence_threshold']}")
        print(f"Evaluation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\nOVERALL PERFORMANCE SUMMARY:")
        print(f"  Overall accuracy: {metrics['overall_accuracy']:.2f}%")
        print(f"  Overall F1 score: {metrics['overall_f1']:.2f}%")
        print(f"  Overall exact match: {metrics['overall_em']:.2f}%")
        
        print(f"\nPERFORMANCE BY QUESTION TYPE:")
        print(f"  Answerable questions ({metrics['answerable_count']} total):")
        print(f"    Accuracy: {metrics['answerable_accuracy']:.2f}%")
        print(f"    F1 score: {metrics['answerable_f1']:.2f}%")
        print(f"    Exact match: {metrics['answerable_em']:.2f}%")
        print(f"  Unanswerable questions ({metrics['unanswerable_count']} total):")
        print(f"    Accuracy: {metrics['unanswerable_accuracy']:.2f}%")
        
        # Confidence analysis
        successful_confidences = [r['confidence'] for r in self.results if r['correct']]
        failed_confidences = [r['confidence'] for r in self.results if not r['correct']]
        print(f"\nCONFIDENCE ANALYSIS:")
        print(f"  Average confidence for correct predictions: {np.mean(successful_confidences):.3f}")
        print(f"  Average confidence for incorrect predictions: {np.mean(failed_confidences):.3f}")
        print(f"  Confidence separation: {np.mean(successful_confidences) - np.mean(failed_confidences):.3f}")
        
        # Generate visualizations
        print(f"\nGenerating analysis visualizations...")
        print("  Creating confidence distribution plots...")
        self.plot_confidence_distributions(metrics)
        print("  Creating performance breakdown charts...")
        self.plot_performance_breakdown(metrics)
        
        # Analyze failures
        self.analyze_failures()
        
        # Generate comprehensive report
        print("\nGenerating comprehensive written report...")
        self.generate_report(metrics)
        
        print(f"\nAnalysis complete!")
        print(f"All results saved to directory: {self.output_dir}")
        print(f"Files generated:")
        print(f"  - confidence_analysis.png (confidence distribution charts)")
        print(f"  - performance_breakdown.png (performance analysis charts)")
        print(f"  - evaluation_report.txt (detailed written report)")
        print("\nThese files are ready for inclusion in your academic report.")
        
        return metrics


# Main execution
if __name__ == "__main__":
    # Configuration parameters
    DATA_FILE = "data/squad/test_set.json"  # Update with your actual file path
    NUM_SAMPLES = 300
    CONFIDENCE_THRESHOLD = 0.4
    
    # Initialize and run analysis
    tester = SQuADTester(DATA_FILE, NUM_SAMPLES, CONFIDENCE_THRESHOLD)
    results = tester.run_full_analysis()