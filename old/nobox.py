import torch
import clip
import numpy as np
from PIL import Image
import json
import os
from typing import List, Dict, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLIPVerificationPipeline:
    def __init__(self, device: Optional[str] = None, model_name: str = "ViT-B/32"):
        """
        Initialize the CLIP verification pipeline.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            model_name: CLIP model variant to use
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        logger.info(f"Loaded CLIP model: {model_name}")
        
        # Define semantic classes and their prompts
        self.class_prompts = {
            "knife": "a photo of a knife",
            "glass": "a photo of a glass",
            "plate": "a photo of a plate", 
            "bowl": "a photo of a bowl",
            "napkin": "a photo of a napkin",
            "fork": "a photo of a fork",
            "dirty_plate": "a photo of a dirty plate with food residues",
            "dirty_bowl": "a photo of a dirty bowl with food residues",
            "spoon": "a photo of a spoon",
            "cup": "a photo of a cup",
            "bottle": "a photo of a bottle",
            "food": "a photo of food on a plate"
        }
        
        # Pre-compute text embeddings
        self.text_embeddings = self._encode_text_prompts()
        self.similarity_threshold = 0.2  # Will be tuned empirically
        
    def _encode_text_prompts(self) -> Dict[str, torch.Tensor]:
        """
        Encode all text prompts using CLIP's text encoder.
        
        Returns:
            Dictionary mapping class names to their text embeddings
        """
        text_embeddings = {}
        
        with torch.no_grad():
            for class_name, prompt in self.class_prompts.items():
                text_input = clip.tokenize([prompt]).to(self.device)
                text_embedding = self.model.encode_text(text_input)
                text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                text_embeddings[class_name] = text_embedding
                
        logger.info(f"Encoded {len(text_embeddings)} text prompts")
        return text_embeddings
    
    def _encode_image(self, image_path: str) -> torch.Tensor:
        """
        Encode a single image using CLIP's vision encoder.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized image embedding
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_embedding = self.model.encode_image(image_input)
                image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
                
            return image_embedding
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None
    
    def _calculate_similarities(self, image_embedding: torch.Tensor) -> Dict[str, float]:
        """
        Calculate cosine similarities between image and all text embeddings.
        
        Args:
            image_embedding: Normalized image embedding
            
        Returns:
            Dictionary mapping class names to similarity scores
        """
        similarities = {}
        
        for class_name, text_embedding in self.text_embeddings.items():
            similarity = torch.cosine_similarity(image_embedding, text_embedding, dim=-1)
            similarities[class_name] = similarity.item()
            
        return similarities
    
    def predict_single_image(self, image_path: str) -> Dict:
        """
        Predict all classes present in a single image (multi-label classification).
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing prediction results
        """
        image_embedding = self._encode_image(image_path)
        
        if image_embedding is None:
            return {
                "image_path": image_path,
                "predicted_classes": [],
                "all_similarities": {},
                "above_threshold_classes": [],
                "error": "Failed to encode image"
            }
        
        similarities = self._calculate_similarities(image_embedding)
        
        # Find all classes above threshold
        above_threshold_classes = [
            class_name for class_name, score in similarities.items()
            if score >= self.similarity_threshold
        ]
        
        # Sort by confidence score (descending)
        predicted_classes = sorted(similarities.keys(), key=lambda x: similarities[x], reverse=True)
        
        return {
            "image_path": image_path,
            "predicted_classes": predicted_classes,
            "all_similarities": similarities,
            "above_threshold_classes": above_threshold_classes,
            "max_confidence": max(similarities.values()) if similarities else 0.0
        }
    
    def verify_image_label_consistency(self, image_path: str, true_labels: List[str]) -> Dict:
        """
        Verify consistency between image and its annotated labels (multi-label).
        
        Args:
            image_path: Path to the image file
            true_labels: List of ground truth labels for the image
            
        Returns:
            Dictionary containing verification results
        """
        prediction_result = self.predict_single_image(image_path)
        
        if "error" in prediction_result:
            return {
                **prediction_result,
                "true_labels": true_labels,
                "detected_labels": [],
                "missing_labels": true_labels,
                "extra_labels": [],
                "verification_status": "error",
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
        
        detected_labels = prediction_result["above_threshold_classes"]
        
        # Calculate label-level metrics
        true_labels_set = set(true_labels)
        detected_labels_set = set(detected_labels)
        
        # Find missing and extra labels
        missing_labels = list(true_labels_set - detected_labels_set)
        extra_labels = list(detected_labels_set - true_labels_set)
        correct_labels = list(true_labels_set & detected_labels_set)
        
        # Calculate precision, recall, and F1 score
        precision = len(correct_labels) / len(detected_labels) if detected_labels else 0.0
        recall = len(correct_labels) / len(true_labels) if true_labels else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Determine verification status
        if missing_labels and extra_labels:
            verification_status = "partial_match"
        elif missing_labels:
            verification_status = "missing_labels"
        elif extra_labels:
            verification_status = "extra_labels"
        elif len(correct_labels) == len(true_labels):
            verification_status = "perfect_match"
        else:
            verification_status = "no_match"
        
        return {
            **prediction_result,
            "true_labels": true_labels,
            "detected_labels": detected_labels,
            "correct_labels": correct_labels,
            "missing_labels": missing_labels,
            "extra_labels": extra_labels,
            "verification_status": verification_status,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
    
    def process_image_dataset(self, image_dir: str, annotations_file: str) -> List[Dict]:
        """
        Process a dataset of images with multi-label annotations.
        
        Args:
            image_dir: Directory containing images
            annotations_file: JSON file with image annotations
            
        Returns:
            List of verification results for each image
        """
        # Load annotations
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        results = []
        
        for annotation in annotations:
            image_name = annotation.get("image_name")
            true_labels = annotation.get("labels", [])
            
            # Handle single label format for backward compatibility
            if "label" in annotation and not true_labels:
                true_labels = [annotation["label"]]
            
            if not image_name or not true_labels:
                logger.warning(f"Skipping annotation with missing data: {annotation}")
                continue
                
            image_path = os.path.join(image_dir, image_name)
            
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
            
            result = self.verify_image_label_consistency(image_path, true_labels)
            results.append(result)
            
            logger.info(f"Processed {image_name}: {result['verification_status']} "
                       f"(P:{result['precision']:.2f}, R:{result['recall']:.2f}, F1:{result['f1_score']:.2f})")
        
        return results
    
    def tune_threshold(self, validation_results: List[Dict]) -> float:
        """
        Empirically tune the similarity threshold based on validation results (multi-label).
        
        Args:
            validation_results: List of verification results from validation set
            
        Returns:
            Optimal threshold value
        """
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = self.similarity_threshold
        best_f1_score = 0.0
        
        for threshold in thresholds:
            total_precision = 0.0
            total_recall = 0.0
            total_f1 = 0.0
            valid_samples = 0
            
            for result in validation_results:
                if "error" in result:
                    continue
                    
                # Recalculate metrics with new threshold
                true_labels_set = set(result["true_labels"])
                all_similarities = result["all_similarities"]
                
                # Apply new threshold
                detected_labels = [
                    class_name for class_name, score in all_similarities.items()
                    if score >= threshold
                ]
                detected_labels_set = set(detected_labels)
                
                # Calculate metrics
                correct_labels = list(true_labels_set & detected_labels_set)
                precision = len(correct_labels) / len(detected_labels) if detected_labels else 0.0
                recall = len(correct_labels) / len(true_labels_set) if true_labels_set else 0.0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1_score
                valid_samples += 1
            
            if valid_samples > 0:
                avg_f1 = total_f1 / valid_samples
                if avg_f1 > best_f1_score:
                    best_f1_score = avg_f1
                    best_threshold = threshold
        
        logger.info(f"Best threshold: {best_threshold:.3f} (avg F1: {best_f1_score:.3f})")
        self.similarity_threshold = best_threshold
        return best_threshold
    
    def generate_report(self, results: List[Dict]) -> Dict:
        """
        Generate a summary report of multi-label verification results.
        
        Args:
            results: List of verification results
            
        Returns:
            Summary statistics
        """
        total_images = len(results)
        
        # Count verification statuses
        status_counts = {}
        for result in results:
            status = result.get("verification_status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Calculate average metrics
        total_precision = sum(r.get("precision", 0) for r in results if "error" not in r)
        total_recall = sum(r.get("recall", 0) for r in results if "error" not in r)
        total_f1 = sum(r.get("f1_score", 0) for r in results if "error" not in r)
        valid_results = len([r for r in results if "error" not in r])
        
        avg_precision = total_precision / valid_results if valid_results > 0 else 0
        avg_recall = total_recall / valid_results if valid_results > 0 else 0
        avg_f1 = total_f1 / valid_results if valid_results > 0 else 0
        
        # Calculate per-class statistics
        class_stats = {}
        for result in results:
            if "error" in result:
                continue
                
            for label in result.get("true_labels", []):
                if label not in class_stats:
                    class_stats[label] = {
                        "total_occurrences": 0,
                        "correctly_detected": 0,
                        "missed": 0,
                        "precision_sum": 0,
                        "recall_sum": 0
                    }
                
                class_stats[label]["total_occurrences"] += 1
                
                if label in result.get("detected_labels", []):
                    class_stats[label]["correctly_detected"] += 1
                else:
                    class_stats[label]["missed"] += 1
        
        # Calculate per-class precision and recall
        for class_name, stats in class_stats.items():
            if stats["total_occurrences"] > 0:
                stats["recall"] = stats["correctly_detected"] / stats["total_occurrences"]
            else:
                stats["recall"] = 0.0
        
        # Overall label-level statistics
        total_true_labels = sum(len(r.get("true_labels", [])) for r in results if "error" not in r)
        total_detected_labels = sum(len(r.get("detected_labels", [])) for r in results if "error" not in r)
        total_correct_labels = sum(len(r.get("correct_labels", [])) for r in results if "error" not in r)
        
        return {
            "total_images": total_images,
            "valid_images": valid_results,
            "error_images": status_counts.get("error", 0),
            "verification_status_counts": status_counts,
            "average_metrics": {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1_score": avg_f1
            },
            "label_statistics": {
                "total_true_labels": total_true_labels,
                "total_detected_labels": total_detected_labels,
                "total_correct_labels": total_correct_labels,
                "overall_precision": total_correct_labels / total_detected_labels if total_detected_labels > 0 else 0,
                "overall_recall": total_correct_labels / total_true_labels if total_true_labels > 0 else 0
            },
            "threshold_used": self.similarity_threshold,
            "per_class_statistics": class_stats
        }

# Example usage
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = CLIPVerificationPipeline()
    
    # Example 1: Verify a single image with multiple labels
    true_labels = ["knife", "glass", "plate", "bowl", "napkin", "fork"]
    result = pipeline.verify_image_label_consistency("data/Images/img__00001_.png", true_labels)
    print(f"Verification result: {result}")
    
    # Example 2: Process a dataset with multi-label annotations
    # JSON format should be:
    # [
    #   {
    #     "image_name": "table1.jpg", 
    #     "labels": ["knife", "glass", "plate", "bowl", "napkin", "fork", "dirty_plate"]
    #   },
    #   {
    #     "image_name": "table2.jpg",
    #     "labels": ["knife", "glass", "plate", "bowl", "napkin", "fork", "dirty_bowl"]
    #   }
    # ]
    
    results = pipeline.process_image_dataset("data/Images", "data/annotations.json")
    
    # Tune threshold on validation set (if you have one)
    # validation_results = pipeline.process_image_dataset("path/to/validation/", "validation_annotations.json")
    # optimal_threshold = pipeline.tune_threshold(validation_results)
    # print(f"Optimal threshold: {optimal_threshold}")
    
    # Generate comprehensive report
    report = pipeline.generate_report(results)
    print(f"Verification report: {report}")
    
    # Example of what the report contains:
    print(f"Average F1 Score: {report['average_metrics']['f1_score']:.3f}")
    print(f"Overall Precision: {report['label_statistics']['overall_precision']:.3f}")
    print(f"Overall Recall: {report['label_statistics']['overall_recall']:.3f}")
    print(f"Perfect matches: {report['verification_status_counts'].get('perfect_match', 0)}")
    print(f"Partial matches: {report['verification_status_counts'].get('partial_match', 0)}")
    
    # Save results
    with open("verification_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open("verification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Example: Print detailed results for first few images
    print("\nDetailed results for first 3 images:")
    for i, result in enumerate(results[:3]):
        print(f"\nImage {i+1}: {result['image_path']}")
        print(f"  True labels: {result['true_labels']}")
        print(f"  Detected labels: {result['detected_labels']}")
        print(f"  Missing labels: {result['missing_labels']}")
        print(f"  Extra labels: {result['extra_labels']}")
        print(f"  Status: {result['verification_status']}")
        print(f"  Precision: {result['precision']:.3f}")
        print(f"  Recall: {result['recall']:.3f}")
        print(f"  F1 Score: {result['f1_score']:.3f}")
        
        # Show top similarities
        print(f"  Top similarities:")
        sorted_similarities = sorted(result['all_similarities'].items(), key=lambda x: x[1], reverse=True)
        for class_name, score in sorted_similarities[:5]:
            status = "✓" if score >= pipeline.similarity_threshold else "✗"
            print(f"    {status} {class_name}: {score:.3f}")