import torch
import clip
import json
import numpy as np
from PIL import Image
import os
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from tqdm import tqdm

class CLIPVerificationPipeline:
    """
    Zero-shot verification pipeline using CLIP model for assessing consistency
    between image regions and their annotated labels without fine-tuning.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize the CLIP verification pipeline.
        
        Args:
            model_name: CLIP model variant to use
            device: Device to run inference on (auto-detected if None)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model: {model_name} on {self.device}")
        
        # Load CLIP model and preprocessing
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        # Stage 1: Text Prompt Construction
        self.class_prompts = {
            "knife": "a photo of a knife",
            "glass": "a photo of a glass",
            "plate": "a photo of a clean plate", 
            "bowl": "a photo of a clean bowl",
            "napkin": "a photo of a napkin",
            "fork": "a photo of a fork",
            "dirty_plate": "a photo of a dirty plate with food residues",
            "dirty_bowl": "a photo of a dirty bowl with food residues",
            "spoon": "a photo of a spoon",
            "cup": "a photo of a cup",
            "bottle": "a photo of a bottle",
            "can": "a photo of a can"
        }
        
        # Stage 2: Extract text embeddings once
        self.text_embeddings = self._extract_text_embeddings()
        self.class_names = list(self.class_prompts.keys())
        
        # Threshold for verification (will be tuned)
        self.similarity_threshold = 0.2
        
    def _extract_text_embeddings(self) -> torch.Tensor:
        """
        Stage 2: Extract and normalize text embeddings for all class prompts.
        
        Returns:
            Normalized text embeddings tensor
        """
        prompts = list(self.class_prompts.values())
        text_tokens = clip.tokenize(prompts).to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_tokens)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            
        return text_embeddings
    
    def _crop_and_preprocess_region(self, image: Image.Image, bbox: Dict) -> torch.Tensor:
        """
        Crop image region based on bounding box and preprocess for CLIP.
        
        Args:
            image: PIL Image
            bbox: Bounding box dictionary with coordinates
            
        Returns:
            Preprocessed image tensor
        """
        # Extract coordinates (assuming format: x1, y1, x2, y2)
        if 'coordinates' in bbox:
            coords = bbox['coordinates']
            x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
        else:
            # Alternative format
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        
        # Crop the region
        cropped_region = image.crop((x1, y1, x2, y2))
        
        # Preprocess for CLIP
        processed_image = self.preprocess(cropped_region).unsqueeze(0).to(self.device)
        
        return processed_image
    
    def _calculate_similarities(self, image_embedding: torch.Tensor) -> torch.Tensor:
        """
        Stage 3: Calculate cosine similarities between image and text embeddings.
        
        Args:
            image_embedding: Normalized image embedding
            
        Returns:
            Similarity scores for all classes
        """
        # Cosine similarity (already normalized embeddings)
        similarities = torch.matmul(image_embedding, self.text_embeddings.T)
        return similarities.squeeze()
    
    def verify_region(self, image: Image.Image, bbox: Dict, true_label: str) -> Dict:
        """
        Verify a single image region against its annotated label.
        
        Args:
            image: PIL Image
            bbox: Bounding box information
            true_label: Ground truth class label
            
        Returns:
            Verification result dictionary
        """
        # Stage 2: Extract image embedding for the cropped region
        processed_region = self._crop_and_preprocess_region(image, bbox)
        
        with torch.no_grad():
            image_embedding = self.model.encode_image(processed_region)
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        
        # Stage 3: Calculate similarities
        similarities = self._calculate_similarities(image_embedding)
        
        # Find predicted class
        max_similarity_idx = torch.argmax(similarities).item()
        predicted_label = self.class_names[max_similarity_idx]
        max_similarity = similarities[max_similarity_idx].item()
        
        # Stage 4: Thresholding and Verification
        is_confident = max_similarity >= self.similarity_threshold
        is_match = predicted_label == true_label
        is_verified = is_confident and is_match
        
        return {
            'predicted_label': predicted_label,
            'true_label': true_label,
            'max_similarity': max_similarity,
            'is_confident': is_confident,
            'is_match': is_match,
            'is_verified': is_verified,
            'all_similarities': {class_name: sim.item() for class_name, sim in zip(self.class_names, similarities)},
            'bbox': bbox
        }
    
    def verify_image(self, image_path: str, annotations: List[Dict]) -> List[Dict]:
        """
        Verify all annotated regions in a single image.
        
        Args:
            image_path: Path to the image file
            annotations: List of annotation dictionaries with bbox and class info
            
        Returns:
            List of verification results for each region
        """
        image = Image.open(image_path).convert('RGB')
        results = []
        
        for annotation in annotations:
            result = self.verify_region(image, annotation['bbox'], annotation['class'])
            result['image_path'] = image_path
            result['annotation_id'] = annotation.get('id', None)
            results.append(result)
            
        return results
    
    def verify_dataset(self, annotations_file: str, images_dir: str) -> List[Dict]:
        """
        Verify an entire dataset.
        
        Args:
            annotations_file: Path to JSON file with annotations
            images_dir: Directory containing images
            
        Returns:
            List of all verification results
        """
        with open(annotations_file, 'r') as f:
            dataset_annotations = json.load(f)
        
        all_results = []
        
        for image_annotation in tqdm(dataset_annotations, desc="Verifying images"):
            image_path = os.path.join(images_dir, image_annotation['image_filename'])
            
            if os.path.exists(image_path):
                results = self.verify_image(image_path, image_annotation['annotations'])
                all_results.extend(results)
            else:
                print(f"Warning: Image not found: {image_path}")
        
        return all_results
    
    def tune_threshold(self, validation_results: List[Dict]) -> float:
        """
        Tune the similarity threshold based on validation results.
        
        Args:
            validation_results: Results from verify_dataset on validation set
            
        Returns:
            Optimal threshold value
        """
        thresholds = np.arange(0.1, 0.8, 0.05)
        best_threshold = 0.2
        best_f1 = 0.0
        
        for threshold in thresholds:
            tp = sum(1 for r in validation_results if r['max_similarity'] >= threshold and r['is_match'])
            fp = sum(1 for r in validation_results if r['max_similarity'] >= threshold and not r['is_match'])
            fn = sum(1 for r in validation_results if r['max_similarity'] < threshold and r['is_match'])
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.similarity_threshold = best_threshold
        print(f"Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.3f})")
        return best_threshold
    
    def generate_report(self, results: List[Dict], output_dir: str = "results"):
        """
        Generate comprehensive verification report.
        
        Args:
            results: List of verification results
            output_dir: Directory to save report files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(results)
        
        # Overall statistics
        total_regions = len(results)
        verified_regions = sum(r['is_verified'] for r in results)
        confident_regions = sum(r['is_confident'] for r in results)
        matching_regions = sum(r['is_match'] for r in results)
        
        print("\n" + "="*50)
        print("CLIP VERIFICATION REPORT")
        print("="*50)
        print(f"Total regions analyzed: {total_regions}")
        print(f"Verified regions (confident + matching): {verified_regions} ({verified_regions/total_regions*100:.1f}%)")
        print(f"Confident predictions: {confident_regions} ({confident_regions/total_regions*100:.1f}%)")
        print(f"Label matches: {matching_regions} ({matching_regions/total_regions*100:.1f}%)")
        print(f"Similarity threshold: {self.similarity_threshold:.3f}")
        
        # Classification report
        y_true = [r['true_label'] for r in results]
        y_pred = [r['predicted_label'] for r in results]
        
        report = classification_report(y_true, y_pred, output_dict=True)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        # Save detailed results
        df.to_csv(os.path.join(output_dir, "verification_results.csv"), index=False)
        
        # Confusion matrix
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_true, y_pred, labels=self.class_names)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Similarity distribution
        plt.figure(figsize=(10, 6))
        similarities = [r['max_similarity'] for r in results]
        plt.hist(similarities, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(self.similarity_threshold, color='red', linestyle='--', 
                   label=f'Threshold: {self.similarity_threshold:.3f}')
        plt.xlabel('Maximum Similarity Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Maximum Similarity Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "similarity_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save summary statistics
        summary = {
            'total_regions': total_regions,
            'verified_regions': verified_regions,
            'verification_rate': verified_regions/total_regions,
            'confident_regions': confident_regions,
            'confidence_rate': confident_regions/total_regions,
            'matching_regions': matching_regions,
            'match_rate': matching_regions/total_regions,
            'threshold': self.similarity_threshold,
            'classification_report': report
        }
        
        with open(os.path.join(output_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {output_dir}/")

def create_sample_annotations(images_dir: str, output_file: str):
    """
    Create sample annotations file for testing the pipeline.
    This simulates the expected input format.
    """
    # Get list of available images
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sample annotations structure
    sample_annotations = []
    
    for i, img_file in enumerate(image_files[:10]):  # Just first 10 images for demo
        # Create mock bounding boxes and labels
        annotations = []
        
        # Add 2-3 random annotations per image
        for j in range(np.random.randint(1, 4)):
            annotation = {
                'id': f"region_{i}_{j}",
                'class': np.random.choice(['knife', 'plate', 'bowl', 'fork', 'glass']),
                'bbox': {
                    'x1': np.random.randint(10, 100),
                    'y1': np.random.randint(10, 100), 
                    'x2': np.random.randint(150, 300),
                    'y2': np.random.randint(150, 300)
                }
            }
            annotations.append(annotation)
        
        sample_annotations.append({
            'image_filename': img_file,
            'annotations': annotations
        })
    
    with open(output_file, 'w') as f:
        json.dump(sample_annotations, f, indent=2)
    
    print(f"Sample annotations created: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="CLIP-based Zero-shot Verification Pipeline")
    parser.add_argument("--mode", choices=['create_sample', 'verify'], default='create_sample',
                       help="Mode: create sample annotations or run verification")
    parser.add_argument("--images_dir", default="data/Images", 
                       help="Directory containing images")
    parser.add_argument("--annotations", default="sample_annotations.json",
                       help="Path to annotations JSON file")
    parser.add_argument("--output_dir", default="results",
                       help="Directory to save results")
    parser.add_argument("--model", default="ViT-B/32",
                       help="CLIP model variant")
    parser.add_argument("--threshold", type=float,
                       help="Similarity threshold (auto-tuned if not provided)")
    
    args = parser.parse_args()
    
    if args.mode == 'create_sample':
        # Create sample annotations for testing
        create_sample_annotations(args.images_dir, args.annotations)
        print(f"\nSample annotations created. Now run:")
        print(f"python judge.py --mode verify --annotations {args.annotations}")
        
    elif args.mode == 'verify':
        # Run verification pipeline
        if not os.path.exists(args.annotations):
            print(f"Error: Annotations file not found: {args.annotations}")
            print("Run with --mode create_sample first to generate sample data")
            return
        
        # Initialize pipeline
        pipeline = CLIPVerificationPipeline(model_name=args.model)
        
        if args.threshold:
            pipeline.similarity_threshold = args.threshold
        
        # Run verification
        print("Running verification pipeline...")
        results = pipeline.verify_dataset(args.annotations, args.images_dir)
        
        # Generate report
        pipeline.generate_report(results, args.output_dir)
        
        print(f"\nVerification complete! Check {args.output_dir}/ for detailed results.")

if __name__ == "__main__":
    main()