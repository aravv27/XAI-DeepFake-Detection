"""
Text Explanation Generator for Deepfake Detection

Generates human-readable explanations of deepfake detection decisions
based on GAT edge importance scores and segment relationships.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from models.face_parser import FACE_SEGMENTS, SEGMENT_COLORS


# Suspicious relationship templates
RELATIONSHIP_TEMPLATES = {
    'boundary': "Blending boundary detected between {seg1} and {seg2}",
    'texture': "Texture inconsistency at {seg1} ↔ {seg2} junction",
    'color': "Color mismatch between {seg1} and {seg2}",
    'noise': "Noise pattern difference at {seg1} and {seg2}",
    'normal': "Natural transition between {seg1} and {seg2}",
}

# Segment importance for deepfake detection
# Based on common manipulation regions
MANIPULATION_HOTSPOTS = {
    'skin': 0.9,       # Most commonly manipulated
    'upper_lip': 0.85,
    'lower_lip': 0.85,
    'mouth': 0.8,
    'nose': 0.75,
    'left_eye': 0.7,
    'right_eye': 0.7,
    'left_brow': 0.6,
    'right_brow': 0.6,
    'hair': 0.5,
    'left_ear': 0.5,
    'right_ear': 0.5,
    'neck': 0.4,
    'cloth': 0.2,
    'background': 0.1,
}


def generate_explanation(
    prediction: str,
    confidence: float,
    edge_importance: Dict[Tuple[str, str], float],
    segment_info: List[Dict],
    top_k_relationships: int = 5
) -> str:
    """
    Generate human-readable explanation for deepfake detection.
    
    Args:
        prediction: 'REAL' or 'FAKE'
        confidence: Confidence score (0-1)
        edge_importance: Dict mapping (segment1, segment2) -> importance score
        segment_info: List of segment info dicts with name, percentage
        top_k_relationships: Number of top relationships to show
    
    Returns:
        Multi-line formatted explanation string
    """
    lines = []
    
    # Header
    lines.append("=" * 60)
    lines.append("DEEPFAKE DETECTION ANALYSIS")
    lines.append("=" * 60)
    lines.append("")
    
    # Classification result
    icon = "⚠️" if prediction == "FAKE" else "✓"
    lines.append(f"{icon} Classification: {prediction} ({confidence*100:.1f}% confidence)")
    lines.append("")
    
    # Detected segments
    lines.append("Detected Facial Regions:")
    for seg in segment_info[:8]:  # Top 8 segments
        lines.append(f"  • {seg['name']}: {seg['percentage']:.1f}% of face")
    lines.append("")
    
    # Key relationships
    if edge_importance:
        lines.append("Key Segment Relationships:")
        lines.append("-" * 40)
        
        # Sort by importance
        sorted_edges = sorted(
            edge_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k_relationships]
        
        for (seg1, seg2), importance in sorted_edges:
            importance_pct = importance * 100
            
            if importance > 0.7:
                status = "🔴 SUSPICIOUS"
                template = RELATIONSHIP_TEMPLATES['boundary']
            elif importance > 0.5:
                status = "🟡 NOTABLE"
                template = RELATIONSHIP_TEMPLATES['texture']
            else:
                status = "🟢 NORMAL"
                template = RELATIONSHIP_TEMPLATES['normal']
            
            explanation = template.format(seg1=seg1, seg2=seg2)
            lines.append(f"  • {seg1} ↔ {seg2}: {importance_pct:.0f}% - {status}")
            lines.append(f"    {explanation}")
        
        lines.append("")
    
    # Interpretation
    lines.append("Interpretation:")
    lines.append("-" * 40)
    
    if prediction == "FAKE":
        # Find most suspicious regions
        suspicious_pairs = [
            (seg1, seg2) for (seg1, seg2), imp in edge_importance.items()
            if imp > 0.6
        ]
        
        if suspicious_pairs:
            seg1, seg2 = suspicious_pairs[0]
            lines.append(
                f"The model detected manipulation artifacts primarily at the "
                f"boundary between {seg1} and {seg2}. This is commonly seen in "
                f"face-swapping techniques where blending imperfections occur."
            )
        else:
            lines.append(
                "The model detected subtle manipulation patterns across "
                "multiple facial regions, suggesting sophisticated editing."
            )
        
        lines.append("")
        lines.append("Common manipulation indicators detected:")
        lines.append("  • Inconsistent texture patterns across region boundaries")
        lines.append("  • Abnormal noise distribution in facial features")
        lines.append("  • Potential blending artifacts at skin-feature junctions")
    else:
        lines.append(
            "The model did not detect significant manipulation artifacts. "
            "Facial region relationships appear consistent and natural."
        )
        lines.append("")
        lines.append("Observations:")
        lines.append("  • Consistent texture across skin regions")
        lines.append("  • Natural lighting and shadow patterns")
        lines.append("  • Coherent noise distribution")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def generate_short_explanation(
    prediction: str,
    confidence: float,
    top_suspicious: Optional[Tuple[str, str, float]] = None
) -> str:
    """
    Generate a short one-line explanation.
    
    Args:
        prediction: 'REAL' or 'FAKE'
        confidence: Confidence score (0-1)
        top_suspicious: Optional (seg1, seg2, importance) for most suspicious pair
    
    Returns:
        Short explanation string
    """
    if prediction == "FAKE" and top_suspicious:
        seg1, seg2, importance = top_suspicious
        return (
            f"FAKE ({confidence*100:.0f}%): Suspicious {seg1}↔{seg2} boundary "
            f"({importance*100:.0f}% anomaly score)"
        )
    elif prediction == "FAKE":
        return f"FAKE ({confidence*100:.0f}%): Manipulation artifacts detected"
    else:
        return f"REAL ({confidence*100:.0f}%): No manipulation detected"


def format_gat_explanation(
    edge_weights: np.ndarray,
    edge_index: np.ndarray,
    segment_names: List[str],
    prediction: str,
    confidence: float
) -> str:
    """
    Format GAT output into explanation.
    
    Args:
        edge_weights: (E,) array of edge importance scores
        edge_index: (2, E) array of edge indices
        segment_names: List of segment names corresponding to node indices
        prediction: 'REAL' or 'FAKE'
        confidence: Confidence score
    
    Returns:
        Formatted explanation string
    """
    # Build edge importance dict
    edge_importance = {}
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src < len(segment_names) and dst < len(segment_names):
            key = (segment_names[src], segment_names[dst])
            edge_importance[key] = float(edge_weights[i])
    
    # Create dummy segment info
    segment_info = [
        {'name': name, 'percentage': 100.0 / len(segment_names)}
        for name in segment_names
    ]
    
    return generate_explanation(
        prediction=prediction,
        confidence=confidence,
        edge_importance=edge_importance,
        segment_info=segment_info
    )


def generate_json_explanation(
    prediction: str,
    confidence: float,
    edge_importance: Dict[Tuple[str, str], float],
    segment_info: List[Dict]
) -> Dict:
    """
    Generate explanation as structured JSON.
    
    Returns dict suitable for API responses or file saving.
    """
    # Top suspicious relationships
    sorted_edges = sorted(
        edge_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    relationships = []
    for (seg1, seg2), importance in sorted_edges:
        relationships.append({
            'segment_1': seg1,
            'segment_2': seg2,
            'importance_score': round(importance, 4),
            'suspicious': importance > 0.6,
            'status': 'suspicious' if importance > 0.7 else (
                'notable' if importance > 0.5 else 'normal'
            )
        })
    
    return {
        'classification': prediction,
        'confidence': round(confidence, 4),
        'is_fake': prediction == 'FAKE',
        'detected_segments': [
            {
                'name': seg['name'],
                'percentage': round(seg['percentage'], 2)
            }
            for seg in segment_info[:10]
        ],
        'key_relationships': relationships,
        'summary': generate_short_explanation(
            prediction, confidence,
            (sorted_edges[0][0][0], sorted_edges[0][0][1], sorted_edges[0][1])
            if sorted_edges else None
        )
    }
