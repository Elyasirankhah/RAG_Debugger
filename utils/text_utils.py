"""
Text utility functions for sentence splitting and text processing.
"""
import re
from typing import List


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple regex-based approach.
    
    Args:
        text: Input text to split
        
    Returns:
        List of sentences
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Split on sentence endings (., !, ?) followed by space or end of string
    # This is a simple approach - for production, consider using spaCy or NLTK
    sentences = re.split(r'([.!?]+(?:\s+|$))', text)
    
    # Recombine sentences with their punctuation
    result = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = (sentences[i] + sentences[i + 1]).strip()
        else:
            sentence = sentences[i].strip()
        
        if sentence:
            result.append(sentence)
    
    # Handle case where last sentence doesn't have punctuation
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1].strip())
    
    return result


def clean_text(text: str) -> str:
    """
    Clean text by removing excessive whitespace and normalizing.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text
