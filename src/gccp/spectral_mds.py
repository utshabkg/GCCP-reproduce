"""
Spectral-based Multi-Document Summarization for Anchor Generation

Implements the unsupervised extractive MDS approach:
1. Sentence graph construction with TF-IDF embeddings
2. Affinity matrix with threshold
3. Normalized Laplacian computation  
4. Fiedler vector extraction for spectral clustering
5. Anchor document generation from larger cluster
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional
import re


def segment_sentences(text: str) -> List[str]:
    """
    Simple sentence segmentation.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Simple regex-based sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter empty and very short sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences


def segment_sentences_spacy(text: str, nlp=None) -> List[str]:
    """
    Sentence segmentation using spaCy.
    
    Args:
        text: Input text
        nlp: Optional spaCy model (loads en_core_web_sm if None)
        
    Returns:
        List of sentences
    """
    if nlp is None:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Download if not available
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
    
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
    return sentences


class SpectralMDS:
    """
    Spectral-based Multi-Document Summarization.
    
    Implements the anchor document generation algorithm from Section 3.2.2
    of the GCCP paper.
    """
    
    def __init__(self, threshold: float = 0.2, use_spacy: bool = True):
        """
        Initialize the MDS module.
        
        Args:
            threshold: Cosine similarity threshold θ for affinity matrix (Eq. 5)
            use_spacy: Whether to use spaCy for sentence segmentation
        """
        self.threshold = threshold
        self.use_spacy = use_spacy
        self.nlp = None
        
        if use_spacy:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Downloading spaCy model...")
                from spacy.cli import download
                download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
    
    def extract_sentences(self, documents: List[str]) -> Tuple[List[str], List[int]]:
        """
        Extract sentences from multiple documents.
        
        Args:
            documents: List of document texts
            
        Returns:
            Tuple of (sentences, document_indices) where document_indices[i]
            indicates which document sentence i came from
        """
        all_sentences = []
        doc_indices = []
        
        for doc_idx, doc in enumerate(documents):
            if self.use_spacy and self.nlp:
                sentences = segment_sentences_spacy(doc, self.nlp)
            else:
                sentences = segment_sentences(doc)
            
            all_sentences.extend(sentences)
            doc_indices.extend([doc_idx] * len(sentences))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sentences = []
        unique_indices = []
        
        for sent, idx in zip(all_sentences, doc_indices):
            sent_lower = sent.lower().strip()
            if sent_lower not in seen:
                seen.add(sent_lower)
                unique_sentences.append(sent)
                unique_indices.append(idx)
        
        return unique_sentences, unique_indices
    
    def build_affinity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Build the affinity matrix A (Eq. 5).
        
        a_{i,j} = cos(e_i, e_j) if cos(e_i, e_j) >= θ else 0
        
        Args:
            sentences: List of sentences
            
        Returns:
            Affinity matrix A ∈ R^{n×n}
        """
        # Compute TF-IDF embeddings
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Compute cosine similarity
        similarity = cosine_similarity(tfidf_matrix)
        
        # Apply threshold (Eq. 5)
        affinity = np.where(similarity >= self.threshold, similarity, 0)
        
        # Set diagonal to 0 (no self-loops)
        np.fill_diagonal(affinity, 0)
        
        return affinity
    
    def compute_fiedler_vector(self, affinity: np.ndarray) -> np.ndarray:
        """
        Compute the Fiedler vector v_2 (Eq. 7-8).
        
        The Fiedler vector is the eigenvector corresponding to the
        second smallest eigenvalue of the normalized Laplacian L.
        
        Args:
            affinity: Affinity matrix A
            
        Returns:
            Fiedler vector v_2
        """
        n = affinity.shape[0]
        
        if n <= 2:
            # Not enough sentences for meaningful clustering
            return np.ones(n)
        
        # Compute degree matrix D (Eq. 6)
        degrees = affinity.sum(axis=1)
        
        # Handle zero degrees
        degrees = np.maximum(degrees, 1e-10)
        
        # D^{-1/2}
        d_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        
        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2} (Eq. 6)
        normalized_affinity = d_inv_sqrt @ affinity @ d_inv_sqrt
        laplacian = np.eye(n) - normalized_affinity
        
        # Make symmetric (numerical stability)
        laplacian = (laplacian + laplacian.T) / 2
        
        try:
            # Get second smallest eigenvector (Eq. 7)
            # We want eigenvectors for smallest eigenvalues
            eigenvalues, eigenvectors = eigsh(
                sparse.csr_matrix(laplacian), 
                k=min(3, n-1), 
                which='SM',  # Smallest magnitude
                tol=1e-6
            )
            
            # Sort by eigenvalue
            idx = np.argsort(eigenvalues)
            
            # Fiedler vector is the second eigenvector (first is constant)
            if len(idx) >= 2:
                fiedler = eigenvectors[:, idx[1]]
            else:
                fiedler = eigenvectors[:, idx[0]]
                
        except Exception as e:
            print(f"Warning: Eigenvalue computation failed ({e}), using random partition")
            fiedler = np.random.randn(n)
        
        return fiedler
    
    def generate_anchor(self, documents: List[str], num_sentences: int = 10) -> str:
        """
        Generate anchor document using spectral clustering (Eq. 9).
        
        Args:
            documents: List of top-m document texts
            num_sentences: Number of sentences z in anchor
            
        Returns:
            Anchor document text
        """
        # Step 1: Extract and deduplicate sentences
        sentences, doc_indices = self.extract_sentences(documents)
        
        if len(sentences) == 0:
            return ""
        
        if len(sentences) <= num_sentences:
            return " ".join(sentences)
        
        # Step 2: Build affinity matrix (Eq. 5)
        affinity = self.build_affinity_matrix(sentences)
        
        # Step 3: Compute Fiedler vector (Eq. 7)
        fiedler = self.compute_fiedler_vector(affinity)
        
        # Step 4: Partition sentences by sign of Fiedler components
        positive_cluster = np.where(fiedler >= 0)[0]
        negative_cluster = np.where(fiedler < 0)[0]
        
        # Select larger cluster (more common themes)
        if len(positive_cluster) >= len(negative_cluster):
            selected_indices = positive_cluster
        else:
            selected_indices = negative_cluster
        
        # Step 5: Order by original position (Eq. 9)
        # Create (sentence_idx, doc_idx, position_in_doc) tuples
        selected_with_positions = []
        doc_sentence_counts = {}
        
        for sent_idx in selected_indices:
            doc_idx = doc_indices[sent_idx]
            if doc_idx not in doc_sentence_counts:
                doc_sentence_counts[doc_idx] = 0
            pos = doc_sentence_counts[doc_idx]
            doc_sentence_counts[doc_idx] += 1
            selected_with_positions.append((sent_idx, doc_idx, pos))
        
        # Sort by document index, then position
        selected_with_positions.sort(key=lambda x: (x[1], x[2]))
        
        # Select top z sentences (Eq. 9)
        anchor_indices = [x[0] for x in selected_with_positions[:num_sentences]]
        anchor_sentences = [sentences[i] for i in anchor_indices]
        
        return " ".join(anchor_sentences)


def generate_anchor_document(documents: List[str], 
                            m: int = 10, 
                            z: int = 10,
                            threshold: float = 0.2,
                            use_spacy: bool = True) -> str:
    """
    Convenience function to generate anchor document.
    
    Args:
        documents: List of candidate documents (should be sorted by initial ranking)
        m: Number of top documents to use
        z: Number of sentences in anchor
        threshold: Similarity threshold
        use_spacy: Use spaCy for sentence segmentation
        
    Returns:
        Anchor document text
    """
    # Use top-m documents
    top_docs = documents[:m]
    
    if not top_docs:
        return ""
    
    # Extract document contents
    if isinstance(top_docs[0], dict):
        doc_texts = [d.get('contents', d.get('text', '')) for d in top_docs]
    else:
        doc_texts = top_docs
    
    # Generate anchor
    mds = SpectralMDS(threshold=threshold, use_spacy=use_spacy)
    anchor = mds.generate_anchor(doc_texts, num_sentences=z)
    
    return anchor


if __name__ == "__main__":
    # Test MDS
    documents = [
        "Paris is the capital of France. It is known for the Eiffel Tower. The city has many museums.",
        "France is a country in Western Europe. Paris is its capital city. French cuisine is world-famous.",
        "The Louvre Museum is located in Paris. It houses the Mona Lisa. Millions visit annually.",
    ]
    
    mds = SpectralMDS(threshold=0.1, use_spacy=False)
    
    print("Extracting sentences...")
    sentences, indices = mds.extract_sentences(documents)
    print(f"Found {len(sentences)} unique sentences")
    
    print("\nBuilding affinity matrix...")
    affinity = mds.build_affinity_matrix(sentences)
    print(f"Affinity matrix shape: {affinity.shape}")
    
    print("\nGenerating anchor document...")
    anchor = mds.generate_anchor(documents, num_sentences=3)
    print(f"Anchor: {anchor}")
