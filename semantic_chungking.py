"""
Semantic Chunking cho văn bản tin tức tiếng Việt
Sử dụng sentence embeddings để phát hiện ranh giới ngữ nghĩa
"""

import numpy as np
from typing import List, Tuple
import re

class SemanticNewsChunker:
    def __init__(
        self,
        model_name: str = "keepitreal/vietnamese-sbert",
        similarity_threshold: float = 0.75,
        min_chunk_size: int = 50,
        max_chunk_size: int = 400,
        window_size: int = 3
    ):
        """
        Args:
            model_name: Tên model sentence transformers (PhoBERT-based)
            similarity_threshold: Ngưỡng similarity để tách chunk (0-1)
            min_chunk_size: Số từ tối thiểu trong một chunk
            max_chunk_size: Số từ tối đa trong một chunk
            window_size: Số câu để tính moving average similarity
        """
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.window_size = window_size

        # Load model
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except:
            print("Cảnh báo: Không thể load model. Sử dụng dummy embeddings.")
            self.model = None

    def split_sentences(self, text: str) -> List[str]:
        """Tách văn bản thành các câu"""
        # Regex tách câu dựa trên dấu câu tiếng Việt
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def get_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Chuyển câu thành vector embeddings"""
        if self.model is None:
            # Dummy embeddings cho testing
            return np.random.rand(len(sentences), 384)
        return self.model.encode(sentences, convert_to_numpy=True)

    def calculate_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """Tính cosine similarity giữa các câu liền kề"""
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
            )
            similarities.append(sim)
        return np.array(similarities)

    def smooth_similarities(self, similarities: np.ndarray) -> np.ndarray:
        """Làm mượt similarity scores bằng moving average"""
        if len(similarities) < self.window_size:
            return similarities

        smoothed = []
        for i in range(len(similarities)):
            start = max(0, i - self.window_size // 2)
            end = min(len(similarities), i + self.window_size // 2 + 1)
            smoothed.append(np.mean(similarities[start:end]))
        return np.array(smoothed)

    def detect_topic_boundaries(
        self,
        sentences: List[str],
        similarities: np.ndarray
    ) -> List[int]:
        """Phát hiện ranh giới chủ đề dựa trên similarity drops"""
        boundaries = [0]  # Bắt đầu từ câu đầu tiên

        # Phát hiện từ khóa chuyển đoạn
        transition_keywords = [
            'tiếp theo', 'bên cạnh đó', 'trong khi đó', 'một tin khác',
            'chuyển sang', 'theo đó', 'ngoài ra', 'mặt khác', 'về vấn đề'
        ]

        for i in range(len(similarities)):
            # Điều kiện 1: Similarity thấp hơn ngưỡng
            if similarities[i] < self.similarity_threshold:
                # Tránh tạo chunk quá nhỏ
                if i > 0 and i - boundaries[-1] >= 3:
                    boundaries.append(i + 1)

            # Điều kiện 2: Phát hiện từ khóa chuyển đoạn
            if i + 1 < len(sentences):
                sentence_lower = sentences[i + 1].lower()
                if any(keyword in sentence_lower for keyword in transition_keywords):
                    if i - boundaries[-1] >= 2:
                        boundaries.append(i + 1)

        return boundaries

    def count_words(self, text: str) -> int:
        """Đếm số từ trong văn bản"""
        return len(text.split())

    def merge_small_chunks(
        self,
        chunks: List[str]
    ) -> List[str]:
        """Gộp các chunk quá nhỏ với chunk kế tiếp"""
        merged = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i]

            # Nếu chunk hiện tại quá nhỏ và không phải chunk cuối
            if self.count_words(current_chunk) < self.min_chunk_size and i < len(chunks) - 1:
                # Gộp với chunk tiếp theo
                current_chunk = current_chunk + " " + chunks[i + 1]
                i += 2
            else:
                i += 1

            merged.append(current_chunk)

        return merged

    def split_large_chunks(
        self,
        chunks: List[str]
    ) -> List[str]:
        """Chia các chunk quá lớn tại điểm ngữ nghĩa tự nhiên"""
        result = []

        for chunk in chunks:
            if self.count_words(chunk) <= self.max_chunk_size:
                result.append(chunk)
            else:
                # Chia chunk lớn thành các sub-chunks
                sentences = self.split_sentences(chunk)
                sub_chunk = []
                current_size = 0

                for sent in sentences:
                    sent_size = self.count_words(sent)

                    if current_size + sent_size > self.max_chunk_size and sub_chunk:
                        result.append(" ".join(sub_chunk))
                        sub_chunk = [sent]
                        current_size = sent_size
                    else:
                        sub_chunk.append(sent)
                        current_size += sent_size

                if sub_chunk:
                    result.append(" ".join(sub_chunk))

        return result

    def chunk(self, text: str, verbose: bool = False) -> List[dict]:
        """
        Chia văn bản thành các chunks có ngữ nghĩa

        Returns:
            List of dicts với keys: 'text', 'start_sentence', 'end_sentence', 'word_count'
        """
        # Bước 1: Tách câu
        sentences = self.split_sentences(text)
        if verbose:
            print(f"Số câu: {len(sentences)}")

        # Bước 2: Tạo embeddings
        embeddings = self.get_embeddings(sentences)
        if verbose:
            print(f"Embeddings shape: {embeddings.shape}")

        # Bước 3: Tính similarity
        similarities = self.calculate_similarities(embeddings)
        smoothed_sims = self.smooth_similarities(similarities)
        if verbose:
            print(f"Similarity trung bình: {np.mean(smoothed_sims):.3f}")
            print(f"Similarity min: {np.min(smoothed_sims):.3f}, max: {np.max(smoothed_sims):.3f}")

        # Bước 4: Phát hiện ranh giới
        boundaries = self.detect_topic_boundaries(sentences, smoothed_sims)
        boundaries.append(len(sentences))  # Thêm điểm kết thúc
        if verbose:
            print(f"Số chunk ban đầu: {len(boundaries) - 1}")
            print(f"Ranh giới: {boundaries}")

        # Bước 5: Tạo chunks
        initial_chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            chunk_text = " ".join(sentences[start:end])
            initial_chunks.append(chunk_text)

        # Bước 6: Merge chunks nhỏ
        merged_chunks = self.merge_small_chunks(initial_chunks)
        if verbose:
            print(f"Sau merge: {len(merged_chunks)} chunks")

        # Bước 7: Split chunks lớn
        final_chunks = self.split_large_chunks(merged_chunks)
        if verbose:
            print(f"Sau split: {len(final_chunks)} chunks")

        # Bước 8: Format kết quả
        results = []
        for i, chunk_text in enumerate(final_chunks):
            results.append({
                'chunk_id': i + 1,
                'text': chunk_text,
                'word_count': self.count_words(chunk_text),
                'sentence_count': len(self.split_sentences(chunk_text))
            })

        return results

