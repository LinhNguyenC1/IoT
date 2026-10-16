"""
Module Quản lý Cơ sở Dữ liệu Cục bộ SQLite (Local Face Database)
Lưu trữ thông tin người dùng và vector đặc trưng 512D ArcFace.
Hỗ trợ so khớp Cosine Similarity siêu tốc bằng ma trận NumPy.
"""

import os
import sqlite3
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict


class LocalDB:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path(__file__).resolve().parent.parent / "faces.db")
        self.db_path = db_path
        self._cache_names = []
        self._cache_vectors = None  # Shape (N, 512)
        self.init_db()
        self.reload_cache()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_db(self):
        """Khởi tạo các bảng users và embeddings nếu chưa tồn tại."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def reload_cache(self):
        """Tải toàn bộ vector từ SQLite lên RAM để so khớp tức thời."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT u.name, e.vector 
                FROM embeddings e 
                JOIN users u ON e.user_id = u.id
            """)
            rows = cursor.fetchall()

        names = []
        vectors = []
        for name, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.shape[0] == 512:
                # Chuẩn hóa L2 norm đề phòng
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    vec = vec / norm
                names.append(name)
                vectors.append(vec)

        self._cache_names = names
        if len(vectors) > 0:
            self._cache_vectors = np.vstack(vectors)  # (N, 512)
        else:
            self._cache_vectors = None

    def add_user(self, name: str, embedding: np.ndarray) -> bool:
        """
        Thêm hoặc cập nhật người dùng kèm vector đặc trưng 512D.
        """
        name = name.strip()
        if not name:
            raise ValueError("Tên người dùng không được để trống!")

        if embedding is None or len(embedding) != 512:
            raise ValueError("Vector đặc trưng phải có kích thước 512 chiều!")

        # Chuẩn hóa L2 norm
        vec = embedding.astype(np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm

        blob = vec.tobytes()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Thêm hoặc lấy id người dùng
            cursor.execute("SELECT id FROM users WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row is None:
                cursor.execute("INSERT INTO users (name) VALUES (?)", (name,))
                user_id = cursor.lastrowid
            else:
                user_id = row[0]

            # Lưu vector mới (hoặc cập nhật)
            cursor.execute("INSERT INTO embeddings (user_id, vector) VALUES (?, ?)", (user_id, blob))
            conn.commit()

        # Làm mới bộ nhớ đệm
        self.reload_cache()
        return True

    def match_face(self, query_vector: np.ndarray, threshold: float = 0.62) -> Tuple[Optional[str], float]:
        """
        So khớp khuôn mặt với cơ sở dữ liệu bằng Cosine Similarity.
        Trả về: (tên_người_hoặc_None, điểm_tương_đồng_tối_đa)
        Nếu điểm tương đồng < threshold -> trả về (None, max_score).
        """
        if self._cache_vectors is None or len(self._cache_names) == 0:
            return None, 0.0

        if query_vector is None or len(query_vector) != 512:
            return None, 0.0

        q = query_vector.astype(np.float32).flatten()
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-6:
            q = q / q_norm

        # Vì cả query và cache đều đã chuẩn hóa L2, Cosine Similarity chính là tích vô hướng (Dot Product)
        # scores: mảng shape (N,)
        scores = np.dot(self._cache_vectors, q)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= threshold:
            return self._cache_names[best_idx], best_score
        else:
            return None, best_score

    def list_users(self) -> List[Dict]:
        """Liệt kê danh sách tất cả người dùng trong database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT u.id, u.name, u.created_at, COUNT(e.id) as sample_count
                FROM users u
                LEFT JOIN embeddings e ON u.id = e.user_id
                GROUP BY u.id
                ORDER BY u.name ASC
            """)
            rows = cursor.fetchall()

        return [
            {"id": r[0], "name": r[1], "created_at": r[2], "samples": r[3]}
            for r in rows
        ]

    def delete_user(self, name: str) -> bool:
        """Xóa người dùng khỏi database theo tên."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE name = ?", (name.strip(),))
            deleted = cursor.rowcount > 0
            conn.commit()

        if deleted:
            self.reload_cache()
        return deleted
