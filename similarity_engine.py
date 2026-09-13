import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FailureModeSimilarityEngine:
    def __init__(self, csv_path):
        self.df = pd.read_csv(
        csv_path,
        encoding="latin-1",  
        )


        self.texts = (
            self.df["Failure Mode"].astype(str) + " " +
            self.df["Effects of Failure"].astype(str) + " " +
            self.df["Potential Cause(s)"].astype(str)
        ).str.lower().str.strip().tolist()

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def find_best_match(self, query_text, top_k=3):
        if not query_text or not query_text.strip():
            return []

        query_text = query_text.lower().strip()
        q_vec = self.vectorizer.transform([query_text])
        scores = cosine_similarity(q_vec, self.matrix)[0]

        ranked_idx = scores.argsort()[::-1]

        results = []
        for idx in ranked_idx[:top_k]:
            if scores[idx] < 0.01:
                break

            row = self.df.iloc[idx].to_dict()
            row["similarity_score"] = float(scores[idx])
            results.append(row)

        return results