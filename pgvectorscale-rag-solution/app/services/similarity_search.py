from app.managers.embedding_manager import create_embeddings  # Use the correct function
from sklearn.metrics.pairwise import cosine_similarity
from app.database.vector_store import VectorStore
from typing import Optional, List
import uuid
import pandas as pd

class SimilaritySearchEngine:
    def __init__(self):
        """
        Initialize the similarity search engine with a vector store.
        """
        self.vector_store = VectorStore()

    def fetch_all(
        self, metadata_filter: Optional[dict] = None, time_range: Optional[tuple] = None
    ) -> List[dict]:
        """
        Fetch all records from the vector store with optional filtering.

        Args:
            metadata_filter (dict, optional): Metadata filtering criteria.
            time_range (tuple, optional): Time range for filtering (start_time, end_time).

        Returns:
            List of dictionaries containing record information.
        """
        results = self.vector_store.search(
            query_text="",  # Empty query text to fetch all embeddings
            limit=1000,  # Large limit to fetch most records
            metadata_filter=metadata_filter,
            time_range=time_range,
            return_dataframe=False,
        )

        if not results:
            raise ValueError("No records found in the vector store.")

        return [
            {
                "id": str(result[0]) if isinstance(result[0], uuid.UUID) else result[0],
                "embedding": result[3],  # Assuming embedding is the 4th element
                "metadata": self._convert_metadata_to_serializable(result[1]),
                "content": result[2]  # Assuming content is the 3rd element
            }
            for result in results
        ]

    def search(
        self,
        query: str,
        limit: int = 5,
        metadata_filter: Optional[dict] = None,
        time_range: Optional[tuple] = None,
    ) -> pd.DataFrame:
        """
        Perform advanced similarity search.

        Args:
            query (str): Search query text.
            limit (int): Number of results to return.
            metadata_filter (dict, optional): Metadata filtering criteria.
            time_range (tuple, optional): Time range for filtering (start_time, end_time).

        Returns:
            pd.DataFrame: DataFrame containing search results with similarity scores.
        """
        try:
            query_embedding = create_embeddings([{"page_content": query}])[0]
        except (IndexError, KeyError):
            raise ValueError("Failed to generate embedding for the query. Ensure the input format is correct.")

        records = self.fetch_all(metadata_filter=metadata_filter, time_range=time_range)
        if not records:
            raise ValueError("No embeddings found in the vector store matching the criteria.")

        embeddings = [record["embedding"] for record in records]
        metadata = [record["metadata"] for record in records]
        ids = [record.get("id", "Unknown") for record in records]

        if not embeddings:
            raise ValueError("No valid embeddings available for similarity search.")

        similarities = cosine_similarity([query_embedding], embeddings).flatten()

        results_df = pd.DataFrame(metadata)
        results_df["similarity"] = similarities
        results_df["id"] = ids

        return results_df.sort_values(by="similarity", ascending=False).head(limit)

    @staticmethod
    def _convert_metadata_to_serializable(metadata: dict) -> dict:
        """
        Convert metadata dictionary to a JSON serializable format.

        Args:
            metadata (dict): Metadata dictionary.

        Returns:
            dict: JSON serializable metadata dictionary.
        """
        serializable_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, uuid.UUID):
                serializable_metadata[key] = str(value)
            else:
                serializable_metadata[key] = value
        return serializable_metadata
