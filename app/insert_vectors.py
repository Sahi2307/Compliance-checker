# %%

from datetime import datetime
from pathlib import Path
import pandas as pd
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time

# Initialize VectorStore
vec = VectorStore()

# Read the CSV file (repo-relative path)
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "final.csv"
df = pd.read_csv(DATA_PATH, sep=",")
df.head()
# Prepare data for insertion
import numpy as np

# Replace NaN values with None
df = df.replace({np.nan: None})
def prepare_record(row):
    # Combine relevant columns for embeddings
    content = f"""
    Filename: {row['Filename']}
    Document Name: {row['Document Name']}
    Document Name-Answer: {row['Document Name-Answer']}
    Parties-Answer: {row['Parties-Answer']}
    Effective Date-Answer: {row['Effective Date-Answer']}
    Expiration Date-Answer: {row['Expiration Date-Answer']}
    Renewal Term-Answer: {row['Renewal Term-Answer']}
    Governing Law-Answer: {row['Governing Law-Answer']}
    Termination For Convenience-Answer: {row['Termination For Convenience-Answer']}
    Exclusivity-Answer: {row['Exclusivity-Answer']}
    Revenue/Profit Sharing-Answer: {row['Revenue/Profit Sharing-Answer']}
    Post-Termination Services: {row['Post-Termination Services']}
    Discrepancy: {row['Discrepancy']}
    """
    # Generate embedding
    embedding = vec.get_embedding(content)

    # Prepare metadata
    metadata = {
        "filename": row["Filename"],
        "document_name": row["Document Name"],
        "parties": row["Parties"],
        "effective_date": row["Effective Date"],
        "expiration_date": row["Expiration Date"],
        "governing_law": row["Governing Law"],
        "exact_law": row["Exact_Law"],
    }

    return pd.Series(
        {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": metadata,
            "contents": content,
            "embedding": embedding,
        }
    )
records_df = df.apply(prepare_record, axis=1)



# Create tables and insert data
vec.create_tables()
vec.create_index()  # DiskAnnIndex
vec.upsert(records_df)

# %%




"""
def prepare_record(row):
    Prepare a record for insertion into the vector store.

    This function creates a record with a UUID version 1 as the ID, which captures
    the current time or a specified time.

    Note:
        - By default, this function uses the current time for the UUID.
        - To use a specific time:
          1. Import the datetime module.
          2. Create a datetime object for your desired time.
          3. Use uuid_from_time(your_datetime) instead of uuid_from_time(datetime.now()).

        Example:
            from datetime import datetime
            specific_time = datetime(2023, 1, 1, 12, 0, 0)
            id = str(uuid_from_time(specific_time))

        This is useful when your content already has an associated datetime.

    content = f"Question: {row['question']}\nAnswer: {row['answer']}"
    
    embedding = vec.get_embedding(content)
    return pd.Series(
        {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "category": row["category"],
                "created_at": datetime.now().isoformat(),
            },
            "contents": content,
            "embedding": embedding,
        }
    )

"""
