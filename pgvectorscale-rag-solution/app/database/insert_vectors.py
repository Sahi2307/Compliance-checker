
import sys
import os

# Add the root directory (where app is located) to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from datetime import datetime
import pandas as pd
from vector_store import VectorStore
from timescale_vector.client import uuid_from_time

# Initialize VectorStore
vec = VectorStore()

# Read the CSV file
df = pd.read_csv("../data/dataset.csv", sep=",")  # Assuming the cleaned file uses commas as separators
df.head()

# Prepare data for insertion
def prepare_record(row):
    """Prepare a record for insertion into the vector store.

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

        This is useful when your content already has an associated datetime."""
    # Combine key columns into a structured content format
    content = (
        f"Category: {row['Category']}\n"
        f"Document Name: {row['Document Name']}\n"
        f"Parties: {row['Parties']}\n"
        f"Agreement Date: {row['Agreement Date']}\n"
        f"Effective Date: {row['Effective Date']}\n"
        f"Renewal Term: {row['Renewal Term']}\n"
        f"Notice To Terminate Renewal: {row['Notice To Terminate Renewal']}\n"
        f"Expiration Date: {row['Expiration Date']}\n"
        f"Governing Law: {row['Governing Law']}\n"
        f"Exclusivity: {row['Exclusivity']}\n"
        f"Post-Termination Services: {row['Post-Termination Services']}\n"
        f"Exact Law: {row['Exact_Law']}\n"
        f"detailed_law: {row['detailed_law']}\n"
        f"Discrepancy: {row['Discrepancy']}"
    )
    
    # Generate embedding for the content
    embedding = vec.get_embedding(content)
    
    # Return a record as a pandas Series
    return pd.Series(
        {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "Documentname": row["Document Name"],  # Store additional metadata
                "created_at": datetime.now().isoformat(),
            },
            "contents": content,
            "embedding": embedding,
        }
    )

# Apply the function to each row
records_df = df.apply(prepare_record, axis=1)

# Create tables and insert data
vec.create_tables()
vec.create_index()  # DiskAnnIndex
vec.upsert(records_df)

print("Data insertion into vector store is complete.")
