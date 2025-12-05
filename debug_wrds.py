import logging
import sys
from director_alpha import db, config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_wrds():
    try:
        conn = db.get_db()
        
        print("="*40)
        print("Inspect frb schema")
        tables = conn.list_tables(library="frb")
        print(f"Tables in 'frb': {tables}")
        
        if "rates_daily" in tables:
            print("\nColumns in frb.rates_daily:")
            desc = conn.describe_table(library="frb", table="rates_daily")
            print(desc['name'].tolist())
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_wrds()

