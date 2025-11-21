import wrds
from . import config

def describe_committee_table():
    print("Connecting to WRDS...")
    db = config.get_wrds_connection()
    if db:
        print("Connected. Describing 'boardex.na_board_dir_committees'...")
        try:
            desc = db.describe_table(library="boardex", table="na_board_dir_committees")
            print("\n--- Table Columns ---")
            print(desc)
            print("---------------------\n")
        except Exception as e:
            print(f"Error describing table: {e}")
    else:
        print("Failed to connect to WRDS.")

if __name__ == "__main__":
    describe_committee_table()
