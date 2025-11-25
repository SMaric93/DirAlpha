import wrds
from . import config, db

def search_link_tables():
    print("Connecting to WRDS...")
    try:
        db_conn = db.get_db()
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    if db_conn:
        table_name = "wrdsapps.exec_boardex_link"
        print(f"\nInspecting columns of '{table_name}'...")
        try:
            desc = db_conn.describe_table(library="wrdsapps", table="exec_boardex_link")
            print(desc)
        except Exception as e:
            print(f"Error describing {table_name}: {e}")
    else:
        print("Failed to connect to WRDS.")

if __name__ == "__main__":
    search_link_tables()
