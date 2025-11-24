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
        tables = ["boardex.na_board_dir_announcements", "boardex.na_wrds_org_composition"]
        for table in tables:
            print(f"\nInspecting columns of '{table}'...")
            try:
                lib, tab = table.split(".")
                desc = db_conn.describe_table(library=lib, table=tab)
                print(desc)
            except Exception as e:
                print(f"Error describing {table}: {e}")
    else:
        print("Failed to connect to WRDS.")

if __name__ == "__main__":
    search_link_tables()
