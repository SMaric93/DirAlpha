import wrds
from . import config, utils

def search_link_tables():
    print("Connecting to WRDS...")
    try:
        db = utils.get_db()
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    if db:
        libraries = ["boardex", "wrds_apps", "wrds_lib"]
        for lib in libraries:
            print(f"\nSearching in '{lib}'...")
            try:
                tables = db.list_tables(library=lib)
                # pyrefly: ignore [not-iterable]
                for table in tables:
                    if "link" in table or "gvkey" in table or "mapping" in table:
                        print(f"FOUND: {lib}.{table}")
            except Exception as e:
                print(f"Error listing {lib}: {e}")
    else:
        print("Failed to connect to WRDS.")

if __name__ == "__main__":
    search_link_tables()
