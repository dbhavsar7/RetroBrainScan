import sqlite3  # Import SQLite library

RETRO_DATABASE = 'RetroBrainScanDB.db'
TABLE_NAME = 'ImageData'


def read_metadata_from_db():
    conn = sqlite3.connect(RETRO_DATABASE)
    c = conn.cursor()
    
    c.execute(f"SELECT * FROM {TABLE_NAME}")
    rows = c.fetchall()  # Fetch all rows from the executed query
    
    if rows:
        print("Current entries in the CameraMetadata database:")
        for row in rows:
            print(f"ID: {row[0]}, File Name: {row[1]}, Upload Timestamp: {row[2]}, "
                    f"metatada: {row[4]}")
            print("--------------------------------------------------")
            # print(f"ID: {row[0]}, File Name: {row[1]}, Upload Timestamp: {row[2]}, "
            #         f"Image Data: {row[3]}, metatada: {row[4]}")
            '''
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_data TEXT NOT NULL,
                metadata TEXT
            '''
    else:
        print(f"No entries found in the {RETRO_DATABASE} database.")

    conn.close()

# Call this function to display the entries
read_metadata_from_db()