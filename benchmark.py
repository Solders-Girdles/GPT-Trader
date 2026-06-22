import sqlite3
import time
from pathlib import Path

def test_database_repair_benchmark(num_tables: int, rows_per_table: int):
    database_path = Path("test_db.sqlite")
    if database_path.exists():
        database_path.unlink()

    conn = sqlite3.connect(str(database_path))

    # Create tables and insert data
    for i in range(num_tables):
        conn.execute(f"CREATE TABLE table_{i} (id INTEGER PRIMARY KEY, data TEXT)")
        # Insert rows
        rows = [(j, f"data_{j}") for j in range(rows_per_table)]
        conn.executemany(f"INSERT INTO table_{i} (id, data) VALUES (?, ?)", rows)

    conn.commit()
    conn.close()

    # Run the N+1 recovery simulation (current code logic)
    start_time = time.time()

    conn = sqlite3.connect(str(database_path))
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type=\"table\" AND name NOT LIKE \"sqlite_%\"")
    tables = [row[0] for row in cursor]

    recovered_data = {}
    for table in tables:
        cursor = conn.execute(f"SELECT * FROM {table}")
        recovered_data[table] = cursor.fetchall()

    conn.close()

    # recreate simulation
    database_path.unlink()
    conn = sqlite3.connect(str(database_path))
    conn.close()

    baseline_time = time.time() - start_time

    # Recreate the data for the next test
    database_path = Path("test_db.sqlite")
    if database_path.exists():
        database_path.unlink()
    conn = sqlite3.connect(str(database_path))
    for i in range(num_tables):
        conn.execute(f"CREATE TABLE table_{i} (id INTEGER PRIMARY KEY, data TEXT)")
        rows = [(j, f"data_{j}") for j in range(rows_per_table)]
        conn.executemany(f"INSERT INTO table_{i} (id, data) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()

    # Run ATTACH method
    start_time_attach = time.time()
    new_db_path = Path("new_db.sqlite")
    if new_db_path.exists():
        new_db_path.unlink()

    # We want to emulate the resilience: read table by table and copy
    conn_new = sqlite3.connect(str(new_db_path))
    conn_new.execute(f"ATTACH DATABASE \"{database_path}\" AS corrupted")

    cursor = conn_new.execute("SELECT name, sql FROM corrupted.sqlite_master WHERE type=\"table\" AND name NOT LIKE \"sqlite_%\"")
    tables_sql = cursor.fetchall()

    recovered_count = 0
    for table_name, table_sql in tables_sql:
        try:
            # Create the table in new db
            if table_sql:
                conn_new.execute(table_sql)
            # Copy data directly within SQLite engine
            conn_new.execute(f"INSERT INTO {table_name} SELECT * FROM corrupted.{table_name}")
            recovered_count += 1
        except sqlite3.Error as e:
            # Drop the partially created table if it failed
            conn_new.execute(f"DROP TABLE IF EXISTS {table_name}")

    conn_new.commit()
    conn_new.execute("DETACH DATABASE corrupted")
    conn_new.close()

    attach_time = time.time() - start_time_attach

    print(f"Num tables: {num_tables}, Rows per table: {rows_per_table}")
    print(f"N+1 Python Memory time: {baseline_time:.4f} seconds")
    print(f"ATTACH Copy time: {attach_time:.4f} seconds")

    if database_path.exists():
        database_path.unlink()
    if new_db_path.exists():
        new_db_path.unlink()

test_database_repair_benchmark(100, 1000)
test_database_repair_benchmark(50, 10000)
test_database_repair_benchmark(10, 50000)
