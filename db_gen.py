import sqlite3
# Connect to SQLite database
try:
    conn = sqlite3.connect('customer_faces_data.db')
    c = conn.cursor()
    #print("Successfully connected to the database")
except sqlite3.Error as e:
    print("SQLite error:", e)

# Create a table to store face data if it doesn't exist
try:
    c.execute("INSERT INTO customers (customer_uid, customer_name) VALUES (?, ?)", (204404712, "Rukundo"))
    #print("Table 'customers' created successfully")
except sqlite3.Error as e:
    print("SQLite error:", e)