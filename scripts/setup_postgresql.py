"""
PostgreSQL Setup Script for SQL RAG and Vector Storage

This script sets up PostgreSQL for SQL RAG and Vector Storage, including:
- Creating a database
- Setting up the pgvector extension
- Creating complex schema structures for testing
- Creating necessary database users and permissions
"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def create_database(host, port, user, password, dbname):
    """
    Create a PostgreSQL database.

    Args:
        host: Database host
        port: Database port
        user: Database user
        password: Database password
        dbname: Database name
    """
    # Connect to PostgreSQL server
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    # Create a cursor
    cursor = conn.cursor()

    # Check if database exists
    cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{dbname}'")
    exists = cursor.fetchone()

    if not exists:
        # Create database
        print(f"Creating database '{dbname}'...")
        cursor.execute(f"CREATE DATABASE {dbname}")
        print(f"Database '{dbname}' created successfully")
    else:
        print(f"Database '{dbname}' already exists")

    # Close connection
    cursor.close()
    conn.close()


def setup_pgvector(host, port, user, password, dbname):
    """
    Set up pgvector extension.

    Args:
        host: Database host
        port: Database port
        user: Database user
        password: Database password
        dbname: Database name
    """
    # Connect to the database
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    # Create a cursor
    cursor = conn.cursor()

    # Check if pgvector extension exists
    cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    exists = cursor.fetchone()

    if not exists:
        # Try to create pgvector extension
        try:
            print("Creating pgvector extension...")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            print("pgvector extension created successfully")
        except Exception as e:
            print(f"Warning: Could not create pgvector extension: {str(e)}")
            print("Continuing without pgvector extension. Vector operations will not be available.")
            print("To install pgvector, follow the instructions at: https://github.com/pgvector/pgvector")
    else:
        print("pgvector extension already exists")

    # Close connection
    cursor.close()
    conn.close()


def create_complex_schema(host, port, user, password, dbname):
    """
    Create complex schema structures for testing.

    Args:
        host: Database host
        port: Database port
        user: Database user
        password: Database password
        dbname: Database name
    """
    # Connect to the database
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname
    )

    # Create a cursor
    cursor = conn.cursor()

    # Create tables
    print("Creating tables...")

    # Create customers table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id SERIAL PRIMARY KEY,
        first_name VARCHAR(50) NOT NULL,
        last_name VARCHAR(50) NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        phone VARCHAR(20),
        address VARCHAR(200),
        city VARCHAR(50),
        state VARCHAR(50),
        country VARCHAR(50),
        postal_code VARCHAR(20),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Create products table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        product_id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        description TEXT,
        price DECIMAL(10, 2) NOT NULL,
        category VARCHAR(50),
        inventory_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Create orders table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        order_id SERIAL PRIMARY KEY,
        customer_id INTEGER REFERENCES customers(customer_id),
        order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status VARCHAR(20) DEFAULT 'pending',
        total_amount DECIMAL(10, 2) NOT NULL,
        shipping_address VARCHAR(200),
        shipping_city VARCHAR(50),
        shipping_state VARCHAR(50),
        shipping_country VARCHAR(50),
        shipping_postal_code VARCHAR(20),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Create order_items table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS order_items (
        order_item_id SERIAL PRIMARY KEY,
        order_id INTEGER REFERENCES orders(order_id),
        product_id INTEGER REFERENCES products(product_id),
        quantity INTEGER NOT NULL,
        price DECIMAL(10, 2) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Check if pgvector extension is available
    cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    pgvector_available = cursor.fetchone() is not None

    if pgvector_available:
        # Create documents table with vector support
        print("Creating documents table with vector support...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id SERIAL PRIMARY KEY,
            title VARCHAR(200) NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB,
            embedding VECTOR(1536),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Create index on embedding
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING ivfflat (embedding vector_cosine_ops)
        """)
    else:
        # Create documents table without vector support
        print("Creating documents table without vector support (pgvector not available)...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id SERIAL PRIMARY KEY,
            title VARCHAR(200) NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB,
            embedding_json JSONB,  -- Store embeddings as JSON array instead of vector
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Create index on title for basic search
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS documents_title_idx ON documents USING btree (title)
        """)

    # Commit changes
    conn.commit()

    print("Tables created successfully")

    # Close connection
    cursor.close()
    conn.close()


def insert_sample_data(host, port, user, password, dbname):
    """
    Insert sample data for testing.

    Args:
        host: Database host
        port: Database port
        user: Database user
        password: Database password
        dbname: Database name
    """
    # Connect to the database
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname
    )

    # Create a cursor
    cursor = conn.cursor()

    # Insert sample customers
    print("Inserting sample customers...")
    cursor.execute("""
    INSERT INTO customers (first_name, last_name, email, phone, address, city, state, country, postal_code)
    VALUES
        ('John', 'Doe', 'john.doe@example.com', '555-123-4567', '123 Main St', 'New York', 'NY', 'USA', '10001'),
        ('Jane', 'Smith', 'jane.smith@example.com', '555-987-6543', '456 Elm St', 'Los Angeles', 'CA', 'USA', '90001'),
        ('Bob', 'Johnson', 'bob.johnson@example.com', '555-555-5555', '789 Oak St', 'Chicago', 'IL', 'USA', '60601')
    ON CONFLICT (email) DO NOTHING
    """)

    # Insert sample products
    print("Inserting sample products...")
    cursor.execute("""
    INSERT INTO products (name, description, price, category, inventory_count)
    VALUES
        ('Laptop', 'High-performance laptop with 16GB RAM', 1299.99, 'Electronics', 10),
        ('Smartphone', 'Latest smartphone with 5G support', 899.99, 'Electronics', 20),
        ('Headphones', 'Noise-cancelling wireless headphones', 199.99, 'Electronics', 30),
        ('T-shirt', 'Cotton t-shirt in various colors', 19.99, 'Clothing', 100),
        ('Jeans', 'Denim jeans in various sizes', 49.99, 'Clothing', 50)
    """)

    # Insert sample orders
    print("Inserting sample orders...")
    cursor.execute("""
    INSERT INTO orders (customer_id, status, total_amount, shipping_address, shipping_city, shipping_state, shipping_country, shipping_postal_code)
    VALUES
        (1, 'completed', 1299.99, '123 Main St', 'New York', 'NY', 'USA', '10001'),
        (2, 'processing', 1099.98, '456 Elm St', 'Los Angeles', 'CA', 'USA', '90001'),
        (3, 'pending', 249.98, '789 Oak St', 'Chicago', 'IL', 'USA', '60601')
    """)

    # Insert sample order items
    print("Inserting sample order items...")
    cursor.execute("""
    INSERT INTO order_items (order_id, product_id, quantity, price)
    VALUES
        (1, 1, 1, 1299.99),
        (2, 2, 1, 899.99),
        (2, 3, 1, 199.99),
        (3, 4, 5, 19.99),
        (3, 5, 3, 49.99)
    """)

    # Commit changes
    conn.commit()

    print("Sample data inserted successfully")

    # Close connection
    cursor.close()
    conn.close()


def main():
    """Main function to set up PostgreSQL."""
    # Load environment variables
    load_dotenv()

    # PostgreSQL connection parameters for vector database
    vector_host = os.getenv("VECTOR_DB_HOST", "localhost")
    vector_port = os.getenv("VECTOR_DB_PORT", "5432")
    vector_user = os.getenv("VECTOR_DB_USER", "postgres")
    vector_password = os.getenv("VECTOR_DB_PASSWORD", "102938")
    vector_dbname = os.getenv("VECTOR_DB_NAME", "vectordb")

    # PostgreSQL connection parameters for SQL RAG database
    sql_rag_host = os.getenv("SQL_RAG_DB_HOST", "localhost")
    sql_rag_port = os.getenv("SQL_RAG_DB_PORT", "5432")
    sql_rag_user = os.getenv("SQL_RAG_DB_USER", "postgres")
    sql_rag_password = os.getenv("SQL_RAG_DB_PASSWORD", "102938")
    sql_rag_dbname = os.getenv("SQL_RAG_DB_NAME", "langgraph_agent_db")

    # Setup Vector Database
    print("\n=== Setting up Vector Database ===")
    create_database(vector_host, vector_port, vector_user, vector_password, vector_dbname)
    setup_pgvector(vector_host, vector_port, vector_user, vector_password, vector_dbname)

    # Create documents table for vector storage
    conn = psycopg2.connect(
        host=vector_host,
        port=vector_port,
        user=vector_user,
        password=vector_password,
        dbname=vector_dbname
    )
    cursor = conn.cursor()

    # Check if pgvector extension is available
    cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    pgvector_available = cursor.fetchone() is not None

    if pgvector_available:
        # Create documents table with vector support
        print("Creating documents table with vector support...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id SERIAL PRIMARY KEY,
            title VARCHAR(200) NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB,
            embedding VECTOR(1536),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Create index on embedding
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING ivfflat (embedding vector_cosine_ops)
        """)
    else:
        # Create documents table without vector support
        print("Creating documents table without vector support (pgvector not available)...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id SERIAL PRIMARY KEY,
            title VARCHAR(200) NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB,
            embedding_json JSONB,  -- Store embeddings as JSON array instead of vector
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Create index on title for basic search
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS documents_title_idx ON documents USING btree (title)
        """)

    conn.commit()
    cursor.close()
    conn.close()

    print(f"Vector database setup completed successfully!")
    print(f"Vector DB Connection string: postgresql://{vector_user}:{vector_password}@{vector_host}:{vector_port}/{vector_dbname}")

    # Setup SQL RAG Database
    print("\n=== Setting up SQL RAG Database ===")
    create_database(sql_rag_host, sql_rag_port, sql_rag_user, sql_rag_password, sql_rag_dbname)

    # Create complex schema for SQL RAG
    create_complex_schema(sql_rag_host, sql_rag_port, sql_rag_user, sql_rag_password, sql_rag_dbname)

    # Insert sample data for SQL RAG
    insert_sample_data(sql_rag_host, sql_rag_port, sql_rag_user, sql_rag_password, sql_rag_dbname)

    print(f"SQL RAG database setup completed successfully!")
    print(f"SQL RAG DB Connection string: postgresql://{sql_rag_user}:{sql_rag_password}@{sql_rag_host}:{sql_rag_port}/{sql_rag_dbname}")


if __name__ == "__main__":
    main()
