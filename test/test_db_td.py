from db_td import TeradataDatabase


if __name__ == "__main__":
    db = TeradataDatabase()
    db.connect()
    print(db.get_schema())

    query = """
    SELECT top 3 * FROM raven.branches
    """
    print(db.execute_query(query))