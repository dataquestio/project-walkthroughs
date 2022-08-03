import sqlite3
import pandas as pd

class DBStorage():
    def __init__(self):
        self.con = sqlite3.connect('links.db')
        self.setup_tables()

    def setup_tables(self):
        cur = self.con.cursor()
        results_table = r"""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY,
                query TEXT,
                rank INTEGER,
                link TEXT,
                title TEXT,
                html TEXT,
                UNIQUE(query, link)
            );
            """
        cur.execute(results_table)
        self.con.commit()
        cur.close()

    def query_results(self, query):
        df = pd.read_sql(f"select * from results where query='{query}'", self.con)
        return df

    def insert_row(self, values):
        cur = self.con.cursor()
        res = cur.execute(r"select count(*) from results where query=? and link=?", (values[0], values[1]))
        rows = res.fetchone()
        if rows[0] == 0:
            cur.execute('INSERT INTO results (query, rank, link, title, html) VALUES(?, ?, ?, ?, ?)', values)
            self.con.commit()
        cur.close()