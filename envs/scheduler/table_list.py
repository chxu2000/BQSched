from sklearn.preprocessing import OneHotEncoder


class TableList():

    def __init__(self, connection):
        self.conn = connection
        self.tables = []
        self.table_sizes =[]
        self.table_encoder = None
        self.build_table()
        self.build_table_size()
        self.build_encoder()

    def build_table_size(self):
        cursor = self.conn.cursor()
        query = 'select tablename from pg_tables where schemaname = \'public\';'
        cursor.execute(query)
        tables = [p[0] for p in cursor.fetchall()]
        table_size = dict()
        for table in tables:
            query = 'SELECT pg_table_size(\'{}\')/8192;'.format(table)
            cursor.execute(query)
            size = cursor.fetchall()[0][0]
            table_size[table] = size
        cursor.close()
        self.table_sizes = table_size

    def build_table(self):
        cursor = self.conn.cursor()
        query = 'select tablename from pg_tables where schemaname = \'public\';'
        cursor.execute(query)
        tables = [p[0] for p in cursor.fetchall()]
        self.tables = tables

    def get_table_size(self, table):
        return self.table_sizes[table]

    def build_encoder(self):
        self.encoder = OneHotEncoder()
        table_2d = [[table] for table in self.tables]
        self.encoder.fit(table_2d)

    def get_encoder(self, query_tables):
        unique_tables = query_tables
        result_vector = None
        for tab in unique_tables:
            encoded_vector = self.encoder.transform([[tab]]).toarray()
            if result_vector is not None:
                result_vector = result_vector + encoded_vector
            else:
                result_vector = encoded_vector
        return result_vector