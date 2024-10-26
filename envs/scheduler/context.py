import random
import time
import os
import numpy as np

class context():

    def __init__(self, conn, table_list, max_worker):
        self.connection = conn
        self.table_list = table_list
        self.table_2_mem = {}  #是说一个table多少数据在内存中
        self.dbmem = 0
        self.decay = 0.5
        self.add = 2
        self.tot_buffer_blocks = 1600 #128000/8
        try:
            self.init_buffer_table()
        except:
            print('init_buffer_table failed')
        self.query_count = 0
        self.max_worker=max_worker #开始的时候，不需要动buffer

    def init_buffer_table(self):
        cache_sql = """
                    SELECT
                    c.relname,
                    count(*)  as buffered
                    FROM pg_class c
                    INNER JOIN pg_buffercache b ON b.relfilenode = c.relfilenode
                    INNER JOIN pg_database d ON (b.reldatabase = d.oid AND d.datname = current_database())
                    GROUP BY c.oid, c.relname
                    ORDER BY 2 DESC
                    LIMIT 20;
                    """
        with self.connection.cursor() as cursor:
            cursor.execute(cache_sql)
            rows = cursor.fetchall()
        for table in self.table_list.tables:
            self.table_2_mem[table] = 0.001
        for table, blocks in rows:
            if table in self.table_list.tables:
                init_ratio = blocks /  self.table_list.get_table_size(table)
                self.table_2_mem[table] = init_ratio


    def normalize_ratio(self):
        tot_blocks = 0
        for table in self.table_list.tables:
            tot_blocks += self.table_list.get_table_size(table)*self.table_2_mem[table]

        all_ratio = tot_blocks/self.tot_buffer_blocks

        for table in self.table_2_mem.keys():
            self.table_2_mem[table] = self.table_2_mem[table]/all_ratio

            if self.table_2_mem[table]>1:
                self.table_2_mem[table] = 1



    def finish_query(self, tables_in_query):
        self.query_count += 1
        if self.query_count<self.max_worker:
            return

        if (self.query_count % 10 == 0):
            self.init_buffer_table()
            return

        for table in self.table_list.tables:
            if table in tables_in_query:
                self.table_2_mem[table] = self.table_2_mem[table]*self.add
                if self.table_2_mem[table]>1:
                    self.table_2_mem[table] =1
            else:
                self.table_2_mem[table] = self.table_2_mem[table] * self.decay

        self.normalize_ratio()

    def get_similarity_by_query(self, tables_in_query, cost_of_query):
        benefit = 0.0
        tot_size = 0.0
        for table in tables_in_query:
            size = self.table_list.get_table_size(table)
            tot_size += size
            benefit += self.table_2_mem[table]* size
        relative_benefit = benefit/tot_size*cost_of_query
        #relative_benefit = benefit*cost_of_query
        #return benefit
        return relative_benefit
