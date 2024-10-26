class db_stat():

    def __init__(self, database):
        #self.database = Database(collect_db_info=True)  # 存放元数据啥的

        self.database = database

        self.table2rows = {}

        #self.att2num ={}
        self.att2values = {}
        self.att2histogram ={}
        #self.relations_attributes = None

        self.get_table_info()
        self.get_column_info()

        self.min_prob = 1e-5

    def get_table_info(self):
        query = """select relname, reltuples
                           from pg_class r join pg_namespace n
                           on (relnamespace = n.oid)
                           where relkind ='r' and
                           n.nspname='public' """

        cursor = self.database.conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        for row in rows:
            """
            if row[1] != 0:
                self.table2rows[row[0]] = math.log(row[1])
            else:
                self.table2rows[row[0]] = 0
            """
            self.table2rows[row[0]] = row[1]
        cursor.close()


    def get_column_info(self):
        query = """select tablename, attname, n_distinct, histogram_bounds 
                                    from pg_stats where schemaname='public' """

        cursor = self.database.conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()

        for row in rows:
            self.att2values[row[0]+'.'+row[1]] = row[2]
            self.att2histogram[row[0] + '.'+ row[1]] = row[3]

        cursor.close()



    def compute_sel_join(self, op, lefttable, leftatt, righttable, rightatt):

        left_distinct_values = self.att2values[lefttable+'.'+leftatt]
        left_histogram = self.att2histogram[lefttable+'.'+leftatt]
        left_rows = self.table2rows[lefttable]


        right_distinct_values = self.att2values[righttable+'.'+rightatt]
        right_histogram = self.att2histogram[righttable+'.'+rightatt]

        right_rows = self.table2rows[righttable]

        if left_distinct_values<0: #一般是-1,表示主键
            left_distinct_values = left_rows
        if right_distinct_values<0:
            right_distinct_values = right_rows


        rought_selecitively =1/max(left_distinct_values, right_distinct_values)

        return max(rought_selecitively, self.min_prob)
        #直方图带后续来补充

        #左右两边的取值都比较少的情况
        if left_distinct_rows>0 and right_distinct_rows>0:
            if op=='eq':
                return left_rows*right_rows/(left_distinct_rows * right_distinct_rows)
            if op=='gt':
                print("exists a gt")
                return left_rows * right_rows / (left_distinct_rows * right_distinct_rows)
            if op=='lt':
                print("exists a lt")
                return left_rows * right_rows / (left_distinct_rows * right_distinct_rows)


        if left_distinct_rows>0 and right_distinct_rows<0:
            if op=='eq':
                return left_rows*right_rows/(left_distinct_rows * right_distinct_rows)
            if op=='gt':
                print("exists a gt")
                return left_rows * right_rows / (left_distinct_rows * right_distinct_rows)
            if op=='lt':
                print("exists a lt")
                return left_rows * right_rows / (left_distinct_rows * right_distinct_rows)

        if left_distinct_rows>0 and right_distinct_rows<0:
            if op=='eq':
                return left_rows*right_rows/(left_distinct_rows * right_distinct_rows)
            if op=='gt':
                print("exists a gt")
                return left_rows * right_rows / (left_distinct_rows * right_distinct_rows)
            if op=='lt':
                print("exists a lt")
                return left_rows * right_rows / (left_distinct_rows * right_distinct_rows)

        #量大，采用histogram
        if left_distinct_rows<0 and right_distinct_rows<0:

            if op=='eq':
                return left_rows*right_rows/(left_distinct_rows * right_distinct_rows)
            if op=='gt':
                print("exists a gt")
                return left_rows * right_rows / (left_distinct_rows * right_distinct_rows)
            if op=='lt':
                print("exists a lt")
                return left_rows * right_rows / (left_distinct_rows * right_distinct_rows)

    def computer_sel_column(self, op, table, att, value):

        table_rows = self.table2rows[table]
        distinct_rows = self.att2values[table + '.' + att]
        histogram = self.att2histogram[table + '.' + att]

        if distinct_rows<0:
            distinct_rows = table_rows

        if op == 'eq' or op == 'missing':
            return max(1/distinct_rows, self.min_prob)

        elif op == 'gt' or op=='gte':
            if len(histogram)>0:
                rows = histogram.split(',')
                cnt =0
                for row in rows:
                    if row>str(value):  #直接转换str, 有点问题，先凑合用吧
                        cnt=cnt+1
                return cnt/len(rows)

        elif op == 'lt' or op=='lte':
            if len(histogram)>0:
                rows = histogram.split(',')
                cnt =0
                for row in rows:
                    if row<str(value):
                        cnt=cnt+1
                return cnt/len(rows)

        elif op == 'between':
            if len(histogram)>0:
                rows = histogram.split(',')
                cnt =0
                for row in rows:
                    if row>str(value[0]) and row<str(value[1]):
                        cnt=cnt+1
                return cnt/len(rows)
        elif op == 'neq':
            if distinct_rows>0:
                return (distinct_rows -1) / distinct_rows

        elif op =='in':
            if distinct_rows ==-1:
                distinct_rows = table_rows
            tot = len(value)
            return tot/distinct_rows
        elif op =='nlike':
            if distinct_rows>0:
                return (distinct_rows -1) / distinct_rows

        elif op =='like':
            if distinct_rows>0:
                return 1 / distinct_rows  #只是近似计算

    def evaluate_queries(self):
        itr =0
        for query in self.queries:
            cost=self.database.db_optimizer_time(query['query'])
            print('ID %d SQL: %s with    Relations %d     Evaluation Cost %f'%(itr, query['file'], int(query['relations_num']), cost))
            itr = itr+1






'''

allqueries = queries()
allqueries.get_queries()

    def get_orexp_selectivity(self, exp):

        sel =0
        for oneand in exp:
            for op in oneand:
                exp = oneand[op]
                selectivity = self.get_exp_selectivity(op, exp)
                sel = sel +selectivity

        return sel

    def get_queries(self):
        self.queries = self.database.get_queries()
        cnt = 0
        for query in self.queries:
            self.handle_table(query['moz'])
            self.handle_clause(query['moz'], cnt)

            self.build_graph(query)

            cnt = cnt + 1

    def handle_table(self, moz):

        #query['moz']['from'][1]['name']
        tables = moz['from']
        for table in tables:
            tablename = table['value'] #alias
            tableid = table['name']
            self.alias2table[tableid]= tablename
            if tablename not in self.table2rows:
                reltuples = self.get_reltuples(tablename)
                self.table2rows[tablename]= reltuples


            #print("%s Tuples= %d" %(tablename, reltuples))

        return tables

    def get_reltuples(self, tablename):
        query = """select reltuples
                    from pg_class r join pg_namespace n
                    on (relnamespace = n.oid)
                    where relkind ='r' and
                    n.nspname='public'
                    and relname ="""
        query = query + "\'"+tablename+"\'"

        cursor = self.database.conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        return rows[0][0]

    def handle_clause(self, clauses):

        logic_fromid = len(self.NodesDic['logic'])-1

        itr=0
        for clause in clauses:
            for op in clause:
                exp = clause[op]
                #print('%d Query %d Clause' %(queryid,itr))
                print(op, end=' ')
                print(exp)

                if op =='or' or op =='and':
                    self.create_logic_node(op)
                    logic_toid = len(self.NodesDic['logic'])-1
                    self.create_edge('lonc',logic_fromid, logic_toid)
                    self.handle_clause(self, op, clauses)
                else:
                    selectivity = self.get_exp_selectivity(op, exp)

                    self.create_clause_node(op, selectivity)
                    clause_toid = len(self.NodesDic['clause']) - 1
                    self.create_edge('lonc',logic_fromid, clause_toid)

                print('Itr %d with selectivity=%f' %(itr, selectivity))
                itr = itr + 1

    def rows_and_histogram(self, table, att):
        query = """select n_distinct, histogram_bounds 
                            from pg_stats where schemaname='public' """
        query = query + "and tablename=\'" + self.alias2table[table]+ "\' and attname=\'" + att + "\'"

        cursor = self.database.conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        return rows[0][0], rows[0][1]

'''
