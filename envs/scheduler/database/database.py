import os, pandas as pd
import psycopg2
import pprint
from configparser import ConfigParser
# import pyodbc


class Database:
    def __init__(self, conf):
        self.conf = conf
        self.use_simulator = self.conf.getboolean('database', 'use_simulator')
        self.pp = pprint.PrettyPrinter(indent=2)
        self.conn = self.connect() if not self.use_simulator else None
        self.counter = 0
        self.aliases = {}

        # Build database related info
        # - tables,
        # - relations (original-tables + aliases),
        # - {relation : attributes}
        # - attributes

    def connect(self, database=None):
        try:
            database = self.conf['database']['database'] if database is None else database
            if database.find('_') > database.find('X'):
                database = database[:database.find('_')]
            if self.conf['database']['port'] == '1433':
                connectionString = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.conf['database']['host']};DATABASE={database};UID={self.conf['database']['user']};PWD={self.conf['database']['password']}"
                # conn = pyodbc.connect(connectionString)
            else:
                conn_string = 'host={} port={} dbname={} user={} password={}'.format(
                    self.conf['database']['host'], self.conf['database']['port'], database, 
                    self.conf['database']['user'], self.conf['database']['password']
                )
                conn = psycopg2.connect(conn_string)
                conn.set_session(autocommit=True)
                if database.startswith('tpcds') or database.startswith('imdb'):
                    with conn.cursor() as cur:
                        cur.execute('SET enable_nestloop=off;')
                #cursor.execute('set ON_ERROR_ROLLBACK true')
                return conn
        except (Exception, psycopg2.Error) as error:
            print("Error connecting or pg_hint_plan should be installed", error)

    def get_relations_attributes(self):
        """
        Returns relations and their attributes

        Uses tables/attributes from the database but also aliases found on the dataset's queries

        Args:
            None
        Returns:
            relations: list ['alias1','alias2',..]
            relations_attributes: dict {'alias1':['attr1','attr2',..], .. }

        """
        cursor = self.conn.cursor()
        q = (
            "SELECT c.table_name, c.column_name FROM information_schema.columns c "
            "INNER JOIN information_schema.tables t ON c.table_name = t.table_name "
            "AND c.table_schema = t.table_schema "
            "AND t.table_type = 'BASE TABLE' "
            "AND t.table_schema = 'public' "
            "AND c.table_name != 'queries'"
        )
        cursor.execute(q)
        rows = cursor.fetchall()
        cursor.close()

        tables_attributes = {}
        for table, attribute in rows:
            if table in tables_attributes:
                tables_attributes[table].append(attribute)
            else:
                tables_attributes[table] = [attribute]

        tables = list(tables_attributes.keys())
        relations_attributes = {}
        relations = []
        relations_tables = {}
        x = self.get_queries_incremental(target="")
        for group in x:
            for q in group:
                for r in q["moz"]["from"]:
                    if r["name"] not in relations:
                        relations.append(r["name"])
                        relations_attributes[r["name"]] = tables_attributes[r["value"]]
                        relations_tables[r["name"]] = r["value"]
        return tables, relations, relations_attributes, relations_tables

    def print_relations_attrs(self):
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(self.relations_attributes)

    def get_query_by_id(self, id):
        cursor = self.conn.cursor()
        q = "SELECT * FROM queries WHERE id = %s"
        cursor.execute(q, (str(id),))
        rows = cursor.fetchone()
        cursor.close()
        attrs = [
            "id",
            "file",
            "relations_num",
            "query",
            "moz",
            "planning",
            "execution",
            "cost",
        ]
        zipbObj = zip(attrs, rows)
        return dict(zipbObj)

    def get_query_by_filename(self, file):
        file = file + ".sql"
        cursor = self.conn.cursor()
        q = "SELECT * FROM queries WHERE file_name = %s"
        cursor.execute(q, (file,))
        rows = cursor.fetchone()
        cursor.close()
        attrs = [
            "id",
            "file",
            "relations_num",
            "query",
            "moz",
            "planning",
            "execution",
            "cost",
        ]
        zipbObj = zip(attrs, rows)
        return dict(zipbObj)

    def get_queries_incremental(self, target):
        cursor = self.conn.cursor()

        attrs = [
            "id",
            "file",
            "relations_num",
            "query",
            "moz",
            "planning",
            "execution",
            "cost",
        ]

        # Yield all groups one by one
        if target == "":
            q = "SELECT * FROM queries ORDER BY relations_num"
            cursor.execute(q)

        # Yield only one target group
        else:
            q = "SELECT * FROM queries WHERE relations_num=%s"
            cursor.execute(q, (str(target),))

        rows = cursor.fetchall()
        cursor.close()

        qs = {}
        for q in rows:
            q = dict(zip(attrs, q))
            num = q["relations_num"]
            if num not in qs:
                qs[num] = [q]
            else:
                qs[num].append(q)

        keys = list(qs.keys())
        keys.sort()
        for key in keys:
            yield qs[key]

        return None

    def get_queries_incremental_all(self):
        cursor = self.conn.cursor()

        attrs = [
            "id",
            "file",
            "relations_num",
            "query",
            "moz",
            "planning",
            "execution",
            "cost",
        ]

        # Yield only queries one by one order by relations_num
        q = "SELECT * FROM queries ORDER BY relations_num"
        cursor.execute(q)
        rows = cursor.fetchall()
        cursor.close()

        for q in rows:
            yield dict(zip(attrs, q))

        return None

    def get_queries(self):
        query_scale = int(self.conf["database"]["query_scale"])
        if query_scale == 1:
            queries_path = f'envs/scheduler/cache/{self.conf["database"]["host"]}/{self.conf["database"]["database"]}/queries.csv'
        else:
            queries_path = f'envs/scheduler/cache/{self.conf["database"]["host"]}/{self.conf["database"]["database"]}/queries{query_scale}X.csv'
        if os.path.exists(queries_path):
            queries = pd.read_csv(queries_path).values.tolist()
            # qlen = len(queries)
            # for _ in range(int(self.conf['database']['query_scale']) - 1):
            #     for i in range(qlen):
            #         tmp = queries[i].copy()
            #         tmp[0] = tmp[0] + qlen
            #         queries.append(tmp)
        else:
            cursor = self.conn.cursor()

            # Yield only queries one by one order by relations_num
            q = "SELECT * FROM queries;"
            # q = "SELECT * FROM sample_queries;"
            cursor.execute(q)
            queries = cursor.fetchall()
            cursor.close()
        return queries
    
    def get_table_size(self):
        cursor = self.conn.cursor()
        q = 'select tablename from pg_tables where schemaname = \'public\';'
        cursor.execute(q)
        tables = [p[0] for p in cursor.fetchall()]
        table_size = dict()
        for table in tables:
            q = 'SELECT pg_relation_size(\'{}\');'.format(table)
            cursor.execute(q)
            size = cursor.fetchall()[0][0]
            table_size[table] = size
        cursor.close()
        return table_size

    def get_groups_size(self, target, num_of_groups):

        cursor = self.conn.cursor()

        # Size of groups 1 to num_of_groups
        if target == "":
            q = (
                "select sum(count) from (select relations_num, count(*) "
                "as count from queries group by relations_num order by relations_num limit  %s) X"
            )
            cursor.execute(q, (str(num_of_groups),))

        # Size of a specific group
        else:
            q = (
                "select count(*) from queries where relations_num=%s"
            )
            cursor.execute(q, (str(target),))

        row = cursor.fetchone()
        cursor.close()
        return row[0]

    def get_queries_size(self):
        cursor = self.conn.cursor()
        q = "SELECT COUNT(*) FROM queries"
        cursor.execute(q, (str(id),))
        row = cursor.fetchone()
        cursor.close()
        return row[0]

    def close(self):
        if self.conn:
            self.conn.close()

    def optimizer_cost(self, query, force_order=False):
        join_collapse_limit = "SET join_collapse_limit = "
        join_collapse_limit += "1" if force_order else "8"
        query = join_collapse_limit + ";EXPLAIN (FORMAT JSON) " + query + ";"
        cursor = self.conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchone()
        cursor.close()
        return rows[0][0]["Plan"]["Total Cost"]

    def get_optimizer_plan(self, query, force_order=False):
        query = "EXPLAIN (FORMAT JSON) " + query + ";"
        cursor = self.conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchone()
        cursor.close()
        return rows

    def optimizer_cost(self, query, force_order=False):
        join_collapse_limit = "SET join_collapse_limit = "
        join_collapse_limit += "1" if force_order else "8"
        query = join_collapse_limit + ";EXPLAIN (FORMAT JSON) " + query + ";"
        cursor = self.conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchone()
        cursor.close()
        return rows[0][0]["Plan"]["Total Cost"]

    def db_optimizer(self, query, metric):
        if metric ==1:
            return self.db_optimizer_cost(query)
        return self.db_optimizer_time(query)

    def db_optimizer_cost(self, query):
        join_collapse_limit = "SET join_collapse_limit = 18"
        query = join_collapse_limit + ";EXPLAIN (FORMAT JSON) " + query + ";"
        cursor = self.conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchone()
        cursor.close()
        return rows[0][0]["Plan"]["Total Cost"]

    def db_hint_cost(self, query):
        load_hint = "load \'pg_hint_plan\'"
        query = " EXPLAIN (FORMAT JSON) " + query + ";"
        cursor = self.conn.cursor()
        cursor.execute(load_hint)
        #rows = cursor.fetchone()
        #不知道为什么，第一次总是得到最优的值
        cursor.execute(query)
        rows = cursor.fetchone()

        cursor.close()
        return rows[0][0]["Plan"]["Total Cost"]

    def db_hint_pair(self, t0, t1, query):
        load_hint = "load \'pg_hint_plan\'"
        hint_query = ' /*+ NestLoop('+t0+' '+t1+')*/ '
        query = " EXPLAIN (FORMAT JSON) " + hint_query+ query + ";"
        cursor = self.conn.cursor()
        cursor.execute(load_hint)
        #rows = cursor.fetchone()
        #不知道为什么，第一次总是得到最优的值
        cursor.execute(query)
        rows = cursor.fetchone()

        cursor.close()
        return rows[0][0]["Plan"]["Total Cost"]

    def db_hint_time(self, query):
        load_hint = "load \'pg_hint_plan\'"
        query = "EXPLAIN ANALYZE " + query
        cursor = self.conn.cursor()
        cursor.execute(load_hint)
        #rows = cursor.fetchone()
        #不知道为什么，第一次总是得到最优的值
        try:
            cursor.execute('SET statement_timeout = 400000')
            cursor.execute(query)
            rows = cursor.fetchall()

            planning = [float(s) for s in rows[-2][0].split() if self.is_number(s)]
            execution = [float(s) for s in rows[-1][0].split() if self.is_number(s)]

            cursor.close()
            #timecost= rows[0][0]["Plan"]["Total Cost"]
            timecost = planning[-1] + execution[0]
        except :
            self.conn.rollback()
            timecost= 400000



        return timecost

    def db_optimizer_time(self, query):
        join_collapse_limit = "SET join_collapse_limit = 18"
        query = join_collapse_limit + ";EXPLAIN ANALYZE " + query + ";"
        cursor = self.conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        planning = [float(s) for s in rows[-2][0].split() if self.is_number(s)]
        execution = [float(s) for s in rows[-1][0].split() if self.is_number(s)]
        cursor.close()
        #return planning[0], execution[0]
        return execution[0]

    def get_query_time(self, query, force_order=False):
        join_collapse_limit = "SET join_collapse_limit = "
        join_collapse_limit += "1" if force_order else "18"
        query = join_collapse_limit + ";EXPLAIN ANALYZE " + query + ";"
        cursor = self.conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        planning = [float(s) for s in rows[-2][0].split() if self.is_number(s)]
        execution = [float(s) for s in rows[-1][0].split() if self.is_number(s)]
        cursor.close()
        return planning[0], execution[0]

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def construct_query(
        self, query_ast, join_ordering, attrs, joined_attrs, alias_to_relations, aliases
    ):

        relations_to_alias = {}

        # print(join_ordering)
        subq, alias = self.recursive_construct(
            join_ordering,
            attrs,
            joined_attrs,
            relations_to_alias,
            alias_to_relations,
            aliases,
        )
        if subq ==-1:
            return -1

        select_clause = utils.get_select_clause(query_ast, relations_to_alias, alias)
        where_clause = utils.get_where_clause(query_ast, relations_to_alias, alias)

        limit = ""
        if "limit" in query_ast:
            limit = " LIMIT " + str(query_ast["limit"])

        # print("\n\nRelations to aliases: ")
        # self.print_dict(relations_to_alias)

        query = select_clause + " FROM " + subq + where_clause + limit

        # print(query)
        self.counter = 0
        return query

    def recursive_construct(
        self,
        subtree,
        attrs,
        joined_attrs,
        relations_to_alias,
        alias_to_relations,
        aliases,
    ):

        if isinstance(subtree, str):
            return subtree, subtree

        left, left_alias = self.recursive_construct(
            subtree[0],
            attrs,
            joined_attrs,
            relations_to_alias,
            alias_to_relations,
            aliases,
        )
        if left == -1:
            return -1, -1
        right, right_alias = self.recursive_construct(
            subtree[1],
            attrs,
            joined_attrs,
            relations_to_alias,
            alias_to_relations,
            aliases,
        )
        if left == -1:
            return -1, -1

        new_alias = "J" + str(self.counter)
        relations_to_alias[left_alias] = new_alias
        relations_to_alias[right_alias] = new_alias
        self.counter += 1

        # print("\n\nAliases to relations: ")
        alias_to_relations[new_alias] = [left_alias, right_alias]
        # self.print_dict(alias_to_relations)

        # print("\n\nJoining subtrees: " + left_alias + " ⟕ " + right_alias)
        #gaojun
        if (left_alias, right_alias) not in joined_attrs:
            return -1, -1

        attr1, attr2 = joined_attrs[(left_alias, right_alias)]
        # print("\n\nJoined Attrs: ")
        # self.print_dict(joined_attrs)
        # print("Attrs: " + attr1 + " , " + attr2)

        if left == left_alias:
            left = aliases[left] + " AS " + left
        if right == right_alias:
            right = aliases[right] + " AS " + right

        clause = self.select_clause(
            alias_to_relations, left_alias, right_alias, attrs, aliases
        )

        subquery = (
            "( SELECT "
            + clause
            + " FROM "
            + left
            + " JOIN "
            + right
            + " on "
            + left_alias
            + "."
            + attr1
            + " = "
            + right_alias
            + "."
            + attr2
            + ") "
            + new_alias
        )

        self.update_joined_attrs((left_alias, right_alias), new_alias, joined_attrs)
        # print("\n\nUpdated Joined Attrs: ")
        # self.print_dict(joined_attrs)

        # print(subquery)

        return subquery, new_alias

    def update_joined_attrs(self, old_pair, new_alias, joined_attrs):
        # Optimize this maybe

        # Delete the two elements corresponding to the subtrees we joined
        # e.g. [A,B]->[id, id2], [B,A]->[id2, id]
        del joined_attrs[(old_pair[0], old_pair[1])]
        del joined_attrs[(old_pair[1], old_pair[0])]

        # Search for other entries with values from the old pair and update their name
        keys = list(joined_attrs.keys())
        for (t1, t2) in keys:

            (rel1, attr1) = (t1, joined_attrs[(t1, t2)][0])
            (rel2, attr2) = (t2, joined_attrs[(t1, t2)][1])

            if t1 == old_pair[0] or t1 == old_pair[1]:
                rel1 = new_alias
                attr1 = t1 + "_" + attr1

            if t2 == old_pair[0] or t2 == old_pair[1]:
                rel2 = new_alias
                attr2 = t2 + "_" + attr2

            if t1 != rel1 or t2 != rel2:
                del joined_attrs[(t1, t2)]
                joined_attrs[(rel1, rel2)] = (attr1, attr2)

    def select_clause(
        self, alias_to_relations, left_alias, right_alias, attrs, aliases
    ):

        # print("\n\nSelect Clause:\n")

        clause = []
        # relations_left = alias_to_relations[left_alias] ; relations_right = alias_to_relations[right_alias]
        # print(relations_left); print(relations_right)

        self.recursive_select_clause(
            clause, "", alias_to_relations, left_alias, attrs, left_alias, aliases
        )
        self.recursive_select_clause(
            clause, "", alias_to_relations, right_alias, attrs, right_alias, aliases
        )

        select_clause = ""
        for i in range(len(clause) - 1):
            select_clause += clause[i] + ", "
        select_clause += clause[len(clause) - 1]
        # print(select_clause)

        return select_clause

    def recursive_select_clause(
        self, clause, path, alias_to_relations, alias, attrs, base_alias, aliases
    ):

        # print(alias)
        rels = alias_to_relations[alias]
        if len(rels) > 1:
            for rel in rels:
                path1 = path + rel + "_"
                self.recursive_select_clause(
                    clause, path1, alias_to_relations, rel, attrs, base_alias, aliases
                )
        else:
            attributes = attrs[rels[0]]
            for attr in attributes:
                tmp = path + attr
                clause.append(base_alias + "." + tmp + " AS " + base_alias + "_" + tmp)

    def get_reward(self, query, phase):
        if phase == 1:
            return self.optimizer_cost(query, True)  # Get Cost Model's Estimate
        return self.get_query_time(query, True)[1]  # Get actual query-execution latency

    def print_dict(self, d):
        for key in d:
            print(str(key) + " -> " + str(d[key]))
        print("\n")
