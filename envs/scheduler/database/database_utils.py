def get_alias(attr, relations_to_alias, alias):

    tmp = attr.split(".")
    relname = tmp[0]
    attrname = tmp[1]

    while relname in relations_to_alias:
        attrname = relname + "_" + attrname
        relname = relations_to_alias[relname]

    return alias + "." + attrname




def get_select_clause(query_ast, relations_to_alias, alias):

    # Construct SELECT clause

    select_operator_map = {
        "min": "MIN",
        "max": "MAX",
    }  # to be filled with other possible values

    select_stmt = query_ast["select"]
    select = []

    if not isinstance(select_stmt, (list,)):
        select_stmt = [select_stmt]

    for v in select_stmt:

        val = v["value"]
        name = ""

        if not isinstance(val, str):
            key = list(val.keys())[0]
            val = (
                select_operator_map[key]
                + "("
                + get_alias(val[key], relations_to_alias, alias)
                + ")"
            )
        else:
            val = get_alias(val, relations_to_alias, alias)

        if "name" in v:
            name = " AS " + v["name"]

        select.append((val, name))

    select_clause = "SELECT "
    for i in range(len(select) - 1):
        select_clause += select[i][0] + select[i][1] + ", "
    select_clause += select[len(select) - 1][0] + select[len(select) - 1][1]

    # print(select_clause)
    return select_clause


def construct_stmt(stmt, operator_map, relations_to_alias, alias):

    # print(stmt)
    key = list(stmt.keys())[0]

    if key == "and" or key == "or":  # Need to go deeper

        return (
            "( "
            + where_and_or(stmt, operator_map, relations_to_alias, alias)
            + " )"
        )
    else:
        if key == "between":

            if isinstance(stmt[key][1], dict):
                left = "'" + stmt[key][1]["literal"] + "'"
            else:
                left = str(stmt[key][1])

            if isinstance(stmt[key][2], dict):
                right = "'" + stmt[key][2]["literal"] + "'"
            else:
                right = str(stmt[key][2])

            rvalue = left + " AND " + right

        elif isinstance(stmt[key][1], dict):  # Dict (Naively assuming it's a literal)

            lit = stmt[key][1]["literal"]

            if isinstance(lit, list):
                rvalue = " ( "
                for i in range(len(lit) - 1):
                    rvalue = rvalue + "'" + lit[i] + "', "
                rvalue = rvalue + "'" + lit[len(lit) - 1] + "' ) "

            elif key == "in":
                rvalue = "( '" + lit + "' )"

            else:
                rvalue = "'" + lit + "'"

        elif isinstance(stmt[key][1], int):  # Integer
            rvalue = str(stmt[key][1])

        else:
            rvalue = get_alias(stmt[key][1], relations_to_alias, alias)

        return (
            get_alias(stmt[key][0], relations_to_alias, alias)
            + " "
            + operator_map[key]
            + " "
            + rvalue
        )


def where_and_or(where_ast, operator_map, relations_to_alias, alias):

    where_and_clause = ""
    if "and" in where_ast:
        and_stmt = where_ast["and"]
        where_and = []
        for v in and_stmt:
            if not (
                "eq" in v
                and isinstance(v["eq"][0], str)
                and isinstance(v["eq"][1], str)
            ):  # if not a joining
                where_and.append(construct_stmt(v, operator_map, relations_to_alias, alias))

        size = len(where_and)
        if size > 0:
            for i in range(size - 1):
                where_and_clause += where_and[i] + " AND \n"
            where_and_clause += where_and[size - 1]

    where_or_clause = ""
    if "or" in where_ast:
        or_stmt = where_ast["or"]
        where_or = []
        for v in or_stmt:
            if not (
                "eq" in v
                and isinstance(v["eq"][0], str)
                and isinstance(v["eq"][1], str)
            ):  # if not a joining
                where_or.append(construct_stmt(v, operator_map, relations_to_alias, alias))

        size = len(where_or)
        if size > 0:
            for i in range(size - 1):
                where_or_clause += where_or[i] + " OR \n"
            where_or_clause += where_or[size - 1]

    return where_and_clause + where_or_clause


def get_where_clause(query_ast, relations_to_alias, alias):

    # Construct WHERE clause

    operator_map = {
        "eq": "=",
        "neq": "!=",
        "gt": ">",
        "lt": "<",
        "gte": ">=",
        "lte": "<=",
        "like": "LIKE",
        "nlike": "NOT LIKE",
        "in": "IN",
        "between": "BETWEEN"
    }  # to be filled with other possible values

    where_ast = query_ast["where"]
    where_clause = where_and_or(where_ast, operator_map, relations_to_alias, alias)

    if where_clause != "":
        return " \nWHERE \n" + where_clause

    else:
        return ""
