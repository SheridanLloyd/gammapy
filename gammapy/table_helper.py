def del_ULs(ob_table):
    del_rows=[]
    for row in ob_table:
        if row['is_ul']:
            del_rows.append(row.index)
    ob_table.remove_rows(del_rows)
    return ob_table

def del_fluxes_below_limit(ob_table,flux_limit):
    del_rows=[]
    for row in ob_table:
        if row['e2dnde']<flux_limit:
            del_rows.append(row.index)
    ob_table.remove_rows(del_rows)
    return ob_table
