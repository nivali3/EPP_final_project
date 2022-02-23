def create_dummy(dt, *treatment):
    for i in treatment:
        dt['dummy1'] += (dt['treatment']==i).astype(int)

    return dt


# usage
# treat = ['t1.1', 't1.2', 't1.3']

# dt_new = create_dummy(dt, 't1.1', 't1.2', 't1.3')