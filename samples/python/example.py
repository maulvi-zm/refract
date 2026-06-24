DEFAULT_TAX_RATE = 7  # ALL_CAPS constant: 7 must NOT be flagged as magic here


def helper(value):
    return value + 1


def compute_order_total(quantity):
    # magic numbers (42, 7) and a deliberately overlong identifier below
    running_accumulated_order_total = 0
    for index in range(quantity):
        if index % 2 == 0 and index > 0:
            running_accumulated_order_total += index * 7
        else:
            running_accumulated_order_total += 42
    helper(running_accumulated_order_total)
    return running_accumulated_order_total


def caller():
    return compute_order_total(10)


def long_function(n):
    # more than 20 statements -> long_method
    a = n
    b = a + 3
    c = b + 4
    d = c + 5
    e = d + 6
    f = e + 8
    g = f + 9
    h = g + 11
    i = h + 12
    j = i + 13
    k = j + 14
    m = k + 15
    o = m + 16
    p = o + 17
    q = p + 18
    r = q + 19
    s = r + 20
    t = s + 21
    u = t + 22
    v = u + 23
    w = v + 24
    return w
