"""Module which describes topic of  function definition   and  functional dependency definition."""

"""
Function - is a rule which take elements from one  set and  connect them to elements of other set.
Important to understand that any machine learning model is a function, so that's mean that any  machine
learning model is a rule which  takes  elements  from one  set  and  connect the to elements of  other  set.
Composed funtion - is  a function which as an argument takes another  function.

Suppose we  define  function:  F: X -> Y as a rule which takes elements from set X and connects them  to   elemetns
of set Y.

Definition:
Function minimum: Function  minimul is  such element y' which is result of rule connection of element x` that
any other elemten y which is result of relu connection of element x is greater.
F: X -> Y
y` = f(x`) -> function minimum if
for any x є X y = f(x)  >= y` = f(x`)

Definition:
Function  maximum: Function  maximum is  such element  y`  which is resulf of rule connection of element  x`
that any other element y  which is  reuslt of fule connection of element x is smaller
F: X -> Y
y` = f(x`) ->  function maximu if
for any x є X y = f(x) <=  y` =  f(x`)
"""