"""6.009 Lab 5 -- Boolean satisfiability solving"""

import sys

sys.setrecursionlimit(10000)


# NO ADDITIONAL IMPORTS


def cnf_equal(actual, expected):
    """
    Test function that compares CNFs.
    :param actual: The actual output.
    :param expected: The expected CNF.
    :return: True or False.

    >>> cnf_equal([[('b', True), ('a', True)]], [('b', True)])
    False

    >>> cnf_equal([[False, ('a', True)]], [[('a', True), False]])
    True
    """
    if type(actual) == type(expected) == bool:
        return actual == expected

    return frozenset(clause if type(clause) == bool else frozenset(clause) for clause in actual) == \
           frozenset(clause if type(clause) == bool else frozenset(clause) for clause in expected)


def simplify_clause(clause):
    """
    Simplify a CNF clause.
    :param clause: A non-empty CNF clause, which is a list of (variable, value) or bool
    :return: The simplified CNF clause, which is either a list of clauses or a bool

    >>> actual = simplify_clause([('b', True), ('b', True)])
    >>> expected = [('b', True)]
    >>> cnf_equal(actual, expected)
    True

    >>> actual = simplify_clause([('c', True), ('a', True)])
    >>> expected = [('a', True), ('c', True)]
    >>> cnf_equal(actual, expected)
    True

    >>> actual = simplify_clause([False, ('c', True)])
    >>> expected = [('c', True)]
    >>> cnf_equal(actual, expected)
    True

    >>> actual = simplify_clause([False, ('c', True), ('c', False)])
    >>> expected = [('c', False), ('c', True)]
    >>> cnf_equal(actual, expected)
    True

    >>> actual = simplify_clause([True, ('b', True)])
    >>> expected = True
    >>> cnf_equal(actual, expected)
    True

    >>> actual = simplify_clause([False])
    >>> expected = False
    >>> cnf_equal(actual, expected)
    True
    """
    literals = []

    for literal in clause:
        if type(literal) == bool:
            if literal:
                # If we have a True literal, the entire clause is True.
                return True
            else:
                # We have a False literal, so simply ignore it and continue.
                continue
        else:
            literals.append(literal)

    # If the set is empty, return False.
    return literals if len(literals) > 0 else False


def simplify_formula(formula):
    """
    Simplify a CNF formula.
    :param formula: A CNF formula, which is a list of clauses who must be non-empty
    :return: The simplified CNF formula, which is either a list of clauses or a bool

    >>> actual = simplify_formula([])
    >>> expected = True
    >>> cnf_equal(actual, expected)
    True

    >>> actual = simplify_formula([
    ...     [('b', True), ('b', True)],
    ...     [('b', True)],
    ...     [('b', True)],
    ...     [('a', True), ('a', True)],
    ...     [('c', True), ('a', True)]
    ... ])
    >>> expected = [[('a', True)], [('a', True), ('c', True)], [('b', True)]]
    >>> cnf_equal(actual, expected)
    True
    """
    clauses = []

    for clause in formula:
        simplified_clause = simplify_clause(clause)
        if type(simplified_clause) == bool:
            if not simplified_clause:
                # If we have a False clause, the entire formula is False.
                return False
            else:
                # We have a True clause, so simply ignore it and continue.
                continue
        else:
            clauses.append(simplified_clause)

    # If the set is empty, return True.
    return clauses if len(clauses) > 0 else True


def replace(formula, variable, value):
    """
    Replace a particular variable with a given value in a CNF formula.
    Does not modify the original formula.
    :param formula: The CNF formula, which must not contain any constants.
    :param variable: The variable to replace.
    :param value: The value to replace it with.
    :return: A generator yielding clause generators which themselves yield literals.

    >>> actual = replace([[('b', True)], [('b', False)]], 'b', False)
    >>> expected = [[False], [True]]
    >>> cnf_equal(actual, expected)
    True

    >>> actual = replace([[('b', True), ('a', False)]], 'a', True)
    >>> expected = [[('b', True), False]]
    >>> cnf_equal(actual, expected)
    True
    """
    for clause in formula:
        yield (literal[1] == value if literal[0] == variable else literal for literal in clause)


def find_unit_literal(formula):
    """
    Locate the first unit clause in a formula if one exists.
    :param formula: A CNF formula
    :return: The literal in the unit cause, or None

    >>> find_unit_literal([[('a', True), ('b', False), ('c', True)]])

    >>> find_unit_literal([[('a', True)], [('a', False)]])
    ('a', True)
    """
    for clause in formula:
        if len(clause) == 1:
            return clause[0]

    return None


def add_assignment(assignments, variable, value):
    """
    Add an assignment, using a copy of the original dictionary.
    :param assignments: The original assignments.
    :param variable: The variable to assign.
    :param value: The value to assign.
    :return: A new dictionary of assignments.

    >>> add_assignment({}, 'a', True)
    {'a': True}
    """
    new_assignment = assignments.copy()
    new_assignment[variable] = value
    return new_assignment


def satisfying_assignment(formula):
    """
    Find a satisfying assignment for a given CNF formula.
    Not guaranteed to find the minimally-constrained assignment.
    :param formula: A CNF formula, which is a list of clauses who must be non-empty
    :return: A minimally-constrained assignment if one exists, or None otherwise.

    >>> satisfying_assignment([[('a', True)], [('a', False)]])

    >>> satisfying_assignment([])
    {}

    >>> assignment = satisfying_assignment([[('a', True), ('b', False), ('c', True)]])
    >>> formula = [[('a', True), ('b', False), ('c', True)]]
    >>> for key, value in assignment.items():
    ...     formula = simplify_formula(replace(formula, key, value))
    ...     if type(formula) == bool and formula:
    ...         break
    >>> formula
    True
    """

    def sat(formula, assignments):
        if type(formula) == bool:
            return assignments if formula else None

        unit_literal = find_unit_literal(formula)
        if unit_literal is not None:
            # There is a unit literal, so we only need to check one assignment.
            variable, value = unit_literal
            new_formula = replace(formula, variable, value)
            return sat(simplify_formula(new_formula), add_assignment(assignments, variable, value))

        first_literal = formula[0][0]
        variable, value = first_literal

        # Test if variable = value will work.
        true_formula = replace(formula, variable, value)
        true_sat = sat(simplify_formula(true_formula), add_assignment(assignments, variable, value))
        if true_sat is not None:
            # Here, variable = value solved the problem.
            return true_sat

        # Next, try variable = ~value.
        false_formula = replace(formula, variable, not value)
        return sat(simplify_formula(false_formula), add_assignment(assignments, variable, not value))

    return sat(simplify_formula(formula), {})


def combination(l, n):
    """
    Create combinations of elements in a list that are of length n.
    :param l: The given list.
    :param n: The number of elements to choose.
    :return: A list of all combinations of length n.

    >>> sorted(combination([1, 2, 3], 2))
    [[1, 2], [1, 3], [2, 3]]

    >>> sorted(combination([1, 2], 2))
    [[1, 2]]

     >>> sorted(combination([1, 2], 0))
     [[]]
    """
    if n == 0:
        return [[]]
    return [[l[i]] + c for i in range(len(l) - n + 1) for c in combination(l[i + 1:], n - 1)]


def boolify_scheduling_problem(student_preferences, session_capacities):
    """
    Convert a quiz-room-scheduling problem into a Boolean formula.
    We assume no student or session names contain underscores.
    :param student_preferences: a dictionary mapping a student name (string) to a set of session names (strings) that
                                work for that student
    :param session_capacities: a dictionary mapping each session name to a positive integer for how many students can
                               fit in that session
    :return: a CNF formula encoding the scheduling problem, as per the lab write-up

    >>> rules = boolify_scheduling_problem(
    ...     {'Alice': {'cave', 'tunnel'}, 'Bob': {'tunnel'}},
    ...     {'cave': 1, 'tunnel': 1}
    ... )
    >>> sorted(key for key, value in satisfying_assignment(rules).items() if value)
    ['Alice_cave', 'Bob_tunnel']
    """

    def get_student_session(student, session):
        return student + '_' + session

    students = list(student_preferences.keys())
    sessions = list(session_capacities.keys())

    # Rule 1 ensures each person is assigned to exactly one location.
    rule1 = []
    if len(sessions) > 1:
        # We only care if there is more than one session.
        session_combinations = combination(sessions, 2)
        for student in students:
            for session_combination in session_combinations:
                clause = [(get_student_session(student, session), False) for session in session_combination]
                rule1.append(clause)

    # Rule 2 ensures each person will only be assigned their preferences.
    rule2 = []
    for student in students:
        clause = [(get_student_session(student, session), True) for session in student_preferences[student]]
        rule2.append(clause)

    # Rule 3 ensures assignment does not exceed session carrying capacity.
    rule3 = []
    for session in sessions:
        if session_capacities[session] < len(students):
            # We only care if there is less room that the number of students.
            student_combinations = combination(students, session_capacities[session] + 1)
            for student_combination in student_combinations:
                clause = [(get_student_session(student, session), False) for student in student_combination]
                rule3.append(clause)

    return rule1 + rule2 + rule3
