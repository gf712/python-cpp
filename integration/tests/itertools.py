import itertools


def test_islice():
    assert [x for x in itertools.islice("ABCDEFG", 2)] == ["A", "B"]
    assert [x for x in itertools.islice("ABCDEFG", 2, 4)] == ["C", "D"]
    assert [x for x in itertools.islice("ABCDEFG", 2, None)] == [
        "C",
        "D",
        "E",
        "F",
        "G",
    ]
    assert [x for x in itertools.islice("ABCDEFG", 0, None, 2)] == ["A", "C", "E", "G"]


test_islice()


def test_permutations():
    assert [x for x in itertools.permutations(range(3))] == [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]
    assert [x for x in itertools.permutations(range(3), 2)] == [
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
    ]
    assert [x for x in itertools.permutations("ABCD", 2)] == [
        ("A", "B"),
        ("A", "C"),
        ("A", "D"),
        ("B", "A"),
        ("B", "C"),
        ("B", "D"),
        ("C", "A"),
        ("C", "B"),
        ("C", "D"),
        ("D", "A"),
        ("D", "B"),
        ("D", "C"),
    ]


test_permutations()


def test_product():
    assert [x for x in itertools.product("ABCD", "xy", repeat=0)] == []

    assert [x for x in itertools.product("ABCD", "xy", repeat=1)] == [
        ("A", "x"),
        ("A", "y"),
        ("B", "x"),
        ("B", "y"),
        ("C", "x"),
        ("C", "y"),
        ("D", "x"),
        ("D", "y"),
    ]

    assert [x for x in itertools.product("ab", range(3))] == [
        ("a", 0),
        ("a", 1),
        ("a", 2),
        ("b", 0),
        ("b", 1),
        ("b", 2),
    ]
    # assert [x for x in itertools.product("ABCD", "xy", repeat=2)] == [
    #     ("A", "x", "A", "x"),
    #     ("A", "x", "A", "y"),
    #     ("A", "x", "B", "x"),
    #     ("A", "x", "B", "y"),
    #     ("A", "x", "C", "x"),
    #     ("A", "x", "C", "y"),
    #     ("A", "x", "D", "x"),
    #     ("A", "x", "D", "y"),
    #     ("A", "y", "A", "x"),
    #     ("A", "y", "A", "y"),
    #     ("A", "y", "B", "x"),
    #     ("A", "y", "B", "y"),
    #     ("A", "y", "C", "x"),
    #     ("A", "y", "C", "y"),
    #     ("A", "y", "D", "x"),
    #     ("A", "y", "D", "y"),
    #     ("B", "x", "A", "x"),
    #     ("B", "x", "A", "y"),
    #     ("B", "x", "B", "x"),
    #     ("B", "x", "B", "y"),
    #     ("B", "x", "C", "x"),
    #     ("B", "x", "C", "y"),
    #     ("B", "x", "D", "x"),
    #     ("B", "x", "D", "y"),
    #     ("B", "y", "A", "x"),
    #     ("B", "y", "A", "y"),
    #     ("B", "y", "B", "x"),
    #     ("B", "y", "B", "y"),
    #     ("B", "y", "C", "x"),
    #     ("B", "y", "C", "y"),
    #     ("B", "y", "D", "x"),
    #     ("B", "y", "D", "y"),
    #     ("C", "x", "A", "x"),
    #     ("C", "x", "A", "y"),
    #     ("C", "x", "B", "x"),
    #     ("C", "x", "B", "y"),
    #     ("C", "x", "C", "x"),
    #     ("C", "x", "C", "y"),
    #     ("C", "x", "D", "x"),
    #     ("C", "x", "D", "y"),
    #     ("C", "y", "A", "x"),
    #     ("C", "y", "A", "y"),
    #     ("C", "y", "B", "x"),
    #     ("C", "y", "B", "y"),
    #     ("C", "y", "C", "x"),
    #     ("C", "y", "C", "y"),
    #     ("C", "y", "D", "x"),
    #     ("C", "y", "D", "y"),
    #     ("D", "x", "A", "x"),
    #     ("D", "x", "A", "y"),
    #     ("D", "x", "B", "x"),
    #     ("D", "x", "B", "y"),
    #     ("D", "x", "C", "x"),
    #     ("D", "x", "C", "y"),
    #     ("D", "x", "D", "x"),
    #     ("D", "x", "D", "y"),
    #     ("D", "y", "A", "x"),
    #     ("D", "y", "A", "y"),
    #     ("D", "y", "B", "x"),
    #     ("D", "y", "B", "y"),
    #     ("D", "y", "C", "x"),
    #     ("D", "y", "C", "y"),
    #     ("D", "y", "D", "x"),
    #     ("D", "y", "D", "y"),
    # ]


test_product()
