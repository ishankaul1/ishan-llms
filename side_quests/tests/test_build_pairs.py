from side_quests.word_2_vec import _build_pairs


def test_simple_range():
    lst = [0, 1, 2, 3, 4, 5]
    window = 1

    assert _build_pairs(lst, window) == [
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
        (3, 4),
        (4, 3),
        (4, 5),
        (5, 4),
    ]


def test_wider_range():
    lst = [0, 1, 2, 3, 4, 5]
    window = 2

    assert _build_pairs(lst, window) == [
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 3),
        (2, 4),
        (3, 1),
        (3, 2),
        (3, 4),
        (3, 5),
        (4, 2),
        (4, 3),
        (4, 5),
        (5, 3),
        (5, 4),
    ]
