import wtm_runner


def test_segment_to_dict_tuple():
    d = wtm_runner._segment_to_dict((0.0, 4.96, "hello there"))
    assert d["start"] == 0.0
    assert d["end"] == 4.96
    assert d["text"] == "hello there"


def test_segment_to_dict_open_end():
    d = wtm_runner._segment_to_dict((10.0, None, "tail"))
    assert d["start"] == 10.0
    assert d["end"] > d["start"]
    assert d["text"] == "tail"
