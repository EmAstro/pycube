from pycube import msgs


def test_indent():
    """Test for from pycube.msgs

    The code checks if the msgs.indent() returns a 13 x spaces string
    """
    assert msgs.indent() == '             ', r'Issue with generating the indent'
