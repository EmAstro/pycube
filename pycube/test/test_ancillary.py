from pycube.ancillary import cleaning_lists


def test_cleaning_lists():
    """Test that the list creator is working

    """
    assert isinstance(cleaning_lists.from_element_to_list(0, element_type=int), list), r'The code failed in generating a list'
