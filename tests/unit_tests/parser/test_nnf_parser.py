import pytest
from src.parser.nnf_parser import NNFParser, AndNode, OrNode, LiteralNode, TrueNode, FalseNode

# ----------------------------------------------------------------------------
# Test Cases for NNFParser
# ----------------------------------------------------------------------------

TEST_FILE_SIMPLE_AND = """
c A simple circuit representing (1 AND 2)
sdd 3
L 1 0 1
L 2 0 2
D 0 1 1 1 2
"""

TEST_FILE_OR_OF_ANDS = """
c Circuit for (1 AND 2) OR (3 AND -4)
sdd 7
L 1 0 1
L 2 0 2
L 3 0 3
L 4 0 -4
D 0 1 2 1 2 3 4 c Root OR node
"""

TEST_FILE_TRUE_FALSE = """
c Circuit for (1 AND TRUE) OR (2 AND FALSE)
sdd 6
L 1 0 1
T 2
L 3 0 2
F 4
D 0 1 2 1 2 3 4
"""

TEST_FILE_NEGATION = """
c Circuit representing a single negative literal -1
sdd 1
L 0 0 -1
"""

TEST_FILE_REUSE = """
c Test file for AndNode reuse
L 10 0 1
L 20 0 2
L 30 0 3
T 40
c OrNode '1' uses And(10, 20) and And(30, 40)
D 1 1 2 10 20 30 40
c OrNode '2' reuses And(10, 20) and adds a new And(10, 30)
D 2 1 2 10 20 10 30
"""

TEST_FILE_ALARM_BALANCED = """
c ids of sdd nodes start at 0
c sdd nodes appear bottom-up, children before parents
sdd 19
L 2 0 -1
L 3 2 2
L 4 0 1
T 5
D 1 1 2 2 3 4 5
L 7 4 -3
L 9 6 4
L 10 8 -5
L 11 6 -4
F 12
D 8 7 2 9 10 11 12
L 13 4 3
L 15 8 5
D 14 7 2 9 15 11 12
D 6 5 2 7 8 13 14
L 17 2 -2
D 16 1 2 2 17 4 12
D 18 7 2 11 10 9 12
D 0 3 2 1 6 16 18
"""

@pytest.fixture
def parser():
    """Provides a fresh NNFParser instance for each test."""
    return NNFParser()

@pytest.fixture
def create_test_file(tmp_path):
    """A factory fixture to create temporary test files."""
    def _create_file(filename, content):
        file_path = tmp_path / filename
        file_path.write_text(content)
        return str(file_path)
    return _create_file

def test_parse_simple_and(parser, create_test_file):
    """Tests parsing a simple AND circuit: (1 AND 2)."""
    file_path = create_test_file("simple_and.sdd", TEST_FILE_SIMPLE_AND)
    root = parser.parse(file_path)

    assert isinstance(root, OrNode)
    assert len(root.children) == 1
    
    and_node = root.children[0]
    assert isinstance(and_node, AndNode)
    assert len(and_node.children) == 2

    child1, child2 = and_node.children
    assert isinstance(child1, LiteralNode)
    assert child1.literal == 1 and not child1.negated
    
    assert isinstance(child2, LiteralNode)
    assert child2.literal == 2 and not child2.negated

def test_parse_or_of_ands(parser, create_test_file):
    """Tests a DNF structure: (1 AND 2) OR (3 AND -4)."""
    file_path = create_test_file("or_of_ands.sdd", TEST_FILE_OR_OF_ANDS)
    root = parser.parse(file_path)

    assert isinstance(root, OrNode)
    assert len(root.children) == 2

    and1 = root.children[0]
    assert isinstance(and1, AndNode)
    assert isinstance(and1.children[0], LiteralNode) and and1.children[0].literal == 1
    assert isinstance(and1.children[1], LiteralNode) and and1.children[1].literal == 2

    and2 = root.children[1]
    assert isinstance(and2, AndNode)
    assert isinstance(and2.children[0], LiteralNode) and and2.children[0].literal == 3
    assert isinstance(and2.children[1], LiteralNode) and and2.children[1].literal == 4 and and2.children[1].negated

def test_parse_true_false(parser, create_test_file):
    """Tests parsing a circuit with TRUE and FALSE terminal nodes."""
    file_path = create_test_file("true_false.sdd", TEST_FILE_TRUE_FALSE)
    root = parser.parse(file_path)

    assert isinstance(root, OrNode)
    assert len(root.children) == 2

    and1 = root.children[0]
    assert isinstance(and1.children[0], LiteralNode) and and1.children[0].literal == 1
    assert isinstance(and1.children[1], TrueNode)

    and2 = root.children[1]
    assert isinstance(and2.children[0], LiteralNode) and and2.children[0].literal == 2
    assert isinstance(and2.children[1], FalseNode)

def test_parse_negation(parser, create_test_file):
    """Tests parsing a single negated literal."""
    file_path = create_test_file("negation.sdd", TEST_FILE_NEGATION)
    root = parser.parse(file_path)

    assert isinstance(root, LiteralNode)
    assert root.literal == 1
    assert root.negated is True

def test_file_not_found(parser):
    """Tests that the parser returns None for a non-existent file."""
    root = parser.parse("non_existent_file.sdd")
    assert root is None

def test_and_node_reuse_with_custom_sdd(parser, create_test_file):
    """
    Tests if AndNodes are reused when parsing a custom SDD file content.
    This test creates a simple, explicit scenario for node reuse.
    """
    file_path = create_test_file("test_reuse.sdd", TEST_FILE_REUSE)
    parser.parse(file_path)

    assert len(parser.nodes) == 9
    assert "and_10_20" in parser.nodes
    assert "and_30_40" in parser.nodes
    assert "and_10_30" in parser.nodes
    assert len(parser.and_node_cache) == 3

    # ---- Verify that the AndNode object is the same instance ----
    or_node_1 = parser.nodes['1']
    or_node_2 = parser.nodes['2']
    assert isinstance(or_node_1, OrNode)
    assert isinstance(or_node_2, OrNode)

    # Get the first child of each OrNode, which should be the reused AndNode
    and_node_from_or1 = or_node_1.children[0]
    and_node_from_or2 = or_node_2.children[0]
    assert and_node_from_or1 is and_node_from_or2

    # Verify the other AndNodes are different objects
    and_node_2_from_or1 = or_node_1.children[1]
    and_node_2_from_or2 = or_node_2.children[1]
    assert and_node_2_from_or1 is not and_node_2_from_or2

def test_and_node_reuse_in_alarm_file(parser, create_test_file):
    """
    Tests AndNode reuse on the 'alarm_balanced.sdd' file.
    It specifically checks for the reuse of the (prime='11', sub='12') pair.
    """
    file_path = create_test_file("alarm_balanced.sdd", TEST_FILE_ALARM_BALANCED)
    parser.parse(file_path)

    assert len(parser.and_node_cache) == 13

    or_node_8 = parser.nodes['8']
    or_node_14 = parser.nodes['14']
    assert isinstance(or_node_8, OrNode)
    assert isinstance(or_node_14, OrNode)

    # Find the AndNode corresponding to ('11', '12') in OrNode '8'
    and_node_from_or8 = None
    for child in or_node_8.children:
        if isinstance(child, AndNode) and child.id == "and_11_12":
            and_node_from_or8 = child
            break
    
    # Find the AndNode corresponding to ('11', '12') in OrNode '14'
    and_node_from_or14 = None
    for child in or_node_14.children:
        if isinstance(child, AndNode) and child.id == "and_11_12":
            and_node_from_or14 = child
            break

    # Assert that both were found and that they are the exact same object.
    assert and_node_from_or8 is not None, "AndNode for ('11','12') not found in OrNode '8'"
    assert and_node_from_or14 is not None, "AndNode for ('11','12') not found in OrNode '14'"
    assert and_node_from_or8 is and_node_from_or14
