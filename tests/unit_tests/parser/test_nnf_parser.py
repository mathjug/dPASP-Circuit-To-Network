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

    # Root should be an OR node with one child (which is the AND node)
    assert isinstance(root, OrNode)
    assert len(root.children) == 1
    
    # The child of the root OR is the implicit AND node
    and_node = root.children[0]
    assert isinstance(and_node, AndNode)
    assert len(and_node.children) == 2

    # Check the children of the AND node
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

    # Check first AND branch: (1 AND 2)
    and1 = root.children[0]
    assert isinstance(and1, AndNode)
    assert isinstance(and1.children[0], LiteralNode) and and1.children[0].literal == 1
    assert isinstance(and1.children[1], LiteralNode) and and1.children[1].literal == 2

    # Check second AND branch: (3 AND -4)
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

    # Check first branch: (1 AND TRUE)
    and1 = root.children[0]
    assert isinstance(and1.children[0], LiteralNode) and and1.children[0].literal == 1
    assert isinstance(and1.children[1], TrueNode)

    # Check second branch: (2 AND FALSE)
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
    """Tests that the parser raises FileNotFoundError for a non-existent file."""
    with pytest.raises(FileNotFoundError):
        parser.parse("non_existent_file.sdd")
