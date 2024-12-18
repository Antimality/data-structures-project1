import unittest
from AVLTree import (
    AVLNode,
    AVLTree,
)


class TestAVLNode(unittest.TestCase):
    def test_node_initialization(self):
        node = AVLNode(10, "value")
        self.assertEqual(node.key, 10)
        self.assertEqual(node.value, "value")
        self.assertIsNone(node.left)
        self.assertIsNone(node.right)
        self.assertIsNone(node.parent)
        self.assertEqual(node.height, -1)

    def test_is_real_node(self):
        node = AVLNode(None, None)
        self.assertFalse(node.is_real_node())

        real_node = AVLNode(1, "real")
        self.assertTrue(real_node.is_real_node())

    def test_is_leaf_node(self):
        node = AVLNode(1, "leaf")
        node.add_virtual_children()
        self.assertTrue(node.is_leaf_node())

    def test_height_calculation(self):
        parent = AVLNode(10, "parent")
        left_child = AVLNode(5, "left")
        right_child = AVLNode(15, "right")
        parent.add_virtual_children()
        parent.insert_child(left_child, False)
        parent.insert_child(right_child, True)
        left_child.height = 0
        right_child.height = 1
        parent.calc_height()
        self.assertEqual(parent.height, 2)


class TestAVLTree(unittest.TestCase):
    def test_empty_tree(self):
        tree = AVLTree()
        self.assertIsNone(tree.get_root())
        self.assertEqual(tree.size(), 0)

    def test_single_insert(self):
        tree = AVLTree()
        node, dist, promotes = tree.insert(10, "ten")
        self.assertEqual(node.key, 10)
        self.assertEqual(tree.size(), 1)
        self.assertEqual(tree.get_root().key, 10)
        self.assertEqual(tree.max_node().key, 10)

    def test_multiple_inserts(self):
        tree = AVLTree()
        tree.finger_insert(10, "ten")
        tree.finger_insert(20, "twenty")
        tree.finger_insert(5, "five")
        self.assertEqual(tree.size(), 3)
        self.assertEqual(tree.max_node().key, 20)
        self.assertEqual(
            tree.avl_to_array(), [(5, "five"), (10, "ten"), (20, "twenty")]
        )

    def test_insert_and_rebalance(self):
        tree = AVLTree()
        tree.insert(10, "ten")
        tree.insert(20, "twenty")
        tree.insert(30, "thirty")  # Requires left rotation
        self.assertEqual(tree.get_root().key, 20)
        self.assertEqual(tree.get_root().left.key, 10)
        self.assertEqual(tree.get_root().right.key, 30)

    def test_delete_leaf(self):
        tree = AVLTree()
        tree.insert(10, "ten")
        node, _, _ = tree.insert(20, "twenty")
        tree.delete(node)
        self.assertEqual(tree.size(), 1)
        self.assertEqual(tree.max_node().key, 10)

    def test_delete_with_one_child(self):
        tree = AVLTree()
        tree.insert(10, "ten")
        tree.insert(20, "twenty")
        tree.delete(tree.get_root())
        self.assertEqual(tree.size(), 1)
        self.assertEqual(tree.max_node().key, 20)
        self.assertEqual(tree.get_root().key, 20)

    def test_delete_with_two_children(self):
        tree = AVLTree()
        tree.insert(10, "ten")
        node, _, _ = tree.insert(20, "twenty")
        tree.insert(30, "thirty")
        tree.delete(node)
        self.assertEqual(tree.size(), 2)
        self.assertEqual(tree.max_node().key, 30)

    def test_delete_root(self):
        tree = AVLTree()
        root, _, _ = tree.insert(10, "ten")
        tree.delete(root)
        self.assertEqual(tree.size(), 0)
        self.assertIsNone(tree.get_root())

    def test_search(self):
        tree = AVLTree()
        tree.insert(10, "ten")
        tree.insert(5, "five")
        tree.insert(15, "fifteen")
        node, dist = tree.search(15)
        self.assertEqual(node.key, 15)
        self.assertEqual(dist, 2)

    def test_finger_search(self):
        tree = AVLTree()
        tree.finger_insert(10, "ten")
        tree.finger_insert(20, "twenty")
        tree.finger_insert(30, "thirty")
        node, dist = tree.finger_search(20)
        self.assertEqual(node.key, 20)
        self.assertGreaterEqual(dist, 1)

    def test_avl_to_array(self):
        tree = AVLTree()
        tree.insert(10, "ten")
        tree.insert(5, "five")
        tree.insert(20, "twenty")
        self.assertEqual(
            tree.avl_to_array(), [(5, "five"), (10, "ten"), (20, "twenty")]
        )

    def test_join(self):
        tree1 = AVLTree()
        tree2 = AVLTree()
        tree1.insert(5, "five")
        tree2.insert(15, "fifteen")
        tree2.insert(20, "twenty")
        tree2.insert(17, "seventeen")
        tree1.join(tree2, 10, "ten")
        self.assertEqual(tree1.size(), 5)
        self.assertEqual(
            tree1.avl_to_array(),
            [
                (5, "five"),
                (10, "ten"),
                (15, "fifteen"),
                (17, "seventeen"),
                (20, "twenty"),
            ],
        )

    def test_split_root(self):
        tree = AVLTree()
        tree.insert(10, "ten")
        tree.insert(5, "five")
        tree.insert(15, "fifteen")
        left_tree, right_tree = tree.split(tree.get_root())
        self.assertEqual(left_tree.avl_to_array(), [(5, "five")])
        self.assertEqual(right_tree.avl_to_array(), [(15, "fifteen")])

    def test_split(self):
        tree = AVLTree()
        node, _, _ = tree.insert(10, "ten")
        tree.insert(5, "five")
        tree.insert(15, "fifteen")
        tree.insert(20, "twenty")
        tree.insert(17, "seventeen")
        tree.insert(15, "fifteen")
        tree.insert(30, "thirty")
        left_tree, right_tree = tree.split(node)
        self.assertEqual(left_tree.avl_to_array(), [(5, "five")])
        # FAILING AT JOIN
        self.assertEqual(
            right_tree.avl_to_array(),
            [(15, "fifteen"), (17, "seventeen"), (20, "twenty"), (30, "thirty")],
        )


if __name__ == "__main__":
    unittest.main()
