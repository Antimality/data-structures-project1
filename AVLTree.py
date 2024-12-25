# id1: 212825715
# name1: Yonatan Nitzan
# username1: yonatann2
# id2: 20816055
# name2: Ortal Simany
# username2: ortalsimany
from typing import Any

"""A class representing a node in an AVL tree"""


class AVLNode(object):
    """Constructor. Can make a virtual node by not passing any parameters.

    @type key: int
    @param key: key of your node
    @type value: string
    @param value: data of your node
    """

    def __init__(self, key=None, value=None):
        self.key: int = key
        self.value = value
        self.left: AVLNode = None
        self.right: AVLNode = None
        self.parent: AVLNode = None
        self.height: int = -1

    """returns whether self is not a virtual node 

	@rtype: bool
	@returns: False if self is a virtual node, True otherwise.
	"""

    def is_real_node(self) -> bool:
        return self.key is not None

    """Returns whether self is a leaf node
    
    @rtype: bool
    @returns: True if self has only virtual children
    """

    def is_leaf_node(self) -> bool:
        return not self.left.is_real_node() and not self.right.is_real_node()

    """Returns whether self is the left child of its parent
    
    @rtype: bool
    @returns: False if self has no parent or if it is not its left child, True otherwise
    """

    def is_left_child(self) -> bool:
        return self == self.parent.left if self.parent is not None else False

    """Returns whether self is the right child of its parent
    
    @rtype: bool
    @returns: False if self has no parent or if it is not its right child, True otherwise
    """

    def is_right_child(self) -> bool:
        return self == self.parent.right if self.parent is not None else False

    """Returns the balance factor of self
    
    @rtype: int
    @returns: The difference between the heights of self's left and right children
    """

    def get_balance_factor(self) -> int:
        return self.left.height - self.right.height

    """Updates the height of self according to its children
    The height should be the max height between the children plus one
    """

    def calc_height(self):
        self.height = max(self.left.height, self.right.height) + 1

    """Add a child to self and update the parent field of the child
    
    @type child: AVLNode
    @param child: The child to add
    @type right: bool
    @param right: True if adding right child, False if left
    """

    def insert_child(self, child: "AVLNode", right: bool):
        if right:
            self.right = child
        else:
            self.left = child
        child.parent = self

    """Add a right child to self and update the parent field of the child
    
    @type child: AVLNode
    @param child: The child to add
    """

    def insert_right(self, child: "AVLNode"):
        self.insert_child(child, True)

    """Add a left child to self and update the parent field of the child
    
    @type child: AVLNode
    @param child: The child to add
    """

    def insert_left(self, child: "AVLNode"):
        self.insert_child(child, False)

    """Adds a virtual child where self has none"""

    def add_virtual_children(self):
        if self.right is None:
            self.insert_right(AVLNode())
        if self.left is None:
            self.insert_left(AVLNode())

    """Replace the other node with self
    Height not updated
    
    @type other: AVLNode
    @param other: The node to replace
    """

    def replace_child(self, other: "AVLNode"):
        # Update self
        self.parent = other.parent
        # Update parent
        if other.is_left_child():
            self.parent.left = self
        if other.is_right_child():
            self.parent.right = self

    """Replace a virtual node with self and set height to 0
    
    @type other: AVLNode
    @param other: The node to replace
    """

    def create_leaf(self, spot: "AVLNode"):
        self.height = 0
        self.replace_child(spot)

    """Represent self as a key, value pair"""

    def __str__(self) -> str:
        return f"({self.key}, {self.value})"


"""
A class implementing an AVL tree.
"""


class AVLTree(object):
    """
    Constructor. Given node, initiate a tree with node as its root

    @type node: AVLNode
    @param node: A node that is to become the root of the tree
    """

    def __init__(self, node: AVLNode = None):
        self.root: AVLNode = None
        self.t_size: int = 0
        self.max: AVLNode = None
        if node is not None:
            self.tree_from_root(node)

    """ Construct a tree from an existing node (used for split & join)
    The new tree will have the given node as its root, but the size will not be updated.
    It was said on the forum that this is allowed.
    
    @type node: AVLNode
    @param node: A node that is to become the root of the tree
    """

    def tree_from_root(self, node: AVLNode):
        # Ignore a tree that is virtually empty
        if not node.is_real_node():
            return
        # Detach tree from its parent
        node.parent = None
        self.root = node
        """ The following would add to split complexity. Max will be updated when needed.
        self.t_size = len(self.in_order(node)) # Updating size would add complexity
        self.max = self.root
        while self.max.right.is_real_node():
            self.max = self.max.right
        """

    """searches for a node in the dictionary corresponding to the key (starting at the root)
    Complexity: O(log(n))
		
	@type key: int
	@param key: a key to be searched
	@rtype: (AVLNode,int)
	@returns: a tuple (x,e) where x is the node corresponding to key (or None if not found),
	and e is the number of edges on the path between the starting node and ending node+1.
	"""

    def search(self, key: int) -> tuple[AVLNode | None, int]:
        return self.search_result_wrapper(self.search_from(key, self.root))

    """searches for a node in the dictionary corresponding to the key, starting at the max
    Complexity: O(log(n))
		
	@type key: int
	@param key: a key to be searched
	@rtype: (AVLNode,int)
	@returns: a tuple (x,e) where x is the node corresponding to key (or None if not found),
	and e is the number of edges on the path between the starting node and ending node+1.
	"""

    def finger_search(self, key: int) -> tuple[AVLNode | None, int]:
        return self.search_result_wrapper(self.search_from_max(key))

    """Wrapper Function for search_from that returns in the format they want
    Receives the result of a search_from function and returns the same,
    but if the node is virtual it replaces it with None.
    
    @type tup: (AVLNode, int)
    @param tup: The result of a search_from function
    @rtype: (AVLNode|None, int)
    @returns: a tuple (x,e) where x is the node corresponding to key (or None if not found),
	and e is the number of edges on the path between the starting node and ending node+1.
    """

    def search_result_wrapper(
        self, tup: tuple[AVLNode, int]
    ) -> tuple[AVLNode | None, int]:
        node = tup[0]
        dist = tup[1]
        return node if node.is_real_node() else None, dist

    """Search for a key beginning from the given node
    Complexity: O(log(n))
    
    @type key: int
	@param key: a key to be searched
	@type dist: int
    @param dist: the distance to start counting from
    @rtype: (AVLNode,int)
	@returns: a tuple (x,e) where x is the node corresponding to key,
    or the virtual node where it should have been,
	and e is the number of edges on the path between the starting node and ending node+1.
	"""

    def search_from(
        self, key: int, node: AVLNode, dist: int = 1
    ) -> tuple[AVLNode, int]:
        while node.key != key and node.is_real_node():
            node = node.right if key > node.key else node.left
            dist += 1

        # If no node was found, only include the search distance within real nodes
        return node, dist if node.is_real_node() else dist - 1

    """Search for a key beginning from the max node
    Complexity: O(log(n))
    
    @type key: int
	@param key: a key to be searched
    @rtype: (AVLNode,int)
	@returns: a tuple (x,e) where x is the node corresponding to key,
    or the virtual node where it should have been,
	and e is the number of edges on the path between the starting node and ending node+1.
	"""

    def search_from_max(self, key: int) -> tuple[AVLNode, int]:
        node = self.max_node()
        dist = 1

        # Go up the tree until the required key can be in the subtree
        while node.parent is not None and key < node.key:
            node = node.parent
            dist += 1

        # Begin the search from the root of the acquired subtree
        return self.search_from(key, node, dist)

    """inserts a new node into the dictionary with corresponding key and value (starting at the root)
    Complexity: O(log(n))

	@type key: int
	@pre: key currently does not appear in the dictionary
	@param key: key of item that is to be inserted to self
	@type val: string
	@param val: the value of the item
	@rtype: (AVLNode,int,int)
	@returns: a 3-tuple (x,e,h) where x is the new node,
	e is the number of edges on the path between the starting node and new node before rebalancing,
	and h is the number of PROMOTE cases during the AVL rebalancing
	"""

    def insert(self, key: int, val) -> tuple[AVLNode, int, int]:
        node = AVLNode(key, val)
        # Add virtual children that may be replaced later
        node.add_virtual_children()

        # Tree empty -> create the root
        if self.t_size == 0:
            node.height = 0
            self.root = node
            self.t_size += 1
            self.max = self.root
            # No travel and no promotions
            return self.root, 0, 0

        # Find the virtual node where the new node need be inserted
        spot, dist = self.search_from(key, self.root)
        p = self.insert_at(spot, node)

        return node, dist, p

    """inserts a new node into the dictionary with corresponding key and value, starting at the max
    Complexity: O(log(n))

	@type key: int
	@pre: key currently does not appear in the dictionary
	@param key: key of item that is to be inserted to self
	@type val: string
	@param val: the value of the item
	@rtype: (AVLNode,int,int)
	@returns: a 3-tuple (x,e,h) where x is the new node,
	e is the number of edges on the path between the starting node and new node before rebalancing,
	and h is the number of PROMOTE cases during the AVL rebalancing
	"""

    def finger_insert(self, key: int, val) -> tuple[AVLNode, int, int]:
        node = AVLNode(key, val)
        # Add virtual children that may be replaced later
        node.add_virtual_children()

        # Tree empty -> create the root
        if self.t_size == 0:
            node.height = 0
            self.root = node
            self.t_size += 1
            self.max = self.root
            # No travel and no promotions
            return self.root, 0, 0

        # Find the virtual node where the new node need be inserted
        spot, dist = self.search_from_max(key)
        p = self.insert_at(spot, node)

        return node, dist, p

    """Insert the given node in the given spot
    Complexity: O(log(n))
    
    @type spot: AVLNode
    @pre spot: is a virtual node
    @param spot: where the node needs to be inserted
    @type node: AVLNode
    @pre node: has virtual children
    @param node: the node to insert into the tree
    @rtype: int
    @returns: number of promotes in the balancing process
    """

    def insert_at(self, spot: AVLNode, node: AVLNode) -> int:
        # Maintain pointer to max node
        if node.key > self.max_node().key:
            self.max = node

        # Update tree size
        self.t_size += 1

        balancing_needed = spot.parent.is_leaf_node()
        node.create_leaf(spot)

        # Simple insertion with non-leaf parent
        if not balancing_needed:
            # No changes => no promotions
            return 0

        return self.rebalance(node.parent)

    """deletes node from the dictionary
    Complexity: O(log(n))

	@type node: AVLNode
	@pre: node is a real pointer to a node in self
	"""

    def delete(self, node: AVLNode):
        # Update tree size
        self.t_size -= 1

        # Only node
        if self.t_size == 0:
            self.root = None
            self.max = None
            return

        # Delete leaf
        if node.is_leaf_node():
            if node.is_right_child():
                node.parent.right = None
            if node.is_left_child():
                node.parent.left = None
            node.parent.add_virtual_children()

        # One child
        elif not node.right.is_real_node() or not node.left.is_real_node():
            child = node.left if node.left.is_real_node() else node.right
            if node.parent is not None:
                node.parent.insert_child(child, node.is_right_child())
            # Update root pointer if needed
            if node == self.root:
                self.root = child
                child.parent = None

        # Two children
        else:
            # Size hasn't changed yet, will change when calling delete below
            self.t_size += 1

            # Find the in-order successor (necessarily in right subtree)
            successor = node.right
            while successor.left.is_real_node():
                successor = successor.left

            # Replace node's key and value with successor's
            node.key = successor.key
            node.value = successor.value

            # Delete the successor, which is guaranteed to have at most one child
            self.delete(successor)

        # Delete max pointer if needed, will be updated the first time max_node() is called
        if node == self.max_node():
            self.max = None

        # Rebalance the tree from the parent of the deleted node upwards
        self.rebalance(node.parent)

    """joins self with item and another AVLTree
    Complexity: O(log(n))

	@type tree2: AVLTree 
	@param tree2: a dictionary to be joined with self
	@type key: int 
	@param key: the key separating self and tree2
	@type val: string
	@param val: the value corresponding to key
	@pre: all keys in self are smaller than key and all keys in tree2 are larger than key,
	or the opposite way
	"""

    def join(self, tree2: "AVLTree", key: int, val):
        # One of them is empty
        if self.root is None:
            tree2.insert(key, val)
            self.tree_from_root(tree2.root)
            self.t_size = tree2.t_size  # Update size
            self.max = tree2.max_node()  # Update max pointer
            return
        if tree2.root is None:
            self.insert(key, val)
            return

        # Update tree size
        self.t_size += tree2.t_size + 1

        # Determine tall and short trees
        tall = self if self.root.height >= tree2.root.height else tree2
        short = tree2 if tall == self else self

        # Find joining point in the taller tree
        target_height = short.root.height
        current = tall.root
        while current.height > target_height:
            current = current.left if tall.root.key > key else current.right
        parent = current.parent

        # Create a new node with the given key and val
        node = AVLNode(key, val)
        node.height = target_height + 1

        # Attach subtrees to the new node
        if tall.root.key > key:
            node.insert_left(short.root)
            node.insert_right(current)
        else:
            node.insert_left(current)
            node.insert_right(short.root)

        # Reattach parent if there was one
        if parent is not None:
            if parent.left == current:
                parent.left = node
            else:
                parent.right = node
            node.parent = parent
            # Update root
            self.root = tall.root

            # Rebalance starting from the parent of the inserted node
            self.rebalance(parent)
        else:
            # Update root
            self.root = node

        # Update max pointer if needed
        if tree2.max_node().key > self.max_node().key:
            self.max = tree2.max_node()

    """splits the dictionary at a given node
    Complexity: O(log(n))

	@type node: AVLNode
	@pre: node is in self
	@param node: the node in the dictionary to be used for the split
	@rtype: (AVLTree, AVLTree)
	@returns: a tuple (left, right), where left is an AVLTree representing the keys in the 
	dictionary smaller than node.key, and right is an AVLTree representing the keys in the 
	dictionary larger than node.key.
	"""

    def split(self, node: AVLNode) -> tuple["AVLTree", "AVLTree"]:
        # Splitting from root is simple
        if node == self.root:
            return AVLTree(node.left), AVLTree(node.right)

        # Initialize two new AVL trees from the node's children
        left_tree = AVLTree(node.left)
        right_tree = AVLTree(node.right)

        # Go up to the root and attach nodes to the left/right trees
        prev = node
        current = node.parent
        while current is not None:
            # If went left, everything above me is smaller
            if prev.is_right_child():
                left_tree.join(AVLTree(current.left), current.key, current.value)
            # If went right, everything above me is bigger
            else:
                right_tree.join(AVLTree(current.right), current.key, current.value)

            prev = current
            current = current.parent

        # Delete max pointers, will be updated the first time max_node() is called
        left_tree.max = None
        right_tree.max = None

        return left_tree, right_tree

    """Rebalance the tree starting from a given node and propagating upward.
    Complexity: O(log(n))

    @type node: AVLNode
    @param node: the node to start rebalancing from
    @returns: an integer p, where p is the number of promotes that happened during rebalancing
    """

    def rebalance(self, node: AVLNode) -> int:
        # Promotions counter
        p = 0

        while node is not None:
            # Update height
            old_height = node.height
            node.calc_height()

            # Calculate balance factor
            bf = node.get_balance_factor()

            # Left-heavy
            if bf > 1:
                if (node.left.left.height) >= (node.left.right.height):
                    self.r_rotate(node)
                else:
                    self.lr_rotate(node)
            # Right-heavy
            elif bf < -1:
                if (node.right.right.height) >= (node.right.left.height):
                    self.l_rotate(node)
                else:
                    self.rl_rotate(node)
            # Promote
            elif old_height < node.height:
                p += 1

            node = node.parent

        return p

    """Rotate the node and its right/left child via left/right rotation
    
    @type node: AVLNode
    @pre node: has a right/left child in respect to the requested rotation
    @param node: the parent node of the two needed for rotation
    @type right: bool
    @param right: True for right rotation, False for left
    """

    def rotate(self, node: AVLNode, right: bool):
        # Handle moving root
        child = node.left if right else node.right
        if node == self.root:
            self.root = child
        # The only subtree that need be moved
        subtree = child.right if right else child.left
        # Rotate
        child.replace_child(node)
        child.insert_child(node, right)
        node.insert_child(subtree, not right)
        # Update heights
        node.calc_height()
        child.calc_height()

    """Rotate the node and its left child via right rotation
    
    @type node: AVLNode
    @pre node: has a left child
    @param node: the parent node of the two needed for rotation
    """

    def r_rotate(self, node: AVLNode):
        self.rotate(node, True)

    """Rotate the node and its right child via left rotation
    
    @type node: AVLNode
    @pre node: has a right child
    @param node: the parent node of the two needed for rotation
    """

    def l_rotate(self, node: AVLNode):
        self.rotate(node, False)

    """Rotate the node's left child and its right child via left rotation,
    and then the node and its left child via right rotation
    
    @type node: AVLNode
    @pre node: has a left child, and it has right child
    @param node: the highest node of the three needed for rotation
    """

    def lr_rotate(self, node: AVLNode):
        self.l_rotate(node.left)
        self.r_rotate(node)

    """Rotate the node's right child and its left child via right rotation,
    and then the node and its right child via left rotation
    
    @type node: AVLNode
    @pre node: has a right child, and it has left child
    @param node: the highest node of the three needed for rotation
    """

    def rl_rotate(self, node: AVLNode):
        self.r_rotate(node.right)
        self.l_rotate(node)

    """returns an array representing dictionary 
    Complexity: O(n)

	@rtype: list
	@returns: a sorted list according to key of tuples (key, value) representing the data structure
	"""

    def avl_to_array(self):
        if self.root is None:
            return []
        # inOrder search will provide a sorted list
        return self.in_order(self.root)

    """Recursively generate a sorted list of (key, value) pairs in a tree
    Complexity: O(n)
    
    @type node: AVLNode
    @param node: node to start the in_order list from
    @rtype: list
	@returns: a sorted list according to key of tuples (key, value) representing the data structure
	"""

    def in_order(self, node: AVLNode) -> list[tuple[int, Any]]:
        if not node.is_real_node():
            return []

        return (
            self.in_order(node.left)
            + [(node.key, node.value)]
            + self.in_order(node.right)
        )

    """returns the node with the maximal key in the dictionary

	@rtype: AVLNode
	@returns: the maximal node, None if the dictionary is empty
	"""

    def max_node(self):
        # Update max pointer if needed (might be None after delete or split)
        if self.max is None and self.root is not None:
            self.max = self.root
            while self.max.right.is_real_node():
                self.max = self.max.right

        return self.max

    """returns the number of items in dictionary 

	@rtype: int
	@returns: the number of items in dictionary 
	"""

    def size(self):
        return self.t_size

    """returns the root of the tree representing the dictionary

	@rtype: AVLNode
	@returns: the root, None if the dictionary is empty
	"""

    def get_root(self):
        return self.root

    # For debugging purposes
    # TODO: Remove before submission
    """Print tree to console"""

    def print_tree(self):
        if self.root is None or not self.root.is_real_node():
            print("The tree is empty.")
            return

        # Level-order traversal to capture nodes at each depth
        current_level = [self.root]
        while any(node.is_real_node() for node in current_level):
            next_level = []
            level_output = []
            for node in current_level:
                if node.is_real_node():
                    level_output.append(f"({node.key}, {node.value}, {node.height})")
                    next_level.append(node.left if node.left else AVLNode())
                    next_level.append(node.right if node.right else AVLNode())
                else:
                    level_output.append("(None)")
                    next_level.append(AVLNode())
                    next_level.append(AVLNode())
            print(" ".join(level_output))
            current_level = next_level
