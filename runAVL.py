from AVLTree import AVLTree

t = AVLTree()
t.finger_insert(50, 1)
t.finger_insert(40, 2)
t.finger_insert(60, 3)
t.finger_insert(30, 4)
t.finger_insert(70, 5)
t.finger_insert(55, 6)
t.finger_insert(20, 7)
t.finger_insert(45, 8)
t.finger_insert(46, 9)
t.print_tree()
t.finger_insert(44, 10)

t.print_tree()
# print(t.avl_to_array())
# print(t.max_node())
# print(t.size())
