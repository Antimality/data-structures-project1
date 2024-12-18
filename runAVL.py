from AVLTree import AVLTree

t = AVLTree()
print(t.insert(50, 1)[1])
print(t.insert(40, 2)[1])
print(t.insert(60, 3)[1])
print(t.insert(30, 4)[1])
print(t.insert(70, 5)[1])
print(t.insert(55, 6)[1])
print(t.insert(20, 7)[1])
print(t.insert(45, 8)[1])
print(t.insert(46, 9)[1])
print(t.insert(44, 10)[1])
print(t.search(7))

# t.print_tree()
# print(t.avl_to_array())
# print(t.max_node())
# print(t.size())
