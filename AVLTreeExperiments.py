import time
import random
from AVLTree import AVLTree  # ודא שהקובץ AVLTree.py מכיל את המימוש של AVL Tree


def create_sorted_array(n):
    """יוצר מערך מסודר בסדר עולה."""
    return list(range(1, n + 1))


def create_reverse_sorted_array(n):
    """יוצר מערך מסודר בסדר יורד."""
    return list(range(n, 0, -1))


def create_random_array(n):
    """יוצר מערך אקראי."""
    arr = list(range(n))
    random.shuffle(arr)
    return arr


def create_slightly_random_array(n):
    """יוצר מערך עם החלפות אקראיות סמוכות."""
    arr = create_sorted_array(n)
    for i in range(n - 1):
        if random.random() < 0.5:
            arr[i], arr[i + 1] = arr[i + 1], arr[i]
    return arr


def calculate_inversions(arr):
    """מחשב את מספר ההיפוכים במערך."""
    inversions = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inversions += 1
    return inversions


def run_experiment(tree: AVLTree, arr):
    """מריץ ניסוי על עץ AVL עם מערך נתון."""
    balance_cost = 0
    search_cost = 0
    start_time = time.time()
    for key in arr:
        node, dist, prom = tree.finger_insert(key, 0)
        balance_cost += prom
        search_cost += dist
    end_time = time.time()

    # השתמש בשורש העץ כארגומנט לשיטת in_order
    sorted_list = tree.in_order(tree.root)
    assert sorted_list == sorted(sorted_list, key=lambda x: x[0])

    return {
        "balance_cost": balance_cost,
        "search_cost": search_cost,
        "time_taken": (end_time - start_time) * 1000,  # זמן במילישניות
        # "sorted_correctly": sorted_list == sorted(sorted_list),
    }


def average_experiment(n, generator, num_trials=20):
    """ממוצע תוצאות הניסוי על מערכים אקראיים."""
    total_balance_cost = 0
    total_search_cost = 0
    total_time = 0
    # total_sorted_correctly = 0
    # total_inversions = 0

    for trial in range(num_trials):
        arr = generator(n)

        tree = AVLTree()  # יצירת עץ חדש לכל ניסוי
        result = run_experiment(tree, arr)

        # inversions = calculate_inversions(arr)  # חשב את מספר ההיפוכים עבור המערך
        # total_inversions += inversions

        total_balance_cost += result["balance_cost"]
        total_search_cost += result["search_cost"]
        total_time += result["time_taken"]
        # total_sorted_correctly += result["sorted_correctly"]

    # average_inversions = total_inversions / num_trials

    return {
        "balance_cost": total_balance_cost / num_trials,
        "search_cost": total_search_cost / num_trials,
        "time_taken": total_time / num_trials,
        # "sorted_correctly": total_sorted_correctly / num_trials,
        # "average_inversions": average_inversions,
    }


# def calculate_average_search_cost(n, generator, search_func):
#     total_cost = 0
#     arr = generator(n)
#     for value in arr:
#         node, dist = search_func(value)
#         total_cost += dist
#     average_cost = total_cost / n
#     return average_cost


def q1():
    sizes = [111 * (2**i) for i in range(1, 11)]
    for n in sizes:
        print("========================================")
        print(f"Running experiments for array size {n}...")
        for create_array in [
            create_sorted_array,
            create_reverse_sorted_array,
            create_random_array,
            create_slightly_random_array,
        ]:
            print(f"Testing with {create_array.__name__}...")
            print(average_experiment(n, create_array)["balance_cost"])


def q2():
    sizes = [111 * (2**i) for i in range(1, 11)]
    for n in sizes:
        print("========================================")
        print(f"Running experiments for array size {n}...")
        for create_array in [
            create_sorted_array,
            create_reverse_sorted_array,
        ]:
            print(f"Testing with {create_array.__name__}...")
            print(calculate_inversions(create_array(n)))

        for create_array in [
            create_random_array,
            create_slightly_random_array,
        ]:
            print(f"Testing with {create_array.__name__} (average)...")
            total_inv = 0
            for i in range(20):
                total_inv += calculate_inversions(create_array(n))
            print(total_inv / 20)


def q3():
    sizes = [111 * (2**i) for i in range(1, 11)]
    for n in sizes:
        print("========================================")
        print(f"Running experiments for array size {n}...")
        for create_array in [
            create_sorted_array,
            create_reverse_sorted_array,
            create_random_array,
            create_slightly_random_array,
        ]:
            print(f"Testing with {create_array.__name__}...")
            print(average_experiment(n, create_array)["search_cost"])


# def main():
#  sizes = [111 * (i ** 2) for i in range(1, 11)]  # i from 1 to 10, array size is 111 * i^2
# experiments = {
#    "sorted": create_sorted_array,
#   "reverse_sorted": create_reverse_sorted_array,
#  "random": create_random_array,
# "slightly_random": create_slightly_random_array,
# }

# for n in sizes:
#    print(f"Running experiments for array size {n}...")
#   for name, generator in experiments.items():
#      if name == "random":
#         result = average_experiment(n, generator)
#    else:
#       arr = generator(n)
#      original_arr = arr.copy()
#     tree = AVLTree()  # יצירת עץ חדש לכל ניסוי
#    result = run_experiment(tree, arr)

#   print(f"Experiment: {name}")
#  if name == "random":
#     print(f"Average number of inversions: {result['average_inversions']}")
# else:
# print(f"Number of inversions: {calculate_inversions(arr)}")
#  print(f"Balance cost: {result['balance_cost']}")
#  print(f"Time taken: {result['time_taken']} ms")
# print(f"Sorted correctly: {result['sorted_correctly']}")
#  print("-" * 40)

if __name__ == "__main__":
    q3()
