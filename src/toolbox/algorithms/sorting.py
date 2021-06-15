from typing import Optional


def swap(a, i: int, j: int) -> None:
    swap_buf = a[i]
    a[i] = a[j]
    a[j] = swap_buf


def quicksort_lomuto(a, lo: int = 0, hi: Optional[int] = None) -> None:
    """Sorts the array `a` from `lo` to `hi` in place using Lomuto partitioning.

    Args:
        a (array-like): The array
        lo (int): The starting index of the section to be partitionned
        hi (int): The ending index of the section to be partitionned

    This partitioning scheme always selects the last element in the array as a
    pivot. It then runs through the array with two cursors `i` and `j`. `i` is
    initialized to `lo`. `j` runs through the array from `lo` to `hi`. When a
    value at index `j` is found to be smaller than the pivot, it is swapped with
    the value at index `i` and `i` is incremented by one. Finally, the pivot is
    swapped with the value at index `i`. Once these operations have run their
    course, it is certain that all values to the left of the pivot (which is
    now at index `i`) are smaller than the pivot and that all values to the
    left are larger than the pivot. The index `i` serves as the
    partitioning (split) point for the next iteration of the sorting
    algorithm, at which point the array is split into the left and right parts
    of `a` and each part is partitionned again until the array is sorted.

    $O(n^{2})$ worst-case when A is already in sorted order.

    | | |
    | --- | --- |
    | **Comparison** | Yes |
    | **Inplace** | Yes |
    | **Stable** | No |

    Examples:

    ```
        [5, 4, 3, 6, 2, 1, 9, 4, 0]
        ---------------------------
    1 - [5, 4, 3, 6, 2, 1, 9, 4, 0] - [0, 4, 3, 6, 2, 1, 9, 4, 5]
    2 - [4, 3, 2, 1, 4, 5, 6, 9]    - [0, 4, 3, 2, 1, 4, 5, 6, 9]
    3 - [3, 2, 1, 4, 4]             - [0, 3, 2, 1, 4, 4, 5, 6, 9]
    4 - [1, 2, 3]                   - [0, 1, 2, 3, 4, 4, 5, 6, 9]
    5 - [2, 3]                      - [0, 1, 2, 3, 4, 4, 5, 6, 9]
    6 - [6, 9]                      - [0, 1, 2, 3, 4, 4, 5, 6, 9]
        ---------------------------
        [0, 1, 2, 3, 4, 4, 5, 6, 9]
    ```
    """
    n = len(a)
    if lo < 0:
        raise ValueError("lo must be non-negative")
    if hi is None:
        hi = n - 1
    elif hi >= n:
        raise ValueError("hi must be within the index range")
    if lo >= hi:
        return
    p = partition_lomuto(a, lo, hi)
    quicksort_lomuto(a, lo, p - 1)
    quicksort_lomuto(a, p + 1, hi)


def partition_lomuto(a, lo: int, hi: int) -> int:
    """Lomuto partitioning.

    Args:
        a (array-like): The list which is being sorted
        lo (int): The starting index of the section to be partitionned
        hi (int): The ending index of the section to be partitionned

    Returns:
        int: The partition index
    """
    pivot = a[hi]
    i = lo
    for j in range(lo, hi + 1):
        if a[j] < pivot:
            swap(a, i, j)
            i += 1
    swap(a, i, hi)
    return i


def quicksort_hoare(a, lo: int = 0, hi: Optional[int] = None) -> None:
    """Sorts the array `a` from `lo` to `hi` in place using Hoare partitioning.

    Args:
        a (array-like): The array
        lo (int): The starting index of the section to be partitionned
        hi (int): The ending index of the section to be partitionned

    $O(n^{2})$ worst-case when A is already in sorted order.

    | | |
    | --- | --- |
    | **Comparison** | Yes |
    | **Inplace** | Yes |
    | **Stable** | No |

    Examples:

    ```
        [5, 4, 3, 6, 2, 1, 9, 4, 0]
        ---------------------------
    1 - [5, 4, 3, 6, 2, 1, 9, 4, 0] - [0, 1, 2, 6, 3, 4, 9, 4, 5]
    2 - [0, 1, 2]                   - [0, 1, 2, 6, 3, 4, 9, 4, 5]
    3 - [6, 3, 4, 9, 4, 5]          - [0, 1, 2, 4, 3, 4, 9, 6, 5]
    4 - [4, 3, 4]                   - [0, 1, 2, 3, 4, 4, 9, 6, 5]
    5 - [9, 6, 5]                   - [0, 1, 2, 3, 4, 4, 5, 6, 9]
        ---------------------------
        [0, 1, 2, 3, 4, 4, 5, 6, 9]
    ```
    """
    n = len(a)
    if lo < 0:
        raise ValueError("lo must be non-negative")
    if hi is None:
        hi = n - 1
    elif hi >= n:
        raise ValueError("hi must be within the index range")
    if (hi - lo) < 2:
        return
    p = partition_hoare(a, lo, hi)
    quicksort_hoare(a, lo, p)
    quicksort_hoare(a, p + 1, hi)


def partition_hoare(a, lo: int, hi: int) -> int:
    """Hoare partitioning.

    Args:
        a (array-like): The list which is being sorted
        lo (int): The starting index of the section to be partitionned
        hi (int): The ending index of the section to be partitionned

    Returns:
        int: The partition index
    """
    pivot = a[(lo + hi) // 2]
    i = lo - 1
    j = hi + 1
    while True:
        while True:
            i += 1
            if a[i] >= pivot:
                break
        while True:
            j -= 1
            if a[j] <= pivot:
                break
        if i >= j:
            print(a)
            return j
        swap(a, i, j)
