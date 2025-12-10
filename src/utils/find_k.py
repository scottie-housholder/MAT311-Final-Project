def find_k(size: int) -> int:
    """
    Return the number of neighbors a KNN algorithm should have
    Returns the square root of the given number and makes sure it's odd
    """

    k = round(size**0.5)
    if k % 2 == 0:
        return k-1
    return k