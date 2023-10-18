def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        fib_list = [0, 1]
        for i in range(2, n):
            next_value = fib_list[i - 1] + fib_list[i - 2]
            fib_list.append(next_value)
        return fib_list

if __name__ == "__main__":
    fib_sequence = fibonacci(10)  # Generate first 10 numbers of the Fibonacci sequence
    for index, value in enumerate(fib_sequence):
        print(f"Term {index+1}: {value}")
