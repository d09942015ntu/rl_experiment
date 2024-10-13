import csv
import numpy as np
import random
def addition_steps(a, b, nm1=3):
    # Convert numbers to digits
    digits_a = [int(d) for d in str(a)]
    digits_b = [int(d) for d in str(b)]

    # Ensure both have length 4 (since we are dealing with 4-digit numbers)
    # Pad with zeros if necessary
    while len(digits_a) < nm1:
        digits_a = [0] + digits_a
    while len(digits_b) < nm1:
        digits_b = [0] + digits_b

    carry = 0
    result_digits = []
    states = []

    # Initial state
    state_str = ' '.join(map(lambda x:str(x[0])+str(x[1]), zip(digits_a,digits_b))) + ' x'
    states.append(state_str)

    for i in range(nm1 - 1, -1, -1):  # Indices from 3 to 0
        sum_digit = digits_a[i] + digits_b[i] + carry
        result_digit = sum_digit % 10
        carry = sum_digit // 10

        result_digits.insert(0, result_digit)

        # Build the state string
        state_str = ' '.join(map(lambda x:str(x[0])+str(x[1]), zip(digits_a[:i],digits_b[:i]))) \
                     + str(carry)+ ' x ' + ' '.join(map(str, result_digits))
        states.append(state_str)

    # After processing all digits, if carry > 0, we need to handle it
    if carry > 0:
        result_digits.insert(0, carry)
        # Build the final state
    state_str = 'x ' + ' '.join(map(str, result_digits))
    states.append(state_str)

    return states





def save_dataset(used,n,seed=0):
    filename = f'data/data_{n}_{seed}.csv'
    nm1 = n-1
    rng = np.random.RandomState(seed)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header to the CSV

        writer.writerow([f's{i}' for i in range(nm1+2)])
        # Call the addition_steps function 1000 times and write results_41 to CSV
        lower_int = 10**(nm1-1)
        upper_int = 10**nm1
        i = 0
        while i< int((upper_int**2)*0.05):
            num1 = rng.randint(lower_int, upper_int)
            num2 = rng.randint(lower_int, upper_int)
            ikey = (num1,num2)
            if ikey in used:
                continue
            i += 1
            used.add((num1,num2))
            steps = addition_steps(num1, num2)  # Replace with desired values or logic
            print(steps)
            num_result = int(steps[-1].replace("x","").replace(" ",""))
            assert num1 + num2 == num_result
            writer.writerow(steps)


if __name__ == '__main__':
    used = set()
    save_dataset(used,4,0)
    save_dataset(used,4,1)
    save_dataset(used,4,2)
    print(used)
