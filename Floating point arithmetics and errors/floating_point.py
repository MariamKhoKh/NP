# 1
a = 0.6 + 0.7
print(a == 1.3)  # Expecting True, but it prints False
print(a)  # Prints 1.2999999999999998




# 2
a = 1e30
b = -1e30
c = 1

result1 = (a + b) + c
result2 = a + (b + c)

print("result1:", result1)
print("result2:", result2)
print("Are they equal?", result1 == result2)




# 3
import math

try:
    result = math.exp(1000)  # Large exponent causes overflow
except OverflowError as e:
    print("Overflow error:", e)




# 4
small_value = 1e-300
underflowed = small_value / 1e100  # Divides by a very large number

print(underflowed)  # Result may be zero due to underflow



