def hello(name: str = None) -> str:
    if name is None or name == '':
        return 'Hello!'
    return f'Hello, {name}!'

def int_to_roman(num: int) -> str:
    symbols = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    roman_numeral = ''
    
    for i in range(len(values)):
        while num >= values[i]:
            roman_numeral += symbols[i]
            num -= values[i]
    
    return roman_numeral

def longest_common_prefix(strs: list) -> str:
    if not strs:
        return ""

    stripped_strs = [s.strip() for s in strs]
    min_str = min(stripped_strs, key=len)
    
    for i in range(len(min_str)):
        for s in stripped_strs:
            if s[i] != min_str[i]:
                return min_str[:i]
    
    return min_str

class BankCard:
    def __init__(self, total_sum, balance_limit=None):
        self.total_sum = total_sum
        self.balance_limit = balance_limit if balance_limit is not None else float('inf')
        self.balance_checks = 0

    def __call__(self, sum_spent):
        if sum_spent > self.total_sum:
            raise ValueError(f"Not enough money to spend {sum_spent} dollars.")

        self.total_sum -= sum_spent
        print(f"You spent {sum_spent} dollars.")

    def __str__(self):
        return "To learn the balance call balance."

    @property
    def balance(self):
        if self.balance_limit == 0:
            raise ValueError("Balance check limits exceeded.")

        self.balance_checks += 1
        if self.balance_limit is not None:
            self.balance_limit -= 1

        return self.total_sum

    def put(self, sum_put):
        self.total_sum += sum_put
        print(f"You put {sum_put} dollars.")

    def __add__(self, other):
        if not isinstance(other, BankCard):
            return NotImplemented

        new_total_sum = self.total_sum + other.total_sum
        new_balance_limit = max(
            self.balance_limit if self.balance_limit is not None else 0,
            other.balance_limit if other.balance_limit is not None else 0
        )

        return BankCard(new_total_sum, new_balance_limit)

def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5

    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6

    return True

def primes() -> int:
    yield 2
    n = 3
    while True:
        if is_prime(n):
            yield n
        n += 2