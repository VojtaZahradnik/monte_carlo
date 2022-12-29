class MersenneTwister:
    """
    Implementation of Mersenne Twister for random generating number
    """
    def __init__(self, low= 0, high= 1) -> None:
        """
        Initial setup of constants for MT
        :param low: Bottom boundary
        :param high: Upper boundary
        """
        # Constants for MT19937
        self.w, self.n, self.m, self.r = 32, 624, 397, 31
        self.a = 0x9908B0DF
        self.u, self.d = 11, 0xFFFFFFFF
        self.s, self.b = 7, 0x9D2C5680
        self.t, self.c = 15, 0xEFC60000
        self.l, self.f = 18, 1812433253

        self.low, self.high = low, high

        self.MT = [0] * self.n
        self.index = self.n + 1
        self.lower_mask = (1 << self.r) - 1
        self.upper_mask = (1 << self.r)

    def set_seed(self, seed: int):
        """
        Setting seed for generator
        :param seed: Given seed for generator, time in miliseconds
        """
        self.index = self.n
        self.MT[0] = seed
        for i in range(1, self.n):
            self.MT[i] = (self.f * (self.MT[i - 1] ^ (self.MT[i - 1] >> (self.w - 2))) + i) & 0xFFFFFFFF

    def extract_number(self) -> int:
        """
        Generating random number based on bite move
        :return: Random number
        """
        if self.index >= self.n:
            if self.index > self.n:
                raise Exception("Generator not exists")
            self.twist()

        y = self.MT[self.index]
        y = y ^ ((y >> self.u) & self.d)
        y = y ^ ((y << self.s) & self.b)
        y = y ^ ((y << self.t) & self.c)
        y = y ^ (y >> self.l)

        self.index = self.index + 1
        return y & 0xFFFFFFFF

    def get_uniform(self, low: int, high: int) -> float:
        """
        Returning number based on low and high interval
        :param low: Bottom boundary for returned number
        :param high: Upper boundary for returned number
        :return:
        """
        self.low = low
        self.high = high
        return self.low + (self.high - self.low) * (self.extract_number() / (2 ** 32))

    def twist(self) -> None:
        """
        Twisting based on MT definition
        """
        for i in range(self.n):
            x = (self.MT[i] & self.upper_mask) + (self.MT[(i + 1) % self.n] & self.lower_mask)
            xA = x >> 1
            if (x % 2) != 0:
                xA = xA^self.a
            self.MT[i] = self.MT[(i + self.m) % self.n]^xA
        self.index = 0


