"""CPU arithmetic oracle for E1; supplements the actual-policy HIP host test.

No GPU, HIP emulation package or external runtime is used. Float64 exactly holds
the dyadic FP16 product and add here; packing to 'e' performs RN16. This does not
qualify compiler lowering or native ISA execution.
"""
import struct


def half(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def rn16(value):
    return struct.unpack("<H", struct.pack("<e", value))[0]


def hash16(state):
    # Independent instruction semantics: packed low-u16 multiplies, then MAD.
    adjusted = 0x83DCD12C
    low = state & 0xFFFF
    high = (low * (adjusted >> 16)) & 0xFFFF
    packed = low | (high << 16)
    return ((packed & 0xFFFF) * (adjusted & 0xFFFF) + packed) & 0xFFFFFFFF


def byte_sum(product):
    return sum(product.to_bytes(4, "little"))


def codebook_value(state):
    product = (state * 0x83DCD12D) & 0xFFFFFFFF
    # This product and sum are exactly representable in FP32 as well as FP64.
    return rn16((1024 + byte_sum(product)) * half(0x1EEE) + half(0xC931))


def decode8(states):
    out = []
    for i in range(0, 8, 2):
        pair = [half(0x6400 + byte_sum(hash16(s))) for s in states[i:i + 2]]
        out.extend(rn16(x * half(0x1EEE) + half(0xC931)) for x in pair)
    return out


def main():
    # K3..K6 differ in extraction, not in the MUL1 map; exercise each policy.
    for bits in range(3, 7):
        for base in range(65536):
            states = [(base + i * 8191) & 0xFFFF for i in range(8)]
            expected = [codebook_value(s) for s in states]
            if decode8(states) != expected or decode8([s | 0xA5A50000 for s in states]) != expected:
                raise AssertionError(f"MISMATCH K{bits} base={base}: FAST-only")
        print(f"EXACT arithmetic oracle K{bits}: 65536 states, all eight lanes")


if __name__ == "__main__":
    main()
