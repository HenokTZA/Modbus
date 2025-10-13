from argon2 import PasswordHasher
import sys
if len(sys.argv) != 2:
    print("Usage: python3 make_hash.py <plaintext>")
    raise SystemExit(2)
print(PasswordHasher().hash(sys.argv[1]))
