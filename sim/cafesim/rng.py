"""
Random number generators for the simulator.

Why not just use one generator for everything?
If the menu code and the arrivals code shared one generator, then adding a
single extra random draw to the menu would shift every number the arrivals
code receives afterwards. Your customer counts would change even though
you never touched the arrivals code, which makes experiments confusing.

So instead, every module gets its OWN generator ("stream"). Each stream is
built from the master seed plus the module's name, so:
  - the same seed always gives the same data (reproducible), and
  - changing one module never disturbs another module's randomness.
"""
import zlib

import numpy as np


class RNGStreams:
    def __init__(self, seed: int):
        self.seed = seed
        # Generators we've already created, stored by name so we reuse them.
        self._cache: dict[str, np.random.Generator] = {}

    def get(self, name: str) -> np.random.Generator:
        """Return the random generator for a given module, e.g. get("arrivals")."""
        if name not in self._cache:
            # Turn the module's name into a fixed number. crc32 always gives
            # the same number for the same text, on any computer.
            key = zlib.crc32(name.encode())

            # Combine the master seed with that number to create a unique,
            # repeatable starting point for this module's randomness.
            seq = np.random.SeedSequence(self.seed, spawn_key=(key,))
            self._cache[name] = np.random.default_rng(seq)

        return self._cache[name]