# Gaussian-map skeleton experiment

Each of POS and NEG trains four independent models at sigma 1/2/3/4 mm. Each map is a 2-D physical-distance Gaussian truncated at 3 sigma; it remains continuous through resize and SAM2 mask conditioning. Training uses random 30–70% axial prompt slices and random contiguous skeleton segments; validation/test use deterministic 50% axial selection and a central 50% segment. The training and validation direction/window protocol is identical to binary_mask.

Validation selects each fixed model epoch, then m, then sigma. Only the final POS and NEG validation locks enter the 36-case test/fusion at 0/25/50/75/100% axial prompt fractions.
