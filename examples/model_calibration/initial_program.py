"""A model hypothesis for the synthetic polynomial example.

The trusted evaluator reads the literal specification and never imports this
file. This deliberately small example is not a model-class restriction in the
framework; real campaigns provide their own model interfaces and evaluators.
"""

# EVOLVE-BLOCK-START
MODEL_SPEC = {
    "degree": 1,
    "hypothesis": "A constant slope explains the response over the observed range.",
}
# EVOLVE-BLOCK-END
