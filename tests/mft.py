from checklist.test_suite import TestSuite
from checklist.test_types import MFT
from checklist.perturb import Perturb
from checklist.expect import Expect

# Import your model's prediction function
from src.model import predict

# Create a test suite
suite = TestSuite()

# Define templates with placeholders for populations
population_templates = (
    'The study focused on {population}.',
    'Results were most significant among {population}.',
    'The impact on {population} was noteworthy.',
    'Interventions were targeted towards {population}.',
    'Data was collected from various {population}.',
    'The {population} showed a remarkable response.',
    'Surveys were conducted across different {population}.',
    'The {population} was observed for changes.',
    'A significant change was recorded in the {population}.',
    'The research aimed to benefit the {population}.'
)

# Define templates with placeholders for interventions
intervention_templates = (
    "The {intervention} was implemented to address the issue.",
    "Researchers studied the effects of the {intervention}.",
    "The {intervention} had a significant impact on the community.",
    "Funding was provided for the {intervention}.",
    "The success of the {intervention} was evident in the results.",
    "Participants were selected for the {intervention} group.",
    "The {intervention} was a key part of the strategy.",
    "The {intervention} targeted specific outcomes.",
    "Outcomes were measured after the {intervention} took place.",
    "The {intervention} was designed to improve overall outcomes."
)

# Define templates with placeholders for outcomes
outcome_templates = (
    "The outcome of the study was {outcome}.",
    "It was observed that the primary outcome was {outcome}.",
    "The expected outcome was {outcome}, which was surprising.",
    "As a result, the outcome was {outcome}.",
    "The final outcome, {outcome}, was recorded after the experiment.",
    "The result of the intervention was {outcome}.",
    "The project led to {outcome}.",
    "The consequences were observed as {outcome}.",
    "The end effect was {outcome}.",
    "The study concluded with {outcome}."
)

# Define templates with placeholders for coreferences
coreference_templates = (
    "This refers to the {coreference}.",
    "Such instances of {coreference} were noted.",
    "As mentioned earlier, the {coreference} plays a crucial role.",
    "This is similar to the {coreference} discussed before.",
    "The case of {coreference} is particularly interesting.",
    "In light of the {coreference}, further analysis is required.",
    "This aligns with the {coreference} we observed.",
    "The {coreference} under discussion was pivotal.",
    "Reflecting on the {coreference}, it becomes clear.",
    "Given the {coreference}, the results are unsurprising."
)

# Define templates with placeholders for effect_sizes
effect_size_templates = (
    "The observed change was {effect_size}.",
    "A {effect_size} increase was noted in the study.",
    "The results showed a {effect_size} decrease.",
    "There was a {effect_size} improvement over the baseline.",
    "The effect was quantified as {effect_size}.",
    "The magnitude of impact measured {effect_size}.",
    "The statistical significance reached {effect_size}.",
    "A {effect_size} reduction in errors was achieved.",
    "The intervention led to a {effect_size} enhancement.",
    "The data indicated a {effect_size} growth rate."
)

# Define MFTs
mft_tests = [
    MFT(**{
        'name': 'Test for correct population recognition',
        'capability': 'NER',
        'description': 'The model should correctly identify and label populations.',
        'data': [
            ('Input text 1', 'Expected label 1'),
            ('Input text 2', 'Expected label 2'),
            # Add more test cases
        ],
        'labels': ['Label 1', 'Label 2'],  # Replace with your actual labels
    }),
    # Add more MFTs as needed
]

# Add MFTs to the test suite
for test in mft_tests:
    suite.add(test)

# Run the test suite
suite.run(predict)