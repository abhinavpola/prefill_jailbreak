# Demonstrating risks from Anthropic API allowing prefill

Where the user can prefill the prompt to fake model complying with user's instructions. This is particularly exacerbated by being allowed to use "assistant" as a role.

## Scenario 1:
On cais/wmdp (bio), given 0, 10, 100 faked turns (with user and assistant roles), how does the refusal rate change?

Results:
- 0-shot: 29/200 (29 refusals out of 200 questions)
- 10-shot: 10/200
- 100-shot: 3/200

## Scenario 2:
On cais/wmdp (bio), given 0, 10, 100 faked turns (with user and assistant prefixes), how does the refusal rate change?

Results:
- 0-shot: 31/200
- 10-shot: 172/200
- 100-shot: 200/200

## Scenario 3:
Given 0, 10, 100 faked turns from the bio dataset, how does refusal rate change in the cyber or chem datasets?

Results:
- 10-shot (cyber examples, bio question): 17/200
- 100-shot: TODO