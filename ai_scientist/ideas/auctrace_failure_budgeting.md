# Title: Failure-Aware Experiment Budgeting for Trustworthy Automated Research

## Keywords
automated research, reliability controller, failure taxonomy, stop-loss, experiment budgeting

## TL;DR
How can an autonomous research system decide when to keep exploring, when to debug, and when to stop, so that failed experiments become useful evidence instead of wasted budget?

## Abstract
Autonomous research systems can already generate ideas, write code, run experiments, and draft papers, but they still behave unreliably when experiments fail repeatedly. In many cases, infrastructure failures, implementation bugs, and scientifically unpromising directions are mixed together, leading the system to spend too much budget on unproductive retry loops. This small research direction asks whether a failure-aware experiment budgeting strategy can improve the trustworthiness of automated research. The central question is not how to make the system generate more aggressively, but how to make it decide more responsibly: when to continue, when to debug, when to switch direction, and when to stop. We are interested in lightweight methods for classifying failure modes, attaching explicit evidence to each failed run, and using those signals to reduce wasted experimentation while preserving useful exploration. The ideal outcome is a system that produces clearer run histories, more defensible claims, and fewer misleading success stories caused by uncontrolled retry behavior.
