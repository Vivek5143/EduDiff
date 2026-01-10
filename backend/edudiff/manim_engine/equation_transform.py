from manim import *
from .base_template import BaseTemplate


class EquationTransformScene(BaseTemplate):
    def __init__(self, steps, explanation, **kwargs):
        self.steps = steps
        self.explanation = explanation
        super().__init__(**kwargs)

    def generate_content(self):
        """
        Simple, time-bounded scene:
        - Shows each step as text, then fades it out.
        - No infinite waits, no ambient camera movements.
        - At most one final short wait.
        """
        # Ensure we have an iterable list of steps
        steps = list(self.steps or [])

        # Fallback: if no steps but an explanation exists, show the explanation as a single step
        if not steps and self.explanation:
            steps = [self.explanation]

        for step in steps:
            text = Text(str(step), font_size=36)
            self.play(Write(text), run_time=1)
            self.play(FadeOut(text), run_time=0.5)

        # Single short final pause to avoid abrupt ending
        self.wait(0.5)
