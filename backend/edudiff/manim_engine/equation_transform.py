from manim import *
from .base_template import BaseTemplate

class EquationTransformScene(BaseTemplate):
    def __init__(self, steps, explanation, **kwargs):
        self.steps = steps
        self.explanation = explanation
        super().__init__(**kwargs)

    def generate_content(self):
        # Initial equation
        current_eq = MathTex(self.steps[0], color=BLACK).scale(1.5)
        self.play(Write(current_eq))
        self.wait(1)

        # Explanation text at the bottom
        explanation_text = Text(self.explanation, font_size=24, color=BLACK).to_edge(DOWN)
        self.play(Write(explanation_text))
        self.wait(1)

        # Iterate through steps
        for i in range(1, len(self.steps)):
            next_eq = MathTex(self.steps[i], color=BLACK).scale(1.5)
            
            # Highlight changes (simple approach: highlight the whole new equation for now, 
            # or we could try to find differences if we had more granular data)
            # For this MVP, we'll just transform and highlight the result.
            
            self.play(TransformMatchingTex(current_eq, next_eq))
            self.wait(0.5)
            
            # Highlight with rectangle
            rect = SurroundingRectangle(next_eq, color=BLUE, buff=0.2)
            self.play(Create(rect))
            self.wait(1)
            self.play(FadeOut(rect))
            
            current_eq = next_eq
            self.wait(1)

        self.wait(2)
