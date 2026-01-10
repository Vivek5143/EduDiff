from manim import *

class EducationalScene(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        
        title = Text("Pythagorean Theorem", font_size=48, color=BLACK)
        title.to_edge(UP)
        
        equation = MathTex("a^2 + b^2 = c^2", color=BLACK)
        equation.scale(1.5)
        
        a_label = MathTex("a", color=BLUE)
        a_label.scale(0.75)
        b_label = MathTex("b", color=GREEN)
        b_label.scale(0.75)
        c_label = MathTex("c", color=RED)
        c_label.scale(0.75)
        
        square_a = Square(side_length=1.5, color=BLUE, fill_opacity=0.3)
        square_b = Square(side_length=1.5, color=GREEN, fill_opacity=0.3)
        square_c = Square(side_length=1.5, color=RED, fill_opacity=0.3)
        
        square_a.move_to(LEFT * 3 + DOWN * 1.5)
        square_b.move_to(ORIGIN + DOWN * 1.5)
        square_c.move_to(RIGHT * 3 + DOWN * 1.5)
        
        a_label.next_to(square_a, DOWN)
        b_label.next_to(square_b, DOWN)
        c_label.next_to(square_c, DOWN)
        
        self.play(Write(title), run_time=1)
        self.wait(0.5)
        self.play(Write(equation), run_time=1.5)
        self.wait(0.5)
        self.play(
            Create(square_a),
            Create(square_b),
            Create(square_c),
            Write(a_label),
            Write(b_label),
            Write(c_label),
            run_time=2
        )
        self.wait(2)

