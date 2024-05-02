from manim import *


class Cosine(Scene):
    def contruct(self):
        fomula = MathTex(r"\text{cosine similarity} = \frac{{A \cdot B}}{{\|A\| \|B\|}}").set_color(BLUE)
        text = Tex("hello")
        
        self.add(text)